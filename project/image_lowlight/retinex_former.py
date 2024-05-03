import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import todos
import pdb


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fn = FeedForward(dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    def __init__(self, dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape # (1, 400, 600, 40)
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x) # size() -- [1, 240000, 40]
        k_inp = self.to_k(x) # size() -- [1, 240000, 40]
        v_inp = self.to_v(x) # size() -- [1, 240000, 40]
        illu_fea = illu_fea_trans.flatten(1, 2) # size() [1, 400, 600, 40] --> [1, 240000, 40]

        q = q_inp.reshape(b, self.num_heads, h*w, c//self.num_heads) # [1, 60000, 80] ==> [1, 2, 60000, 40]
        k = k_inp.reshape(b, self.num_heads, h*w, c//self.num_heads)
        v = v_inp.reshape(b, self.num_heads, h*w, c//self.num_heads)
        illu_attn = illu_fea.reshape(b, self.num_heads, h*w, c//self.num_heads)

        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(2, 3) # [1, 1, 240000, 40] ==> [1, 1, 40, 240000]
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        q = F.normalize(q, dim=3, p=2.0)
        k = F.normalize(k, dim=3, p=2.0)
        attn = (k @ q.transpose(2, 3))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim,
        dim_head=64,
        heads=8,
        num_blocks=2, # 2 or 1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim)
            ]))

    def forward(self, x, illu_fea):
        # tensor [x] size: [1, 40, 400, 600], min: -0.533583, max: 0.779429, mean: 0.017096
        # tensor [illu_fea] size: [1, 40, 400, 600], min: -1.862743, max: 4.447268, mean: 0.038667
        x = x.permute(0, 2, 3, 1)
        for i, (attn, ff) in enumerate(self.blocks): # torch.jit.script not support !!!
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)

        # tensor [out] size: [1, 40, 400, 600], min: -0.763899, max: 0.866437, mean: 0.010528
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super().__init__()
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False), # FeaDownSample
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # IlluFeaDownsample
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0), # FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False), # Fution
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, illu_fea):
        # tensor [x] size: [1, 3, 400, 600], min: 0.0, max: 0.821723, mean: 0.229827
        # tensor [illu_fea] size: [1, 40, 400, 600], min: -1.862743, max: 4.447268, mean: 0.038667

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)

            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # illu_fea_list is list: len = 2
        #     tensor [item] size: [1, 40, 400, 600], min: -1.862743, max: 4.447268, mean: 0.038667
        #     tensor [item] size: [1, 80, 200, 300], min: -6.597843, max: 7.82939, mean: 0.084403

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x
        # tensor [out] size: [1, 3, 400, 600], min: -0.028076, max: 1.041572, mean: 0.400492
        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2]):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
                        in_dim=in_channels, 
                        out_dim=out_channels, 
                        dim=n_feat, 
                        level=level, 
                        num_blocks=num_blocks,
                    )
    
    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 4
        # GPU -- 9G, 360 ms

        modules_body = [RetinexFormer_Single_Stage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                n_feat=n_feat, 
                level=2, 
                num_blocks=num_blocks)
            for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

        self.load_weights()

    def forward(self, x):
        B, C, H, W = x.size()

        pad_h = self.MAX_TIMES - (H % self.MAX_TIMES)
        pad_w = self.MAX_TIMES - (W % self.MAX_TIMES)
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        out = self.body(x)

        return out[:, :, 0:H, 0:W]

    def load_weights(self, model_path="models/image_lowlight.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        sd = torch.load(checkpoint)
        self.load_state_dict(sd['params'])


# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = RetinexFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')