import torch
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F


# originated from https://arxiv.org/pdf/2102.12122

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        proj_out = not (dim_head == dim and num_heads == 1)

        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(drop)

        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(drop),
        ) if proj_out else nn.Identity()

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.to_sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_sr = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, h, w):
        x = self.norm(x)
        b, n, c = x.shape
        q = self.to_q(x).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).view(b, c, h, w)
            x = self.to_sr(x).view(b, c, -1).permute(0, 2, 1)
            x = self.norm_sr(x)

        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.drop(self.softmax(dots))
        out = torch.matmul(attn, v)
        out = out.view(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., sr_ratio=1, drop_path=0.):
        super(Transformer, self).__init__()
        self.ff = FeedForward(dim, mlp_ratio, drop)
        self.attention = Attention(dim, num_heads, drop, sr_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, h, w):
        x = x + self.drop(self.attention(self.norm1(x), h, w))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, in_channels=3):
        super(PatchEmbedding, self).__init__()
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.num_patches = self.h * self.w
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        h_nb = h // self.patch_size
        w_nb = w // self.patch_size

        x = x.permute(0, 2, 3, 1)
        x = x.view(b,
                   h_nb, self.patch_size,
                   w_nb, self.patch_size,
                   c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, -1, self.patch_dim)
        x = self.proj(x)
        return x, (h_nb, w_nb)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, nb_classes=1000, embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8), mlp_ratio=(8, 8, 4, 4), drop_rate=0., drop_path_rate=0., depths=(3, 3, 6, 3),
                 sr_ratios=(8, 4, 2, 1), num_stages=4):
        super(PyramidVisionTransformer, self).__init__()
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            patch_embed = PatchEmbedding(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                         patch_size=patch_size if i == 0 else 2,
                                         in_channels=in_channels if i == 0 else embed_dims[i - 1],
                                         embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(drop_rate)

            block = nn.ModuleList([
                Transformer(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i],
                            drop_path=dpr[cur + j], sr_ratio=sr_ratios[i]) for j in range(depths[i])
            ])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'pos_embed{i + 1}', pos_embed)
            setattr(self, f'pos_drop{i + 1}', pos_drop)
            setattr(self, f'block{i + 1}', block)

        self.norm = nn.LayerNorm(embed_dims[-1])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dims[-1]))

        self.fc = nn.Linear(embed_dims[3], nb_classes)

    def _get_pos_embed(self, pos_embed, patch_embed, h, w):
        if h * w == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.h, patch_embed.h, -1).permute(0, 3, 1, 2),
                size=(h, w), mode='bilinear').reshape(1, -1, h * w).permute(0, 2, 1)

    def forward(self, x):
        b = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            pos_embed = getattr(self, f'pos_embed{i + 1}')
            pos_drop = getattr(self, f'pos_drop{i + 1}')
            block = getattr(self, f'block{i + 1}')
            x, (h, w) = patch_embed(x)

            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(b, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, h, w)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, h, w)

            x = pos_drop(x + pos_embed)

            for block in block:
                x = block(x, h, w)
            if i != self.num_stages - 1:
                x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)
        x = x[:, 0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = PyramidVisionTransformer()
    net(torch.randn(1, 3, 224, 224))
