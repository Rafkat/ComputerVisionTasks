import torch
import torch.nn as nn
from timm.layers import DropPath


# originated from https://arxiv.org/pdf/2103.15808


class FeedForward(nn.Module):
    def __init__(self, dim_in, mlp_ratio, drop=0.):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in, eps=1e-6),
            nn.Linear(dim_in, mlp_ratio * dim_in),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_ratio * dim_in, dim_in),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, drop=0., kernel_size=3, stride=1, padding=1):
        super(Attention, self).__init__()
        head_dim = dim_in // num_heads
        self.scale = head_dim ** -0.5

        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1)
        )

        self.conv_k = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1)
        )

        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.Dropout(drop)
        )

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        out = torch.matmul(attn, v)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, drop=0., kernel_size=3,
                 stride=1, padding=1, mlp_ratio=4, drop_path=0.):
        super(Block, self).__init__()
        self.attn = Attention(dim_in, dim_out, num_heads, drop, kernel_size, stride, padding)
        self.ff = FeedForward(dim_in, mlp_ratio=int(dim_out * mlp_ratio), drop=drop)
        self.norm1 = nn.LayerNorm(dim_in, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_in, eps=1e-6)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = res + self.drop_path(x)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class ConvEmbedder(nn.Module):
    def __init__(self, dim_in, dim_out, patch_size, stride=4, padding=2):
        super(ConvEmbedder, self).__init__()
        self.patcher = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x = self.patcher(x)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(b, -1, c)

        x = self.norm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, patch_stride=16, patch_padding=0,
                 in_channels=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, drop_rate=0):
        super(VisionTransformer, self).__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        self.patcher = ConvEmbedder(dim_in=in_channels, patch_size=patch_size,
                                    stride=patch_stride, padding=patch_padding,
                                    dim_out=embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(dim_in=embed_dim,
                  dim_out=embed_dim,
                  num_heads=num_heads,
                  drop=drop_rate,
                  drop_path=dpr[j],
                  mlp_ratio=mlp_ratio) for j in range(depth)
        ])

    def forward(self, x):
        x = self.patcher(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2).view(b, -1, c)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        cls_tokens, x = torch.split(x, [1, h * w], 1)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self, nb_classes=1000, blocks_config=(1, 2, 10), heads_config=(1, 3, 6),
                 embed_dim_config=(64, 192, 384), conv_in_embed_config=(3, 64, 192)):
        super(ConvolutionalVisionTransformer, self).__init__()

        self.blocks = nn.ModuleList(
            [
                VisionTransformer(in_channels=conv_in_embed_config[i],
                                  embed_dim=embed_dim_config[i],
                                  depth=blocks_config[i],
                                  num_heads=heads_config[i])
                for i in range(len(blocks_config))
            ]
        )

        self.norm = nn.LayerNorm(embed_dim_config[-1])
        self.head = nn.Linear(embed_dim_config[-1], nb_classes)

    def forward(self, x):
        for block in self.blocks:
            x, cls_tokens = block(x)

        x = self.norm(cls_tokens)
        x = torch.squeeze(x)

        x = self.head(x)
        return x
