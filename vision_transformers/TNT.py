import math

import torch
from timm.layers import DropPath
from torch import nn


# originated from https://arxiv.org/pdf/2103.00112


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim, patch_size, inner_stride, drop=0.):
        super(PatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_dim = in_channels * self.patch_size * self.patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=7, padding=3, stride=inner_stride)

        self.drop = nn.Dropout(drop)
        self.num_words = math.ceil(patch_size / inner_stride) * math.ceil(patch_size / inner_stride)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0),
                   x.size(1) // self.patch_size, self.patch_size,
                   x.size(2) // self.patch_size, self.patch_size,
                   x.size(3))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, self.patch_dim)
        x = x.view(b * self.num_patches, c, self.patch_size, self.patch_size)
        x = self.proj(x)
        x = x.view(b * self.num_patches, self.embed_dim, -1).transpose(1, 2)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim_in, num_heads, drop=0.):
        super(Attention, self).__init__()
        self.dim_head = dim_in // num_heads
        self.num_heads = num_heads
        proj_out = not (num_heads == 1 and dim_in == self.dim_head)
        self.scale = self.dim_head ** -0.5

        self.norm = nn.LayerNorm(dim_in)
        self.to_qkv = nn.Linear(dim_in, dim_in * 3)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop)

        self.to_out = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Dropout(drop)
        ) if proj_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.size(0), t.size(1), self.num_heads, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.drop(self.softmax(dots))
        out = torch.matmul(attn, v)
        out = out.view(out.size(0), out.size(2), -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim_in, mlp_ratio, drop=0.):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, mlp_ratio * dim_in),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_ratio * dim_in, dim_in),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4,
                 drop=0., drop_path=0.):
        super(Block, self).__init__()

        self.inner_norm1 = nn.LayerNorm(inner_dim)
        self.inner_attn = Attention(inner_dim, num_heads=inner_num_heads, drop=drop)
        self.inner_norm2 = nn.LayerNorm(inner_dim)
        self.inner_mlp = FeedForward(inner_dim, mlp_ratio=mlp_ratio, drop=drop)
        self.proj_norm1 = nn.LayerNorm(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2 = nn.LayerNorm(outer_dim)

        self.outer_norm1 = nn.LayerNorm(outer_dim)
        self.outer_attn = Attention(outer_dim, num_heads=outer_num_heads, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = nn.LayerNorm(outer_dim)
        self.outer_mlp = FeedForward(outer_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, inner_tokens, outer_tokens):
        inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))
        inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))
        b, n, c = outer_tokens.size()
        outer_tokens[:, 1:] = (outer_tokens[:, 1:]
                               + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(b, n - 1, -1)))))

        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
        outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class TransformerInTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, nb_classes=1000, outer_dim=192, inner_dim=12,
                 depth=12, outer_num_heads=3, inner_num_heads=2, mlp_ratio=4, drop_rate=0., drop_path_rate=0.,
                 inner_stride=4):
        super(TransformerInTransformer, self).__init__()

        self.patcher = PatchEmbedding(
            img_size=img_size, in_channels=in_channels, embed_dim=inner_dim,
            patch_size=patch_size, drop=drop_rate, inner_stride=inner_stride
        )
        self.num_words = self.patcher.num_words
        self.num_patches = self.patcher.num_patches

        self.proj_norm1 = nn.LayerNorm(self.num_words * inner_dim)
        self.proj = nn.Linear(self.num_words * inner_dim, outer_dim)
        self.proj_norm2 = nn.LayerNorm(outer_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, outer_dim))
        self.outer_tokens = nn.Parameter(torch.randn(1, self.num_patches, outer_dim))
        self.outer_pos = nn.Parameter(torch.randn(1, self.num_patches + 1, outer_dim))
        self.inner_pos = nn.Parameter(torch.randn(1, self.num_words, inner_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                  inner_num_heads=inner_num_heads, num_words=self.num_words, mlp_ratio=mlp_ratio,
                  drop=drop_rate, drop_path=dpr[i]) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(outer_dim)
        self.fc = nn.Linear(outer_dim, nb_classes)

    def forward(self, x):
        inner_tokens = self.patcher(x) + self.inner_pos
        outer_tokens = self.proj_norm2(
            self.proj(self.proj_norm1(inner_tokens.reshape(x.size(0), self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(x.size(0), -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for block in self.blocks:
            inner_tokens, outer_tokens = block(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        x = outer_tokens[:, 0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = TransformerInTransformer()
    net(torch.randn(1, 3, 224, 224))
