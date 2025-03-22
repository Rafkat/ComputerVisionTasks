import torch
from torch import nn


# originated from https://arxiv.org/pdf/2102.10882


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, drop=0.):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim, eps=1e-6),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(self.patch_dim, eps=1e-6),
        )
        self.drop = nn.Dropout(drop)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_dim))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0),
                   x.size(1) // self.patch_size, self.patch_size,
                   x.size(2) // self.patch_size, self.patch_size,
                   x.size(3))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(x.size(0), -1, self.patch_dim)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.patch_embedding(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop=0.):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Linear(embed_dim, int(mlp_ratio * embed_dim)),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, dim_in, num_heads, drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_in // num_heads
        proj_out = not (num_heads == 1 and dim_in == self.dim_head)

        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, dim_in * 3)
        self.norm = nn.LayerNorm(dim_in, eps=1e-6)

        self.to_out = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Dropout(drop)
        ) if proj_out else nn.Identity()

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.size(0), t.size(1), self.num_heads, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.drop(self.softmax(dots))
        out = torch.matmul(attn, v)
        out = out.view(out.size(0), out.size(2), -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, depth=12, drop=0.):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.block = nn.ModuleList([
            nn.ModuleList([
                Attention(embed_dim, num_heads, drop=drop),
                FeedForward(embed_dim, mlp_ratio, drop=drop)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.block:
            x = attn(x) + x
            x = ff(x)
        return self.norm(x)


class PositionEncodingGenerator(nn.Module):
    def __init__(self, dim_in, patch_dim):
        super(PositionEncodingGenerator, self).__init__()
        self.patch_dim = patch_dim
        self.pos_encoder = nn.Conv2d(patch_dim, patch_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, x, h, w):
        input_tensor = x
        cls_token, x = torch.split(x, [1, h * w], dim=1)
        x = x.view(x.size(0), h, w, x.size(3)).permute(0, 3, 1, 2)
        pos_encoding = self.pos_encoder(x).permute(0, 2, 3, 1).view(x.size(0), h * w, self.patch_dim)
        x = input_tensor + pos_encoding
        x = torch.cat((cls_token, x), dim=1)
        return self.norm(x)


class ConditionalPositionVisionTransformer(nn.Module):
    def __init__(self, patch_size, nb_classes, dim, depth, heads, mlp_ratio=4., channels=3,
                 dropout=0., emb_dropout=0.):
        super(ConditionalPositionVisionTransformer, self).__init__()
        self.patcher = PatchEmbedding(channels, dim, patch_size, emb_dropout)
        self.transformer = Transformer(dim, heads, mlp_ratio, depth, dropout)

        self.fc = nn.Linear(dim, nb_classes)

    def forward(self, x):
        x = self.patcher(x)
        x = self.transformer(x)
        x = x[:, 0]
        return self.fc(x)


if __name__ == '__main__':
    net = ConditionalPositionVisionTransformer(patch_size=16, nb_classes=10, dim=768, depth=12, heads=12)
    net(torch.randn(1, 3, 224, 224))
