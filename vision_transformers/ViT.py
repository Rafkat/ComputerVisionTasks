import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, conv_type=False, channels=3, emb_dropout=0.):
        super(PatchEmbedding, self).__init__()

        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.conv_type = conv_type
        if conv_type:
            self.patcher = nn.Sequential(
                nn.Conv2d(in_channels=channels,
                          out_channels=embed_dim,
                          kernel_size=patch_size,
                          stride=patch_size, ),
                nn.Flatten(2)
            )
        else:
            self.patch_dim = channels * patch_height * patch_width
            self.patcher = nn.Sequential(
                nn.LayerNorm(self.patch_dim, eps=1e-6),
                nn.Linear(self.patch_dim, embed_dim),
                nn.LayerNorm(embed_dim, eps=1e-6),
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        if not self.conv_type:
            x = x.reshape(x.size(0), -1, self.patch_dim)

        x = self.patcher(x)

        if self.conv_type:
            x = x.permute(0, 2, 1)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim_in, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim_in)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim_in)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_in),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.size(0), self.heads, t.size(1), -1), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = out.view(out.size(0), out.size(2), -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, nb_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT, self).__init__()
        self.patch_embedder = PatchEmbedding(image_size, patch_size, dim,
                                             conv_type=False,
                                             channels=channels,
                                             emb_dropout=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, nb_classes)

    def forward(self, img):
        x = self.patch_embedder(img)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    net = ViT(image_size=32, patch_size=16, nb_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072)
    net(torch.randn(1, 3, 32, 32))

