import numpy as np
import torch
from timm.layers import DropPath
from torch import nn


# originated from https://arxiv.org/pdf/2101.11986


def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim_in, mlp_ratio=4, drop=0.):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, mlp_ratio * dim_in),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_ratio * dim_in, dim_in),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.ff(x)


class TokenAttention(nn.Module):
    def __init__(self, dim, dim_in, num_heads=8, drop=0.):
        super(TokenAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        proj_out = not (num_heads == 1 and dim == head_dim)

        self.to_qkv = nn.Linear(dim, dim_in * 3)
        self.drop = nn.Dropout(drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Dropout(drop)
        ) if proj_out else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.drop(self.softmax(dots))
        out = torch.matmul(attn, v)
        out = out.view(b, n, -1)
        return self.to_out(out)


class TokenTransformer(nn.Module):
    def __init__(self, dim, dim_in, num_heads=8, mlp_ratio=4, drop=0., drop_path=0.):
        super(TokenTransformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TokenAttention(dim, dim_in, num_heads, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_in)
        self.ff = FeedForward(dim_in, mlp_ratio, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class Tokens2Token(nn.Module):
    def __init__(self, img_size=224, in_channels=3, embed_dim=768, token_dim=64):
        super(Tokens2Token, self).__init__()
        self.soft_split0 = nn.Unfold(kernel_size=7, stride=4, padding=2)
        self.soft_split1 = nn.Unfold(kernel_size=3, stride=2, padding=1)
        self.soft_split2 = nn.Unfold(kernel_size=3, stride=2, padding=1)

        self.attention1 = TokenTransformer(dim=in_channels * 7 * 7, dim_in=token_dim, num_heads=1, mlp_ratio=1)
        self.attention2 = TokenTransformer(dim=token_dim * 3 * 3, dim_in=token_dim, num_heads=1, mlp_ratio=1)
        self.proj = nn.Linear(token_dim * 3 * 3, embed_dim)

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

    def forward(self, x):
        x = self.soft_split0(x).transpose(1, 2)
        x = self.attention1(x)

        b, new_hw, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, int(np.sqrt(new_hw)), int(np.sqrt(new_hw)))
        x = self.soft_split1(x).transpose(1, 2)

        x = self.attention2(x)
        b, new_hw, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, int(np.sqrt(new_hw)), int(np.sqrt(new_hw)))
        x = self.soft_split2(x).transpose(1, 2)

        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim_in, num_heads=8, drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = head_dim ** -0.5
        proj_out = not (num_heads == 1 and dim_in == head_dim)

        self.to_qkv = nn.Linear(dim_in, dim_in * 3)
        self.drop = nn.Dropout(drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Dropout(drop)
        ) if proj_out else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.tranpose(-1, -2)) * self.scale
        attn = self.drop(self.softmax(dots))
        out = torch.matmul(attn, v)
        out = out.view(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim_in, num_heads=8, mlp_ratio=4, drop=0., drop_path=0.):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim_in)
        self.attn = TokenAttention(dim_in, dim_in, num_heads, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_in)
        self.ff = FeedForward(dim_in, mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class T2TViT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, embed_dim=768, token_dim=64, depth=12,
                 num_heads=12, mlp_ratio=4, drop_rate=0., drop_path_rate=0., nb_classes=1000):
        super(T2TViT, self).__init__()
        self.tokens_to_token = Tokens2Token(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim,
                                            token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Transformer(
                dim_in=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, nb_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


if __name__ == '__main__':
    net = T2TViT(img_size=224)
    net(torch.randn(1, 3, 224, 224))
