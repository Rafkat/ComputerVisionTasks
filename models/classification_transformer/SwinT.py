import torch
from timm.layers import DropPath
from torch import nn
from timm.models.layers import trunc_normal_


# originated from https://arxiv.org/pdf/2103.14030


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim=None, dropout=0.):
        super(FeedForward, self).__init__()
        out_features = output_dim or dim
        hidden_features = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        dim_head = dim // num_heads
        proj_out = not (num_heads == 1 and dim == dim_head)
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(drop)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(drop),
        ) if proj_out else nn.Identity()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.to_qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = dots + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).view(out.size(0), out.size(2), -1)
        return self.out(out)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., drop=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim,
                                    window_size=self.window_size,
                                    num_heads=num_heads,
                                    drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ff = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

        if self.shift_size > 0:
            h, w = input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h_ in h_slices:
                for w_ in w_slices:
                    img_mask[:, h_, w_, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, h, w)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, h, w)
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.ff(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., drop=0., drop_path=0.,
                 downsample=None):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.downsample = downsample
        self.blocks = nn.ModuleList(
            [SwinTransformerBlock(dim, input_resolution=input_resolution,
                                  num_heads=num_heads,
                                  window_size=window_size,
                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  drop=drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, (tuple, list)) else drop_path)
             for i in range(depth)
             ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, conv_type=False, channels=3, emb_dropout=0.):
        super(PatchEmbedding, self).__init__()

        self.channels = channels
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

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
        self.dropout = nn.Dropout(emb_dropout)

    def _get_patches_manually_alt(self, img):
        img = img.permute(0, 2, 3, 1)
        img = img.view(img.size(0),
                       img.size(1) // self.patch_size, self.patch_size,
                       img.size(2) // self.patch_size, self.patch_size,
                       img.size(3))
        img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(img.size(0), -1, self.patch_dim)
        return img

    def _get_patches_manually(self, img):
        patches = img.unfold(1,
                             self.channels,
                             self.channels).unfold(2,
                                                   self.patch_size,
                                                   self.patch_size).unfold(3,
                                                                           self.patch_size,
                                                                           self.patch_size)
        return patches.contiguous().view(img.size(0), -1, self.patch_dim)

    def forward(self, x):
        if not self.conv_type:
            x = self._get_patches_manually(x)

        x = self.patcher(x)

        if self.conv_type:
            x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, channels=3, embed_dim=96, nb_classes=1000, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., drop=0., drop_path=0.1):
        super(SwinTransformer, self).__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))

        self.patcher = PatchEmbedding(patch_size, embed_dim, conv_type=False, channels=channels)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=((img_size[0] // patch_size[0]) // (2 ** i_layer),
                                                 (img_size[1] // patch_size[1]) // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               drop=drop,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=PatchMerging if (i_layer < len(depths) - 1) else None)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, nb_classes)

    def forward(self, x):
        x = self.patcher(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    swint = SwinTransformer(img_size=224)
    up_side = torch.hstack([torch.cat([torch.ones(4, 4) * (i + 1), torch.zeros(4, 4)], dim=1) for i in range(224 // 8)])
    down_side = torch.hstack(
        [torch.cat([torch.zeros(4, 4), (i + 1) * torch.ones(4, 4)], dim=1) for i in range(224 // 8)])
    test_tensor = torch.vstack([torch.cat([up_side, down_side], dim=0) for _ in range(224 // 8)])
    test_tensor = test_tensor.expand(3, -1, -1)
    test_tensor = test_tensor.unsqueeze(0)
    swint(test_tensor)
