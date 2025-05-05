import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names


# originated from https://arxiv.org/pdf/2005.12872

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim_in, dim_hidden=768, num_heads=8, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_hidden * num_heads
        proj_out = not (num_heads != 1 and dim_hidden == dim_in)

        self.num_heads = num_heads
        self.scale = dim_hidden ** -0.5

        self.norm = nn.LayerNorm(dim_in)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim_in, inner_dim)
        self.to_k = nn.Linear(dim_in, inner_dim)
        self.to_v = nn.Linear(dim_in, inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_in),
            nn.Dropout(dropout)
        ) if proj_out else nn.Identity()

    def forward(self, q, k, v, mask=None):
        q = self.to_q(self.norm(q))
        k = self.to_k(self.norm(k))
        v = self.to_v(self.norm(v))

        q = q.reshape(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-1e20'))

        dots = dots * self.scale
        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = out.reshape(out.size(0), out.size(2), -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, dim_in, depth, dim_hidden=768, num_heads=8, dropout=0.):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim_in, dim_hidden, num_heads, dropout),
                FeedForward(dim_in, dim_hidden, dropout)
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim_in, depth, dim_hidden=768, num_heads=8, dropout=0.):
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim_in, dim_hidden, num_heads, dropout),
                FeedForward(dim_in, dim_hidden, dropout)
            ])
            for _ in range(depth)
        ])

    def forward(self, enc_q, enc_k, x, mask):
        for attn, ff in self.layers:
            x = attn(x, x, x, mask) + x
            x = self.norm(x)
            x = attn(enc_q, enc_k, x) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim_in, enc_depth, dec_depth, dim_hidden=768, num_heads=8, dropout=0.):
        super(Transformer, self).__init__()
        self.encoder = Encoder(dim_in, enc_depth, dim_hidden, num_heads, dropout)
        self.decoder = Decoder(dim_in, dec_depth, dim_hidden, num_heads, dropout)

    def forward(self, enc_input, dec_input):
        x = self.encoder(enc_input)
        mask = self._make_trg_mask(dec_input)
        x = self.decoder(x, x, dec_input, mask)
        return x

    def _decode(self, src, trg):
        trg_mask = self._make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        b, seq_len = src.shape[0], src.shape[1]
        out = trg
        for i in range(seq_len):
            out = self.decoder(out, enc_out, trg_mask)
            out = out[:, -1, :]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, dim=0)
        return out_labels

    @staticmethod
    def _make_trg_mask(trg):
        b, seq_len, _ = trg.shape
        trg_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(b, 1, seq_len, seq_len)
        return trg_mask


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_heads=8, n_encoder_layers=6, n_decoder_layers=6):
        super(DETR, self).__init__()

        self.feature_extractor = FeatureExtractor()

        self.conv = nn.Conv2d(in_channels=2048, out_channels=hidden_dim, kernel_size=1)

        self.transformer = Transformer(dim_in=hidden_dim, enc_depth=n_encoder_layers,
                                       dec_depth=n_decoder_layers, num_heads=num_heads)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.pos_embed = nn.Parameter(torch.randn(100, hidden_dim))

        self.row_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))

    def forward(self, x):
        x = self.feature_extractor(x)

        x = self.conv(x)
        h, w = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)

        t = self.transformer(pos + 0.1 * x.flatten(2).permute(0, 2, 1),
                             self.pos_embed[:pos.size(1)].repeat(x.size(0), 1, 1))
        return self.linear_class(t), self.linear_bbox(t).sigmoid()


if __name__ == '__main__':
    model = DETR(10)
    classes, bboxes = model(torch.randn(2, 3, 224, 224))
    print(classes.shape, bboxes.shape)
