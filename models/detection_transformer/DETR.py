import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_heads=8, n_encoder_layers=6, n_decoder_layers=6):
        super(DETR, self).__init__()

        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(in_channels=2048, out_channels=hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads,
                                          num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.pos_embed = nn.Parameter(torch.randn(100, hidden_dim))

        self.row_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.conv(x)
        h, w = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        t = self.transformer(pos + 0.1 * x.flatten(2).permute(2, 0, 1), self.pos_embed.unsqueeze(1)).transpose(0, 1)
        return self.linear_class(t), self.linear_bbox(t).sigmoid()


if __name__ == '__main__':
    model = DETR(10)
    classes, bboxes = model(torch.randn(1, 3, 224, 224))
    print(classes.shape, bboxes.shape)
