import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool3d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, in_shape=(1,32,128,128)):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.in_shape=in_shape

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat+1, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool3d((8, 32, 32)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        # self.timeembed2 = EmbedFC(1, 1 * n_feat)
        # self.contextembed1 = EmbedFC(1, 2 * n_feat)
        # self.contextembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(4 * n_feat, 2 * n_feat, (8,32,32), (8,32,32)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv3d(2 * n_feat+1, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv3d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None, None, None, None]
        context_mask = context_mask.repeat(1, *self.in_shape)
        context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
        c = c * context_mask

        x = self.init_conv(x)
        x = torch.cat((x, c), 1)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)



        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        hiddenvec = torch.cat((hiddenvec, temb1), 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1, down2)
        up3 = self.up2(up2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out

