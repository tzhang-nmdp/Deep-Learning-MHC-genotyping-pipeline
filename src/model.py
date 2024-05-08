import torch
import torch.nn as nn

from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

class KirAI_10k(nn.Module):
    S = 10000

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
        )

        self.res_conv3 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
        )

        self.res_conv4 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding='same')
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x)
        detour += self.res_conv3(x)

        x = self.block3(x)
        detour += self.res_conv4(x)

        x = self.block4(x) + detour
        x = self.conv_last(x)

        return rearrange(x[..., 5000:5000 + 5000], 'b c l -> b l c')

class KirAI():

    @staticmethod
    def from_preconfigured(model_name):
        if model_name == '10k':
            return KirAI_10k()
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

if __name__ == '__main__':
    import torch

    x = torch.randn([16, 4, 10000 + 5000])
    model = KirAI_10k()
    print(model(x).shape)
