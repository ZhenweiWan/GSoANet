import torch.nn as nn


class temporal(nn.Module):

    def __init__(self, dim, kernel_size=1, stride=1, padd=0):
        super().__init__()
        # self.reduce_dim = nn.Conv3d(in_channels=dim, out_channels=dim // 2, kernel_size=1, stride=1, padding=0,
        #                             bias=False)

        self.temporal_1 = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(kernel_size, 1, 1),
                                    stride=(stride, 1, 1),
                                    padding=(padd, 0, 0), bias=False)
        # self.temporal_21 = nn.Conv3d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=(3, 1, 1), stride=1,
        #                             padding=(1, 0, 0), bias=False)
        # self.temporal_22 = nn.Conv3d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=(3, 1, 1), stride=1,
        #                             padding=(1, 0, 0), bias=False)

        # self.up_dim = nn.Conv3d(in_channels=dim // 2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

        # self.norm = nn.LayerNorm(dim)

        # nn.init.trunc_normal_(self.temporal_1.weight, std=.02)
        # nn.init.constant_(self.temporal_1.bias, 0)

    def forward(self, x):
        # print('aaaaaaaaaaaaaaaaaaaaaaaaa')
        # res = x

        x = x.reshape(x.size(0) // 8, 8, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        # x = self.reduce_dim(x)
        # temp1 = x
        # temp2 = x
        x = self.temporal_1(x)
        # temp2 = self.temporal_21(temp2)
        # temp2 = self.temporal_22(temp2)
        # x = temp1 + temp2
        # x = self.up_dim(temp1)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))

        # x = res + x
        # x = x.permute(0, 2, 3, 1)
        # x = self.norm(x)
        # x = x.permute(0, 3, 1, 2)

        return x
