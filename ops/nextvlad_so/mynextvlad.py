import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.MPNCOV import MPNCOV

PD = False  # Print Dimensions


class NextVLAD(nn.Module):

    def __init__(self, num_tokens=512, dim=1024, num_clusters=128, expansion=2, groups=8, alpha=100.0,
                 normalize_input=True,out_dim=64):
        super(NextVLAD, self).__init__()
        self.feature_size = dim
        self.expansion = expansion
        self.num_clusters = num_clusters
        self.groups = groups
        # self.alpha = alpha
        self.normalize_input = normalize_input
        self.centroids = nn.Parameter(torch.rand(num_clusters, self.expansion * self.feature_size // self.groups))
        self.fc_inp = nn.Linear(self.feature_size, expansion * self.feature_size)
        self.fc_alpha_g = nn.Linear(self.feature_size * expansion, groups)
        self.fc_alpha_gk = nn.Linear(self.feature_size * expansion, groups * num_clusters)
        self.softmax = nn.Softmax(dim=1)
        in_dim = dim * expansion // groups
        self.reduce_dim = nn.Linear(in_dim, out_dim)
        # self.fc_last = nn.Linear(self.expansion*self.feature_size, self.feature_size)
        self._init_params()

    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.centroids, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_normal_(self.fc_inp.weight)
        torch.nn.init.uniform_(self.fc_inp.bias)
        torch.nn.init.xavier_normal_(self.fc_alpha_g.weight)
        torch.nn.init.uniform_(self.fc_alpha_g.bias)
        torch.nn.init.xavier_normal_(self.fc_alpha_gk.weight)
        torch.nn.init.uniform_(self.fc_alpha_gk.bias)
        torch.nn.init.xavier_normal_(self.reduce_dim.weight)
        torch.nn.init.uniform_(self.reduce_dim.bias)
        # torch.nn.init.xavier_normal_(self.fc_last.weight)
        # torch.nn.init.uniform_(self.fc_last.bias)

    def forward(self, x):
        """
            bs = batch size = 1
            x = [bs, M, N]
        """
        x = x.reshape(x.size(0) // 8, 8, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3) * x.size(4))
        x = x.permute(0, 2, 1)
        bs, M, N = x.shape[:3]
        # D = self.feature_size
        # print(x.shape)
        # input = [batch_size, num_tokens, dimension]
        # input = [1, 512, 1024]
        # N = x.shape[0]
        # M = x.shape[1]  # M = number of Tokens
        # D = x.shape[2]  # D = dimension of descriptor, feature size
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim
        x = self.fc_inp(x)
        alpha_g = torch.sigmoid(self.fc_alpha_g(x))
        alpha_gk = self.softmax(self.fc_alpha_gk(x))
        alpha_gk = alpha_gk.view(bs, M, self.groups, self.num_clusters)
        reshaped_x = x.view(bs, M, self.groups, self.expansion * N // self.groups)
        residual = reshaped_x.expand(self.num_clusters, -1, -1, -1, -1).permute(1, 2, 3, 0, 4) - \
                   self.centroids
        final = residual * alpha_g.unsqueeze(3).unsqueeze(4) * alpha_gk.unsqueeze(4)
        vlad = final
        # vlad = F.normalize(final, p=2, dim=-1)
        vlad = vlad.sum(dim=2)
        vlad = self.reduce_dim(vlad)
        vlad = vlad.permute(0, 3, 2, 1)
        vlad = vlad.reshape(vlad.size(0), vlad.size(1), vlad.size(2), vlad.size(3) // 6, 6)
        # vlad = vlad.sum(dim=2)
        #
        # cov = MPNCOV.CovpoolLayer(vlad)
        # cov = MPNCOV.SqrtmLayer(cov, 3)
        # cov = MPNCOV.TriuvecLayer(cov)
        # cov = cov.reshape(cov.size(0), cov.size(1))
        #
        # cov = F.normalize(cov, p=2, dim=1)

        K_ = vlad.size(2)
        B_ = vlad.size(0)
        cov = torch.zeros(B_, K_, 1176, requires_grad=True).cuda()
        for k in range(K_):
            x = vlad[:, :, k, :, :]
            x = MPNCOV.CovpoolLayer(x)
            # fast matrix power normalization, P_{TCP}^{1/2}
            x = MPNCOV.SqrtmLayer(x, 3)
            x = MPNCOV.TriuvecLayer(x)
            x = x.reshape(x.size(0), x.size(1))
            # print(x.size())
            cov[:, k] = x
        cov = cov.reshape(cov.size(0), cov.size(1) * cov.size(2))
        # cov = F.normalize(cov, p=2, dim=1)
        # print(cov.size())
        return cov
