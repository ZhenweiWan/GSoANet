import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_ContextGating(nn.Module):
    def __init__(self, vlad_dim, hidden_size, drop_rate=0.5, gating_reduction=8):
        super(SE_ContextGating, self).__init__()

        self.fc1 = nn.Linear(vlad_dim, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.gate = torch.nn.Sequential(
            nn.Linear(hidden_size, hidden_size // gating_reduction),
            # nn.BatchNorm1d(hidden_size // gating_reduction),
            nn.ReLU(),

            nn.Linear(hidden_size // gating_reduction, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.bn1(self.dropout(self.fc1(x)))
        x = self.dropout(self.fc1(x))
        gate = self.gate(x)
        activation = x * gate
        return activation


class NextVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, expansion=2, group=8, dim=2048, num_class=1):
        super(NextVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.expansion = expansion
        self.group = group
        self.dim = dim
        self.bn1 = nn.BatchNorm1d(group * num_clusters)
        # self.bn2 = nn.BatchNorm1d(num_clusters * expansion * dim // group)
        self.centroids1 = nn.Parameter(torch.rand(expansion * dim, group * num_clusters))
        self.centroids2 = nn.Parameter(torch.rand(1, expansion * dim // group, num_clusters))
        self.fc1 = nn.Linear(dim, expansion * dim)
        self.fc2 = nn.Linear(dim * expansion, group)

        self.cg = SE_ContextGating(num_clusters * expansion * dim // group, dim)
        # self.fc3 = nn.Linear(dim, num_class)

    def forward(self, x):  # 2,4,2048,1,1
        x = x.reshape(x.size(0) // 8, 8, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3) * x.size(4))
        x = x.permute(0, 2, 1)
        max_frames = x.size(1)
        x = x.view(x.size()[:3])  # 2,4,2048

        x_3d = F.normalize(x, p=2, dim=2)  # across descriptor dim, torch.Size([2,4, 2048, 1, 1])

        vlads = []
        for t in range(x_3d.size(0)):
            x = x_3d[t, :, :]  # 4,2048
            x = self.fc1(x)  # expand, 4,2*2048

            # attention
            attention = torch.sigmoid(self.fc2(x))  # 4,8
            attention = attention.view(-1, max_frames * self.group, 1)

            feature_size = self.expansion * self.dim // self.group
            # reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
            reshaped_input = x.view(-1, self.expansion * self.dim)  # 4,2*2048
            # activation = tf.matmul(reshaped_input, cluster_weights)
            activation = torch.mm(reshaped_input, self.centroids1)  # 4,8*32
            activation = self.bn1(activation)
            # activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
            activation = activation.view(-1, max_frames * self.group, self.num_clusters)  # 1,32,32
            # activation = tf.nn.softmax(activation, axis=-1)
            activation = F.softmax(activation, dim=-1)  # 1,32,32
            # activation = tf.multiply(activation, attention)
            activation = activation * attention  # 1,32,32
            # a_sum = tf.sum(activation, -2, keep_dims=True)
            a_sum = activation.sum(dim=-2, keepdim=True)  # 1,32,1

            # a = tf.multiply(a_sum, cluster_weights2)
            a = a_sum * self.centroids2  # 1,512,32 (512=dim*expansion//group,32=clusters)
            # activation = tf.transpose(activation, perm=[0, 2, 1])
            activation = activation.permute(0, 2, 1)  # 1,32,1
            # reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
            reshaped_input = x.view(-1, max_frames * self.group, feature_size)  # 1,32,512
            vlad = torch.bmm(activation, reshaped_input)  # 1,32,512
            # vlad = tf.transpose(vlad, perm=[0, 2, 1])
            vlad = vlad.permute(0, 2, 1)
            # vlad = tf.subtract(vlad, a)
            vlad = vlad - a  # 1,512,32
            # vlad = tf.nn.l2_normalize(vlad, 1)
            vlad = F.normalize(vlad, p=2, dim=1)
            # vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
            vlad = vlad.contiguous()
            vlad = vlad.view(self.num_clusters * feature_size)  # [1, 16384]
            vlads.append(vlad)
        vlads = torch.stack(vlads, dim=0)
        # vlads = self.bn2(vlads)  # [2, 16384]

        x = self.cg(vlads)  # SE Context Gating
        # x = self.fc3(x)

        return x
