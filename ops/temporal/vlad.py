import torch
import torch.nn as nn


class SeqVLADModule(nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None, with_center_loss=False,
                 init_method='xavier_normal'):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(SeqVLADModule, self).__init__()
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        self.with_relu = with_relu

        self.in_shape = None
        self.out_shape = self.num_centers * self.redu_dim
        self.batch_size = None
        self.activation = activation

        self.with_center_loss = with_center_loss

        self.init_method = init_method

        self.redu_w = nn.Conv2d(768, self.redu_dim, kernel_size=1 ,bias=True, stride=1, padding=0, dilation=1, groups=1)
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)
        if self.with_relu:
            print('redu with relu ...')

        self.share_w = nn.Conv2d(self.redu_dim, self.num_centers,kernel_size=1, bias=True, stride=1, padding=0, dilation=1, groups=1)

        self.U_z = nn.Conv2d(self.num_centers,self.num_centers,kernel_size=3,bias=False,stride=1,padding=1)
        self.sigmoid_U_z = nn.Sigmoid()

        self.U_r = nn.Conv2d(self.num_centers,self.num_centers,kernel_size=3,bias=False,stride=1,padding=1)
        self.sigmoid_U_r = nn.Sigmoid()

        self.U_h = nn.Conv2d(self.num_centers,self.num_centers,kernel_size=3,bias=False,stride=1,padding=1)
        self.tanh_U_h = nn.Tanh()

        self.sofmax = nn.Softmax()

        def init_func(t):
            if self.init_method == 'xavier_normal':
                return torch.nn.init.xavier_normal(t)
            elif self.init_method == 'orthogonal':
                return torch.nn.init.orthogonal_(t)
            elif self.init_method == 'uniform':
                return torch.nn.init.uniform(t, a=0, b=0.01)


        self.centers = torch.Tensor(self.num_centers, self.redu_dim)  # weight : out, in , h, w
        self.centers = init_func(self.centers)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=True)


    def forward(self, input):
        self.in_shape = input.size()
        self.batch_size = self.in_shape[0] // self.timesteps
        if self.batch_size == 0:
            self.batch_size = 1
        input_tensor = input
        if self.redu_dim == None:
            self.redu_dim = self.in_shape[1]
        elif self.redu_dim < self.in_shape[1]:
            input_tensor = self.redu_w(input_tensor)
            if self.with_relu:
                input_tensor = self.relu(input_tensor)
        self.out_shape = self.num_centers * self.redu_dim
        wx_plus_b = self.share_w(input_tensor)
        wx_plus_b = wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2],
                                   self.in_shape[3])

        h_tm1 = torch.autograd.Variable(
            torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        h_tm1 = torch.nn.init.constant(h_tm1, 0).cuda()

        assignments = []

        for i in range(self.timesteps):
            wx_plus_b_at_t = wx_plus_b[:, i, :, :, :]

            Uz_h = self.U_z(h_tm1)
            z = self.sigmoid_U_z(wx_plus_b_at_t + Uz_h)

            Ur_h = self.U_r(h_tm1)
            r = self.sigmoid_U_r(wx_plus_b_at_t + Ur_h)

            Uh_h = self.U_h(r * h_tm1)
            hh = self.tanh_U_h(wx_plus_b_at_t + Uh_h)

            h = (1 - z) * hh + z * h_tm1
            assignments.append(h)
            h_tm1 = h

        ## timesteps, batch_size , num_centers, h, w

        assignments = torch.stack(assignments, dim=0)
        # print('assignments shape', assignments.size())

        ## timesteps, batch_size, num_centers, h, w ==> batch_size, timesteps, num_centers, h, w
        assignments = torch.transpose(assignments, 0, 1).contiguous()
        # print('transposed assignments shape', assignments.size())

        ## assignments: batch_size, timesteps, num_centers, h*w
        assignments = assignments.view(self.batch_size * self.timesteps, self.num_centers,
                                       self.in_shape[2] * self.in_shape[3])
        if self.activation is not None:
            if self.activation == 'softmax':
                assignments = torch.transpose(assignments, 1, 2).contiguous()
                assignments = assignments.view(self.batch_size * self.timesteps * self.in_shape[2] * self.in_shape[3],
                                               self.num_centers)
                assignments = self.sofmax(assignments)  # my_softmax(assignments, dim=1)
                assignments = assignments.view(self.batch_size * self.timesteps, self.in_shape[2] * self.in_shape[3],
                                               self.num_centers)
                assignments = torch.transpose(assignments, 1, 2).contiguous()
            else:
                print('TODO implementation ...')
                exit()

        a_sum = torch.sum(assignments, -1, keepdim=True)

        a = a_sum * self.centers.view(1, self.num_centers, self.redu_dim)
        input_tensor = input_tensor.view(self.batch_size * self.timesteps, self.redu_dim,
                                         self.in_shape[2] * self.in_shape[3])
        input_tensor = torch.transpose(input_tensor, 1, 2)

        x = torch.matmul(assignments, input_tensor)
        vlad = x - a
        vlad = vlad.view(self.batch_size, self.timesteps, self.num_centers, self.redu_dim)
        vlad = torch.sum(vlad, 1, keepdim=False)
        ## intor normalize
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)

        ## l2-normalize
        vlad = vlad.view(self.batch_size, self.num_centers * self.redu_dim)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
        # print('vlad type', type(vlad))
        # print(vlad.size())
        # vlad = torch.Tensor([vlad]).cuda() # NEW line
        if not self.with_center_loss:
            return vlad
        else:
            assignments
            assignments = assignments.view(self.batch_size, self.timesteps, self.num_centers,
                                           self.in_shape[2] * self.in_shape[3])
            assign_predict = torch.sum(torch.sum(assignments, 3), 1)
            return assign_predict, vlad