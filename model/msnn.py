import torch
import torch.nn as nn


class MSNN(nn.Module):
    def __init__(self, Layers, parameter=17):
        super(MSNN, self).__init__()

        self.layers = Layers
        self.net = self.network()
        self.lambda0 = nn.Parameter(torch.tensor(1.5))
        self.p = parameter

    def network(self):

        last_layer = nn.Linear(self.layers[-2], self.layers[-1])
        net = [nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
             for input_, output_ in
             zip(self.layers[:-1], self.layers[1:-1])] + \
            [nn.Sequential(last_layer)]
        return nn.Sequential(*net)

    def forward(self, x):
        u = self.net(x.reshape(-1, 1))
        return u

    def get_loss1(self, x_obs, U_obs):

        U_approx = self.forward(x_obs)
        num_sol = U_approx.shape[1]
        err = (U_approx[:, 0:1] - U_obs) ** 2
        if num_sol > 1:
            for i in range(num_sol - 1):
                err_0 = (U_approx[:, i + 1:i + 2] - U_obs) ** 2
                err = torch.where(err < err_0, err, err_0)
        return torch.mean(err)

    def get_loss2(self, x_col):

        xc = x_col.requires_grad_()
        Uc = self.forward(xc)
        Ux = torch.ones_like(Uc)
        Uxx = torch.ones_like(Uc)
        num_sol = Uc.shape[1]
        for i in range(num_sol):
            Ux[:, i:i + 1] = torch.autograd.grad(Uc[:, i:i + 1].sum(), xc, create_graph=True)[0]
            Uxx[:, i:i + 1] = torch.autograd.grad(Ux[:, i:i + 1].sum(), xc, create_graph=True)[0]
        F = Uxx + self.lambda0 * (1 + Uc ** 4)

        return torch.mean(F ** 2)

    def Predict(self, x):
        x_tensor = torch.Tensor(x)
        u = self.forward(x_tensor)
        u_numpy = u.cpu().detach().numpy()
        return u_numpy
