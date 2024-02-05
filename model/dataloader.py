import numpy as np
import random
import torch
import scipy
import scipy.io


class loader:
    def __init__(self, data_dir, N_sols=2, N_each_sols=40, N_collocation=100):

        self.n_sols = N_sols
        self.data_dir = data_dir
        self.n_col = N_collocation
        self.n_each_sol = N_each_sols

        self.x_obs, self.u_obs, self.collocation, self.data_all = self.load_data()

    def load_data(self):

        x_collocation = (np.arange(self.n_col) / (self.n_col - 1)).reshape((-1, 1))
        xg_collocation = torch.Tensor(x_collocation)
        data_obs = scipy.io.loadmat(self.data_dir)['data']
        num_sols_tol = data_obs.shape[1] - 1
        idx_class = np.random.choice(np.arange(num_sols_tol) + 1, size=self.n_sols, replace=False)
        U_obs, x_obs = generate_obs(data_obs, idx_class, self.n_each_sol)
        U_obst = torch.Tensor(U_obs.reshape((-1, 1)))
        x_obst = torch.Tensor(x_obs.reshape((-1, 1)))

        return x_obst, U_obst, xg_collocation, data_obs


def generate_obs(data_obs, idx, N_each_solution):
    size = data_obs.shape
    m = size[0]
    num_class = len(idx)
    n_per = N_each_solution
    if m < n_per * num_class:
        raise ValueError('The data set is not large enough.')
    index = list(range(m))
    idx_loc = random.sample(index, n_per)
    U_obs = data_obs[idx_loc, idx[0]:idx[0] + 1]
    X_obs = data_obs[idx_loc, 0:1]
    for i in range(1, num_class):
        for j in range(n_per):
            index.remove(idx_loc[j])
        idx_loc = random.sample(index, n_per)
        U_obs = np.concatenate((U_obs, data_obs[idx_loc, idx[i]:idx[i] + 1]))
        X_obs = np.concatenate((X_obs, data_obs[idx_loc, 0:1]))
    return U_obs, X_obs
