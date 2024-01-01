import os
import torch
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from model.utils import generate_learning_rates, detach_data


class Event:
    def __init__(self, model, optimizer, loader, max_epochs, lr_tol=20000, learn_rate=None,
                 save_path=None, fig_path=None, device=None):

        self.model = model
        self.loader = loader
        self.optimizer = optimizer

        self.epoch = max_epochs
        self.lr_tol = lr_tol

        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        
        if learn_rate is not None:
            self.learn_rate = learn_rate

        if save_path is not None:
            self.path = save_path

        if fig_path is not None:
            self.fig_path = fig_path

    def train_each_step(self, alpha, hom_step):

        if self.epoch <= self.lr_tol:
            lrseq = generate_learning_rates(self.learn_rate[0], self.learn_rate[1], self.epoch) * 0.9 ** hom_step
        else:
            lrseq = generate_learning_rates(self.learn_rate[0], self.learn_rate[2], self.epoch) * 0.9 ** hom_step

        num_epoch = trange(self.epoch, desc="HomPINN Training")

        for epoch in num_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrseq[epoch]

            # data
            x_obs = self.loader.x_obs.clone().requires_grad_().to(self.device)
            x_col = self.loader.collocation.clone().requires_grad_().to(self.device)
            U_obs = self.loader.u_obs.to(self.device)

            # loss
            error1 = self.model.get_loss1(x_obs, U_obs)
            error2 = self.model.get_loss2(x_col)
            loss = error1 + alpha * error2

            # optimize
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # record
            num_epoch.set_postfix({
                'loss1': '{0:1.2e}'.format(error1),
                'loss2': '{0:1.2e}'.format(error2),
                'Total': '{0:1.2e}'.format(loss),
                'Lambda': '{0:1.4f}'.format(self.model.lambda0)})

    def save_model(self, model_path=None):

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")
        if model_path is None:
            model_path = self.path

        try:
            torch.save(self.model.state_dict(), model_path + "/model_" + time + ".pt")
        except FileNotFoundError:
            os.makedirs(model_path)
            torch.save(self.model.state_dict(), model_path + "/model_" + time + ".pt")


        print("Save model_" + time + ".pt successfully.")

    def plot_data(self, path=None):

        if path is None:
            path = self.fig_path

        x = self.loader.data_all[:, 0:1]

        plt.figure(figsize=(7, 6), dpi=150)

        plt.scatter(detach_data(self.loader.x_obs), detach_data(self.loader.u_obs),
                    marker='o', s=100, linewidths=3, c='none', edgecolors='k', label='obs')
        plt.plot(x, self.loader.data_all[:, 1], color='blue', linewidth=4, label='u_1')
        plt.plot(x, self.loader.data_all[:, 2], color='red', linewidth=4, label='u_2')

        plt.xlabel('x', fontsize=20)
        plt.ylabel('u(x)', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=20, loc='best')

        # plt.show()
        plt.tight_layout()

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = '/figure_data_' + time + '.pdf'
        try:
            plt.savefig(path + '/' + file_name)
        except FileNotFoundError:
            os.makedirs(path)
            plt.savefig(path + '/' + file_name)
        print('Save data figure figure_data_' + time + ' successfully.')

    def plot_pred(self, path=None):

        if path is None:
            path = self.fig_path

        x_min, x_max = np.min(self.loader.data_all[:, 0:1]), np.max(self.loader.data_all[:, 0:1])
        x = np.arange(1001) / 1000 * (x_max - x_min)
        x = torch.Tensor(x).clone().requires_grad_().to(self.device)
        u = self.model.forward(x.reshape(-1, 1))

        plt.figure(figsize=(7, 6), dpi=150)

        plt.scatter(detach_data(self.loader.x_obs), detach_data(self.loader.u_obs),
                    marker='o', s=100, linewidths=3, c='none', edgecolors='k', label='obs')
        plt.plot(detach_data(x), detach_data(u[:, 0]), color='blue', linewidth=4, label='u_1')
        plt.plot(detach_data(x), detach_data(u[:, 1]), color='red', linewidth=4, label='u_2')

        plt.xlabel('x', fontsize=20)
        plt.ylabel('u(x)', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=20, loc='best')

        # plt.show()
        plt.tight_layout()

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = '/figure_pred_' + time + '.pdf'
        try:
            plt.savefig(path + '/' + file_name)
        except FileNotFoundError:
            os.makedirs(path)
            plt.savefig(path + '/' + file_name)
        print('Save prediction figure figure_pred_' + time + ' successfully.')
