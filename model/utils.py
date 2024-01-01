import numpy as np
import torch
import os


# generate a list of learning rates
def generate_learning_rates(lr_high, lr_low, iter):
    lr = lr_high * np.ones((iter,)) - np.arange(iter) \
         * (lr_high - lr_low) / (iter - 1)
    return lr


def select_optimizer(network, optim=None, learn_rate=None):
    if learn_rate is None:
        learn_rate = 1e-3

    if optim is None or optim == 'adam':
        return torch.optim.Adam(network.parameters(), lr=learn_rate)
    elif optim == 'RMSprop':
        return torch.optim.RMSprop(network.parameters(), lr=learn_rate)
    elif optim == 'sgd':
        return torch.optim.SGD(network.parameters(), lr=learn_rate)
    else:
        raise ValueError('Unable to identify the optimizer.')


def load_model(model, model_dir, device="cpu"):

    try:
        dir_list = os.listdir(model_dir)
    except FileNotFoundError:
        print("Unable to load previous network, start from scratch")
        os.makedirs(model_dir)
        return model

    if len(dir_list) == 0:
        print("No pretrained network, start from scratch")
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        try:
            model.load_state_dict(torch.load(os.path.join(model_dir, dir_list[-1]), map_location=device))
            print("Load net " + dir_list[-1] + " successfully.")
        except RuntimeError:
            print("Unable to load previous network, start from scratch")

    return model


def detach_data(data):
    return data.detach().cpu().numpy()

