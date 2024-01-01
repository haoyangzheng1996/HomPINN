import torch
import random
import numpy as np

from model.utils import select_optimizer, load_model
from model.dataloader import loader
from model.flags import get_flags
from model.events import Event
from model.msnn import MSNN


def run_func(args):

    device = args.set_device

    # Set parameters
    alpha_0 = args.alpha
    r = args.decay_rate
    Hom_step = args.n_step

    # Data generation
    dataloader = loader(data_dir=args.data_dir, N_each_sols=args.num_obs, N_collocation=args.num_col)

    # Network
    Layers = [args.num_input]
    for i in range(args.depth - 2):
        Layers.append(args.width)
    Layers.append(args.num_sol)
    
    u_net = MSNN(Layers).to(device)
    u_net = load_model(u_net, args.model_dir, device)  # load previous model
    
    # Optimizer
    optimizer = select_optimizer(u_net, optim=args.optimizer)
    
    # Configure event
    lr = [args.learning_rate, args.lr_low1, args.lr_low2]
    event = Event(u_net, optimizer, dataloader, max_epochs=args.n_epoch, device=device, learn_rate=lr,
                  save_path=args.model_dir, fig_path=args.fig_dir)

    # Plot observations
    event.plot_data()

    # Homotopy process
    for step in range(Hom_step):
        alpha = alpha_0 * r ** step
        print('Step: {}; alpha: {:.4f}'.format(step + 1, alpha))
        if step == Hom_step-1:
            event.epoch = int(20000)
        event.train_each_step(alpha, step)
        event.save_model()

    # Plot prediction
    event.plot_pred()

    print("Homotopy steps completed.")


if __name__ == "__main__":

    # Get flags
    flags = get_flags()
    
    random.seed(flags.seed)
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)

    run_func(flags)
