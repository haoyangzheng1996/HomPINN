from argparse import ArgumentParser
import torch


def get_flags():
    parser = ArgumentParser(description='Argument Parser')

    parser.add_argument(
        "--set_device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="Set up set_device")
    parser.add_argument("--seed", default=888, type=int, help="Random seed")

    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--lr_low1", default=9e-5, type=float, help="Learning rate 2")
    parser.add_argument("--lr_low2", default=1e-5, type=float, help="Learning rate 3")

    parser.add_argument("--num_input", default=1, type=int, help="Data dimensions")
    parser.add_argument("--num_sol", default=2, type=int, help="Number of solutions")
    parser.add_argument("--num_obs", default=40, type=int, help="Number of observations for each solution")
    parser.add_argument("--num_col", default=100, type=int, help="Number of collocations")
    parser.add_argument("--num_test", default=50, type=int, help="Number of testing data")
    parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs for each homotopy step")
    parser.add_argument("--n_step", default=10, type=int, help="Number of homotopy steps")

    parser.add_argument("--alpha", default=1.0, type=int, help="Initial homotopy tracking parameter")
    parser.add_argument("--decay_rate", default=0.6, type=int, help="Number of homotopy steps")

    parser.add_argument("--width", default=30, type=float, help="Number of units per layer for the hidden layers")
    parser.add_argument("--depth", default=5, type=float, help="Number of layers for the model")

    parser.add_argument("--save_after", default=10000, type=int, help="Save net after number of epochs")
    parser.add_argument("--data_dir", default='./data/obs.mat', type=str,
                        help="Directory for loading dataset")
    parser.add_argument("--model_dir", default="./logs/model/", type=str, help="Directory for trained model")
    parser.add_argument("--fig_dir", default="./logs/figure/", type=str, help="Directory for figures")

    args = parser.parse_args()

    return args
