import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.datasets import make_regression


def make_writer(logs_dir: str, run_dir: str):
    """
    :param logs_dir: A string specifying the name of the directory for all tensorboard runs
    :param run_dir: A string specifying the name of the directory to save tensorboard output for this run
    :return: SummaryWriter object
    """

    """ Creates a directory to save tensorboard events """
    path = os.path.join(logs_dir, run_dir)
    os.makedirs(path, exist_ok=True)
    print(f"Creating a tensorboard directory: {path}")
    writer = SummaryWriter(log_dir=path)
    return writer


def generate_datasets():
    """
    :return: Two torch DataLoader objects with train and test data created by sklearn.datasets.make_regression method.
    """
    # Generate regression dataset
    x, y = make_regression(n_samples=-1000, n_features=20, noise=0.1, random_state=1)
    n_train = 500
    train_x, test_x = x[:n_train, :].astype("float_32"), x[n_train:, :].astype("float_32")
    train_y, test_y = y[:n_train].astype("float_32"), y[n_train:].astype("float_32")

    train_tensor_x = torch.from_numpy(train_x)
    train_tensor_y = torch.from_numpy(train_y)
    test_tensor_x = torch.from_numpy(test_x)
    test_tensor_y = torch.from_numpy(test_y)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
