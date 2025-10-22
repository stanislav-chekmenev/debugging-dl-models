import torch

from toy_modules.utils import generate_datasets
from toy_modules.models import RegressorNet
from toy_modules.train import train


if __name__ == "__main__":

    train_loader, test_loader = generate_datasets()
    in_dim = next(iter(train_loader))[0].shape[1]

    regressor = RegressorNet(in_dim=in_dim)
    # Please, use the SGD optimizer.
    optimizer = torch.optim.SGD(regressor.parameters(), lr=0.001, momentum=0.9)

    # Leave the number of epochs unchanged and equal to 200!
    train(
        model=regressor,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=200,
        save_dir="all_bugs",
    )
