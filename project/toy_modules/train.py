import os
import torch

from toy_modules.utils_correct import make_writer


def get_loss(y_pred, y_true):
    l2_loss = torch.mean((y_true - y_pred) ** 2)
    return l2_loss


def train(model, optimizer, train_loader, test_loader, epochs, save_dir):

    writer = make_writer(os.path.join("summary"), save_dir)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        test_loss = 0

        if epoch % 10 == 0:
            print("Epoch {} is running...".format(epoch))

        for x, y in train_loader:
            y_pred = model(x)
            loss = get_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_epoch_loss = train_loss / len(train_loader)

        with torch.no_grad():
            for x, y in test_loader:
                y_pred = model(x)
                test_loss += get_loss(y_pred, y).item()

        test_epoch_loss = test_loss / len(test_loader)
