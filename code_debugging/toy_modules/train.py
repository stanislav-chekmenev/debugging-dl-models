import os
import tensorflow as tf

from toy_modules.utils import make_writer


def train(model, epochs, train_dataset, test_dataset, save_dir):

    """
    :param model: Model instance
    :param epochs: Number of epochs to train (set it to 200)
    :param train_dataset: tf.data.Datasets object for train data
    :param test_dataset: tf.data.Datasets object for test data
    :param save_dir: Directory to save tensorboard output
    """

    writer = make_writer(os.path.join('summaries'), save_dir)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    for epoch in range(0, epochs + 1):

        train_loss.reset_states()
        test_loss.reset_states()

        if epoch % 10 == 0:
            print('Epoch {} is running...'.format(epoch))

        for X, y in train_dataset:
            # Gradient update step
            loss_train, gradients = model.grad_step(X, y)
            train_loss(loss_train)

        for X, y in test_dataset:
            # Test loss calculation
            loss_test = model.get_loss(X, y)
            test_loss(loss_test)
