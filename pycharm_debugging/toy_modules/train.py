import os
import tensorflow as tf

from toy_modules.utils import make_writer


def train(model, epochs, train_dataset, test_dataset, save_dir, train_loss, test_loss):

    """

    :param model: Model instance
    :param epochs: Number of epochs to train (set it to 200)
    :param train_loss: tf.keras.metrics.Mean() object
    :param test_loss: tf.keras.metrics.Mean() object
    :param train_dataset: tf.data.Datasets object for train data
    :param test_dataset: tf.data.Datasets object for test data
    :param save_dir: Directory to save tensorboard output
    :return: None
    """

    writer = make_writer(os.path.join('summaries/exercise_1'), save_dir)

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

        if epoch % 10 == 0:
            print(f'Train loss: {train_loss.result()}. Test loss {test_loss.result()}')

        # Tensorboard output
        with writer.as_default():
            tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
            tf.summary.scalar('Train loss', train_loss.result(), step=epoch)
            for layer_number in range(len(gradients)):
                if layer_number % 2 == 0:
                    tf.summary.histogram(
                        f'{model.layers[layer_number // 2].name}/kernel_0',
                        gradients[layer_number], step=epoch, buckets=1
                    )
                else:
                    tf.summary.histogram(
                        f'{model.layers[(layer_number - 1) // 2].name}/bias_0',
                        gradients[layer_number], step=epoch, buckets=1
                    )