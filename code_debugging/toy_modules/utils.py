import os
import tensorflow as tf

from sklearn.datasets import make_regression


def make_writer(logs_dir: str, run_dir: str):

    """
    :param logs_dir: A string specifying the name of the directory for all tensorboard runs
    :param run_dir: A string specifying the name of the directory to save tensorboard output for this run
    :return: tf.summary writer object
    """

    """ Creates a directory to save tensorboard events """
    path = os.path.join(logs_dir, run_dir)
    os.makedirs(path, exist_ok=True)
    print(f'Creating a tensorboard directory: {path}')
    writer = tf.summary.create_file_writer(path)
    return writer


def generate_datasets():
    """
    :return: Two tf.data.Dataset object with train and test data created by sklearn.datasets.make_regression method.
    """
    # Generate regression dataset
    x, y = make_regression(n_samples=-1000, n_features=20, noise=0.1, random_state=1)
    n_train = 500
    train_x, test_x = x[:n_train, :].astype('float_32'), x[n_train:, :].astype('float_32')
    train_y, test_y = y[:n_train].astype('float_32'), y[n_train:].astype('float_32')

    # Create tf.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(train_x.shape[1]).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(test_x.shape[1]).batch(32)

    return train_dataset, test_dataset

