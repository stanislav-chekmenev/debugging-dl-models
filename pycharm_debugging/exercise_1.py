import tensorflow as tf

from toy_modules.utils import generate_datasets
from toy_modules.models import RegressorNet
from toy_modules.train import train


if __name__ == '__main__':

    # Create two metric objects to calculate mean loss across batches
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Create an optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # Generate datasets
    train_dataset, test_dataset = generate_datasets()

    # Create the model
    regressor = RegressorNet(input_shape=40, optimizer=optimizer)

    # Train (leave the number of epochs unchanged and equal to 200)
    train(
        model=regressor,
        epochs=200,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loss=train_loss,
        test_loss=test_loss,
        save_dir='test1'
    )
