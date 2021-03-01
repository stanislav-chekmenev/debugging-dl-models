import os
import tensorflow as tf

def make_writer(filepath, dir_name):
    """ Creates a directory to save tensorboard events """
    path = os.path.join(filepath, dir_name)
    os.makedirs(path, exist_ok=True)
    print(f'Creating a tensorboard directory: {path}')
    writer = tf.summary.create_file_writer(path)
    return writer