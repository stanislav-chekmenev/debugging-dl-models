import tensorflow as tf

class RegressorNet(tf.keras.Model):
    """
    A class for solving regression problems.
    """
    def __init__(self, input_shape, optimizer):
        super(RegressorNet, self).__init__()

        self.optimizer = optimizer
        self.dense_1_1 = tf.keras.layers.Dense(
            64, activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='dense_1_1'
        )
        self.dense_1_2 = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='dense_1_2'
        )
        self.concat = tf.keras.layers.Concatenate(name='concat')
        self.dense_2 = tf.keras.layers.Dense(
            32, activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='dense_2'
        )
        self.out = tf.keras.layers.Dense(
            1, activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='dense_out'
        )

        # Create network logic
        x_input = tf.keras.Input(shape=input_shape)
        x1 = x_input[:, :x_input.shape[1] // 2]
        x2 = x_input[:, x_input.shape[1] // 2:]

        h1 = self.dense_1_1(x1)
        h2 = self.dense_1_2(x1)
        h = self.concat([h1, h2])
        h = self.dense_2(h)
        out = self.out(h)

        self.regressor = tf.keras.Model(inputs=x_input, outputs=out)

    def call(self, x):
        return self.regressor(x)

    def get_loss(self, x, y_true):
        y_pred = self(x)
        l2_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return l2_loss

    def grad_step(self, x, y_true):
        with tf.GradientTape() as tape:
            loss = self.get_loss(x, y_true)
        gradients = tape.gradient(loss, self.trainable_variables[1:2])
        gradients = tf.clip_by_global_norm(gradients, 1.0)[0]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables[1:2]))
        return loss, gradients
