# Ex 1

# Relu
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))

with tf.GradientTape() as tape:
    y = tf.nn.relu(x)

grad = tape.gradient(y, x).numpy()
x = x.numpy()
y = y.numpy()
    
plt.plot(x, y, 'b', label='relu')
plt.plot(x, grad, 'r', label='gradient') 
plt.legend()
plt.show()

# LeakyRelu
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))

with tf.GradientTape() as tape:
    y = tf.nn.leaky_relu(x)

grad = tape.gradient(y, x).numpy()
x = x.numpy()
y = y.numpy()
    
plt.plot(x, y, 'b', label='leaky relu')
plt.plot(x, grad, 'r', label='gradient') 
plt.legend()
plt.show()


# Ex 2

# Clipped regressor
class RegressorNetClipped(tf.keras.Model):
    def __init__(self, input_shape, optimizer):
        super(RegressorNetClipped, self).__init__()
        
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform', name='dense_1'),
            tf.keras.layers.Dense(1, activation='linear', name='out')
        ])
    
    def summary(self):
        self.regressor.summary()
    
    def call(self, X):
        return self.regressor(X)
    
    def get_loss(self, X, y_true):
        y_pred = self(X)
        l2_loss = self.loss_object(y_true, y_pred)
        return l2_loss
    
    def grad_step(self, X, y):
        with tf.GradientTape() as tape:
            loss = self.get_loss(X, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = tf.clip_by_global_norm(gradients, 2.0)[0]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, gradients
    
    
# Make a new model
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model_clipped = RegressorNetClipped(input_shape=trainX.shape[1], optimizer=opt)

# Train
train(model_clipped, 100, train_dataset, test_dataset, 'exploding_grads/clipped/')