# Ex 1

# Standardizing:
def standardize(array):
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return (array - means) /  std

# Reinitialize the model instance
optimizer = tf.keras.optimizers.Adam()
model = RegressorNet(input_shape=train_set.shape[1], optimizer=optimizer)
# Scale
stand_train_set = standardize(train_set)
# Retrain
train(model, 1000, stand_train_set, labels, 'scaling/regression_standard')



# Ex 2
model = MyModel()
# Fix the loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Retrain
EPOCHS = 5
writer = make_writer(os.path.join('summaries'), 'loss_bug/logits_true')

for epoch in range(EPOCHS):

    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_ds:
        gradients = train_step(images, labels)

    # Tensorboard
    with writer.as_default():
        tf.summary.scalar('Train loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Train Accuracy', train_accuracy.result() * 100, step=epoch)
        
        for layer_number, layer in enumerate(model.trainable_variables):
            tf.summary.histogram('/'.join(layer.name.split('/')[1:]), gradients[layer_number], step=epoch, buckets=1)    

    message = (f'Epoch: {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}'
              )
    print(message)  


