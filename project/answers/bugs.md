### The bugs:

- `float_32` instead of `float32`, `-1000` for the `n_samples`
- only 1 trainable layer
- shape mismatch in the loss
- disconnected layers
- learning rate too low
- relu in the last layer
- tanh in the first layer
- shuffle according to the shape[1]