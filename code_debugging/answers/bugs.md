### The bugs:

- `float_32` instead of `float32`, `-1000` for the `n_samples`
- shape mismatch in loss
- learning rate too low
- only 1 trainable layer
- disconnected layers
- tanh in the first layer
- relu in the last layer
- shuffle according to the shape[1]
- wrong initializer/try glorot_normal