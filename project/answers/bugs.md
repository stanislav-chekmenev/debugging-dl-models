### The bugs:

- `float_32` instead of `float32`, `-1000` for the `n_samples`
- no optimizer.zero_grad()
- shape mismatch in the loss between target and true values
- no gradient cliping which leads to huge updates
- model.eval() missing
- relu in the last layer
- disconnected input layers
- learning rate too low
- tanh in the first layer
- wrong train/test split ratio
- no shuffle for the train set
  