## Exercise

- Explore the structure of the project and find out where each module is located and defined. Make sure you understand 
the arguments of the functions and methods being specified and called.
- Get your script to run without throwing exceptions.
- Modify train.py to **print** the losses and **output them to tensorboard**.
    - Hint: You can use the function make_writer in the toy_modules.utils.
    - Hint: Check the Colab: 2_most_common_bugs and the implementation of how to write the loss and the gradients to 
    tensorboard. The tensorboard part code is given below: 
    ```python
    with writer.as_default():
        tf.summary.scalar('Train loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
    ```

- Find the bugs that affect the learning:
    - For each bug found create a new tensorboard directory to compare results. 
    (Example: found bug X --> TB output goes into summaries/bug_X)