## Breakpoints

### Finding and fixing a bug

- Try running the script and check for average speed by pressing S.
- Put a breakpoint on lines 26 and 32. Run debugger.
- Check how the debugger shows the error message and a lightning bolt to mark the place 
with the exception.
- In the debugger console we can also see that self.time is 0.
- Let's debug and change the code.

### Debugging in detail

#### Stepping

- Run the debugger and press `Continue` (F5).
- Press `step over` (F10) to execute step by step.
- When on the line with `input` press `Step into` (F11).

#### Watching

- Go to `Run and Debug` and in the `Variables` section press `+` and add `my_car.time` to watch.
It will be watched and displayed there from no on. 
- You can remove it from there by a right click and pressing `Remove watch`

#### Evaluating expressions

- You can press Ctrl+Shift+Y to go to the debug console.


## Useful shortcuts

- Ctrl+F12 - find the definition of a function or a class highlighted by the cursor
- Ctrl+Shift+F - search for files, classes, etc:
    - For example, type 'Dense' in classes and you can go immediately to the place where the class for keras Dense() layer is defined. 
- Ctrl+Alt+- - move through the list of the most recent change points backwards
- F5 - run debugger for the selected script
- Ctrl+F5 - run the selected script
