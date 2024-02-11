## Breakpoints

### Finding and fixing a bug

- Try running the script and check for average speed by pressing S.
- Put a breakpoint on lines 26 and 32. Run debugger.
- Press `Resume program` on the left to go to the next breakpoint.
- Check how the debugger shows the error message and a lightning bolt to mark the place 
with the exception.
- In the debugger console we can also see that self.time is 0.
- Let's debug and change the code.

### Debugging in detail

#### Stepping

- Run the debugger and press `Resume program`.
- Press `step over` to execute step by step.
- When on the line with `input` press `Step into`.
- Place caret and press `Run to cursor`.
- When on my_car.step() press `Step into my code`

#### Watching

- Go to `Debugger` and in the `Variables` section press `+` and add `my_car.time` to watch.
It will be watched and displayed there from no on. 
- You can remove it from there by a right click and pressing `Remove watch`

#### Evaluating expressions

- You can press Shift+Alt+8 or `Evaluate expression`.
- It's a nice view to evaluate any expression or change some variables, for example.
- You can also use `Console` to write any code, check, reassign variables, etc.


## Useful shortcuts

- Ctrl+b - find the definition of a function or a class highlighted by the cursor
- Ctrl+Shift+N - search for files, classes, etc:
    - For example, type 'Dense' in classes and you can go immediately to the place where the class for keras Dense() layer is defined. 
- Shift+Ctrl+Backspace - move through the list of the most recent change points

- Shift+Ctrl+F8 - view and manage all breakpoints and their settings
- Shift+F9 - run debugger for the selected script
- Shift+F10 - run the selected script
