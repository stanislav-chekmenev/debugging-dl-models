## Breakpoints

- place a breakpoint on line 13 and 32
- start the debugger session with a right click and then choosing 'Debug debugger_intro'
- click play button on the right to move to the next breakpoint
- click button 'step into' when the debugger executes a = int(input("a: "))
- if you want to concentrate on your won code then use the button 'step into my code', which helps to avoid stepping into library classes
- use watch to watch a variable
- use 'Evaluate' to evaluate expressions or write code
- use console to write any code (Alt+Shift+8), as well. Similar to 'Evaluate'

### Managing breakpoints

- Uncomment the text on line 28
- Remove the breakpoint from line 13 and place it on line 28 
- Press Shift+Ctrl+F8
- Choose the breakpoint on line 28 and click disable until hitting the following breakpoint (breakpoint on line 32)
- Run debugger

## Useful shortcuts

- Ctrl+b - find the definition of a function or a class highlighted by the cursor
- Ctrl+Shift+N - search for files, classes, etc:
    - For exmaple, type 'Dense' in classes and you can go immediately to the place where the class for keras Dense() layer is defined. 
- Shift+Ctrl+Backspace - move through the list of the most recent change points

- Shift+Ctrl+F8 - view and manage all breakpoints
- Shift+F9 - run debugger for the selected script
- Shift+F10 - run the selected script
