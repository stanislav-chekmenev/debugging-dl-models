## Exercise

- Get your script to run without throwing exceptions.
- Modify train.py to print the losses and output them to tensorboard. Send the output to summaries. 
You can use the function make_writer in the toy_modules.utils.


- Find the bugs that affect learning:
    - For each bug found create a new tensorboard directory to compare results. 
    (Example: found bug X --> TB output goes into summaries/bug_X)