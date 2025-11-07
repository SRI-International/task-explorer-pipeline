If there are any issues running the TEP, please contact me at thomas.a.odem@gmail.com

Further details about the architecture of the TEP can be found in my thesis paper [here](https://arxiv.org/abs/2511.01728)

# Requirements
All requirements are found in requirements.txt. I would suggest making a virtual enviornment with the requirements.

# Input Data format
The input data (which should be placed in the /data/ directory) should be a csv with the following columns, sorted by time per participant per task
- action: the action completed by the participant
- participant: the participant/user who completed the action
- task: the task the participant completed the action for

For example, if participant "1" completed the actions "wave", "smile", "walk" in that sequence for the task "greet", the csv should look like:
```
action,participant,task
wave,1,greet
smile,1,greet
walk,1,greet
```

ASCEND data can be found at [ascend-data.sri.com](https://ascend-data.sri.com). Download a zip file ending with "-clean-data.zip" and extract the 300_wf.csv file and the 200_terminal_features.csv files.

# Running the TEP
If running on ASCEND data, then ensure you have the 300_wf.csv file and the 200_terminal_features.csv file in the data directory. Also set the "ascend_ctf_data" boolean to True, otherwise set to false and rename "csv_file_name"

First, run cyber_wf_pipeline_parameter_finding.ipynb (I use VS Code for Jupyter notebooks) for every task in your dataset to identify the parameters you want the TEP to run the tasks on. The parameters will be below markdown cells with text colored yellow, which explain what the parameters are/do. 

While identifying the optimal parameters for each task, add them to the get_parameters() function in custom_funcs_for_wf.py so that the TEP can automatically exract the parameters when running.

With all parameters for every task in your data defined, run the cyber_wf_pipine_task_explorer.ipynb file (directing it to the same data directory). There are some options that can be turned on or off during the TEP run, which are all in the "Options" section and have definitions outlined there. One of the options includes saving data to be used in a Task Explorer Application.

There are various forms of the TEP results at the end, all in the form of pandas dataframes that can be easily exported or used for other purposes. Additionally, there is a "save_report" boolean that can be switched to True in order to save csv files that contain easily interpretable results for the top 10 most used strategies and subtasks for each task.


# Task Explorer Application

The Task Explorer application is a Windows executable developed in Rust using the egui library. Also attached in this folder is a pdf that walks through the Task Explorer Application's functionality and also highlights what each computed statistic means (as well as how statistics are formatted, in the case that one would like to add more).


To open, simply run the application like any other Windows exectuable (like double-clicking to open).


The Task Explorer application requires artifacts produced from the Task Explorer Pipeline (TEP). The TEP can be run using the cyber_wf_pipeline_task_explorer.ipynb file (ensure sure you have all input data as well) with the "save" value set to "true" and a satisfiable directory listed in the directory_to_save_json_objects variable. The TEP will create three JSON files (runs, statistics, and subtasks) and an "images" folder filled with spider graphs and user hierarchical encoding diagrams in the form of png files.


With the produced artifacts, simply open the Task Explorer application and use the in-app utility to select the folder with the TEP artifacts (the Task Explorer application will crash if you do not select a folder with artifacts!).

Please Note: The Task Explorer has only been tested on a native Windows 11 machine and Windows 11 running in VMWare on MacOS, but should be able to run on any modern Windows device.


