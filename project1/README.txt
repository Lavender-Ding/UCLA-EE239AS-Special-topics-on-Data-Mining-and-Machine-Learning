All the codes are python
=========================
Network backup
For this dataset, there are two python files: Network Backup.py and Neural Network.py
(The reason for why there are two python files is that the lib our group uses in neural network regression can only be applied on OSX. So we finished 
Network Backup.py on windows and Neural Network.py on OSX seperately)
--Network Backup.py includes answers to all the questions except neural network prediction. 
--Neural Network.py predicts size of backups using neural network regression.
--------------------------------------------------------------------------------------
How to run?
For Network Backup.py, just use the command line "python Network Backup.py" or run it in idle.exe.
For Neural Network, first you need to install the sklearn following the instructions on this site:
http://scikit-neuralnetwork.readthedocs.org/en/latest/guide_installation.html and then type 
"expect LC_ALL=en_US.UTF-8" and "expect LANG=en_US.UTF-8" on osx's terminal. Then you can type 
"python Neural Network.py" on the terminal to run Neural Network.py

(**to be noticed, in the running process, when a figure appears, you have to close the figure window 
to continue the running process.
**The code for Network Backup must be run on python 2.7.10
**Network Backup.py and Neural Network.py uses the preprocessed network_backup_dataset.csv which 
included in the project1 folder. So you need to run Network Backup.py and Neural Network.py in the project1 folder.)
=========================
Boston Housing Dataset
For this project, it just contains one python file: Boston Housing.py. Please make sure that the running 
environment is python 2.7.10
To be noticed, in the running process, when a figure appears, you have to close the figure window 
to continue the running process.