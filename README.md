Run with python3 assign_roommates.py.
See options with python3 assign_roommates.py --help.
Default input files are preferences.csv, rooms.csv, and default output file is output.csv, but you can change these!

If "Could not find optimal solution" is printed, then no feasible assignment meeting all constraints was found :(

Need numpy, pandas, and pulp installed before. Numpy and pandas are pretty standard, you can use pip or conda to install (I recommend creating a conda environment to use them). For pulp, use pip to install (https://pypi.org/project/PuLP/) -> python-m pip install pulp


TODOs:
allow for single rooms!
add some sort of room preferences??