import os
from main import determine_next_filename

dirname = determine_next_filename('run',folder='output data',direc=True)
os.mkdir(dirname)