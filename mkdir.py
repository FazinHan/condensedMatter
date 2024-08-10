import os
from main import determine_next_filename

dirname = determine_next_filename('run',folder='output_data',direc=True)
os.mkdir(dirname)

print(dirname)