import os, re
from main import determine_next_filename

with open('starter','r') as file:
    for line in file.readlines():
        if 'cpus-per-task' in line:
            cpus_per_task = int(line.split('=')[-1])
        elif 'ntasks' in line:
            ntasks = int(line.split('=')[-1])
        elif 'array' in line:
            match = re.search(r'(\d+)-(\d+)%\d+', line)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))

                # Generate the list
                array_size = len(list(range(start, end + 1)))
                break

print(cpus_per_task, ntasks, array_size)


dirname = determine_next_filename('run',folder='output_data',direc=True)
# os.mkdir(dirname)

print(dirname)