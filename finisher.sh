#!/bin/bash

cd output_data

python new_version.py

echo "Data moved into directory"

cd ..

python plotter.py save

echo "plot created"

cd output_data

python to_scratch.py

echo "data sent to scratch"

push