#!/bin/bash

# Compile the program
g++ -I"./include" src/program.cpp -o program

# Check if compilation was successful
if [ $? -eq 0 ]; then
    # Run the program if compilation succeeded
    ./program
else
    echo "Compilation failed."
fi
