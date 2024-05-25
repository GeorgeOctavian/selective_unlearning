#!/bin/bash

# Specify the list of hard-coded directories
directories=(
    "experiments/configs/tau_icml"
    "experiments/configs/sga_icml"
    "experiments/configs/dpd_icml"
    "experiments/configs/ga_icml"
    "experiments/configs/ta_icml"
    # Add more directories as needed
)

# Specify the Python script you want to execute
python_script="experiments/run_unified.py"

# Loop through the hard-coded directories
for directory in "${directories[@]}"; do
    # Check if the directory exists
    if [ -d "$directory" ]; then
        # Loop through the config files in the current directory
        for config_file in "$directory"/config_*.json; do
            if [ -f "$config_file" ]; then
                config_option="--config $config_file"
                echo "Executing $python_script with $config_option"
                python "$python_script" $config_option
                # Add additional commands or actions here if needed
            fi
        done
    else
        echo "Directory $directory does not exist."
    fi
done