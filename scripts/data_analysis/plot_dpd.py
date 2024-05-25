# steps for experiment 1: 
# pull data specific to 125M parameter 
# compute average metrics per algorithm, per task
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
import wandb
import os 
import re
import json 
# import pandas as pd 
import argparse
from argparse import ArgumentParser
from scipy import stats
# Assuming you have a sample data array called 'data'
# confidence_interval = stats.norm.interval(0.95, loc=np.mean(data), scale=np.std(data))
# print("Confidence Interval:", confidence_interval)

from scipy.stats import t


api = wandb.Api()


tr_dict = {'gpt-neo-2.7b': 34.02, 'gpt-neo-1.3b':33.27, 'gpt-neo-125M': 29.94}

sample_size =  'lm_extraction_16_'
model_name = 'gpt-neo-125M'

# TODO: add your wandb space here. 
runs = api.runs("<wandbspace>")


summary_list, name_list = [], []
for run in runs:

    if sample_size in run.name and model_name in run.name:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        # config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)



# Define a function to perform some operation on the 'name' column
def create_algo(name):
    # Example: Convert the names to uppercase
    
    if 'og' in name: 
        return 'OG'
    elif 'dp' in name:
        return 'DPD'


def extract_index(file_name):
    match = re.search(r'(\d+)', file_name)
    return match.group() if match else None

# Function to load val_data_idx.json with the highest index from a directory
def load_json(directory_path):
    json_files = [file for file in os.listdir(directory_path) if file.startswith('val_data_') and file.endswith('.json') and not file.startswith('val_data_init')]
    
    # Check if there are any val_data_idx.json files in the directory
    if json_files:
        # Find the val_data_idx.json file with the highest index
        highest_index_file = max(json_files, key=lambda x: int(extract_index(x)))
        
        if extract_index(highest_index_file) is not None:
            highest_index_path = os.path.join(directory_path, highest_index_file)
            
            with open(highest_index_path, 'r') as json_file:
                data = json.load(json_file)
                # Process or use the loaded data as needed
                # print(f"Loaded data from {highest_index_path}: {data}")
                # Format the index with leading zeros
                formatted_index = extract_index(highest_index_file).zfill(2)
                
                # Extract array "acc_idx" with the formatted index
                acc_idx_array = data.get(f"acc_init", [])

                below_015_count = sum(1 for num in acc_idx_array if num <= 0.15)
                above_060_count = sum(1 for num in acc_idx_array if num >= 0.60)

                return below_015_count, above_060_count, np.std(acc_idx_array)
        else:
            print(f"Error: Unable to extract index from {highest_index_file}")
    else:
        print(f"No val_data_idx.json files found in {directory_path}")
        return 0, 0

# Function to loop through directories and load data.json
def process_directories(parent_directory, target_string, possible_starts):
    # Create an empty DataFrame
    df_list = []
    
    # Loop through directories in the parent directory
    for directory_name in os.listdir(parent_directory):
        directory_path = os.path.join(parent_directory, directory_name)
        
        # Check if the directory contains the target string
        if os.path.isdir(directory_path) and target_string in directory_name and any(directory_name.startswith(start) for start in possible_starts):
            # Load data.json from the specific directory
            below_015_count, above_060_count, stddv = load_json(directory_path)
            
            # Append data to the list
            df_list.append({'name': directory_name, 'below_015_count': below_015_count, 'above_060_count': above_060_count, 'total_extractable': above_060_count + below_015_count, 'Memory_std': stddv})
            
    # Create DataFrame from the list
    df = pd.DataFrame(df_list)

    return df


runs_df = pd.DataFrame(summary_list)
runs_df["name"] = name_list

mem_df = process_directories("llmu_results", model_name, ["_unlearn_dp_"+sample_size, "og_unlearn_ufl_"+sample_size])
runs_df = pd.merge(runs_df, mem_df, on='name') 

runs_df.rename(columns={'target/acc': 'memory'}, inplace=True)
runs_df['memory'] = runs_df['memory'] *100
# runs_df['mem_difference'] = runs_df.apply(lambda row: abs(row['value'] - v) if k in row['name'].lower() else None, axis=1)

# Use the apply function to create a new column based on the custom function
runs_df['algo'] = runs_df['name'].apply(lambda x: create_algo(x))

# Filter columns and create a new DataFrame
runs_df = runs_df.drop(columns=[col for col in runs_df.columns if 'lambada' in col])

runs_df.to_csv(f'experiment2/{model_name}_{sample_size}_full.csv')

# Identify numerical columns for which you want to calculate the mean
numerical_columns = runs_df.select_dtypes(include=['number']).columns

# Group by 'algo_type' and calculate the mean for numerical columns
average_accuracy_df = runs_df.groupby('algo')[numerical_columns].mean().reset_index()

# Filter columns that end with "/acc"
acc_columns = average_accuracy_df.filter(regex='/acc$', axis=1)

# Calculate the row-wise average for the selected columns
average_accuracy_df['average_acc'] = acc_columns.mean(axis=1)

# Filter columns that end with "/loss"
loss_columns = average_accuracy_df.filter(regex='/loss$', axis=1)

# Calculate the row-wise average for the selected columns
average_accuracy_df['average_perplexity'] = loss_columns.mean(axis=1)

# Filter columns that end with "/loss"
f1_columns = average_accuracy_df.filter(regex='/f1$', axis=1)

# Calculate the row-wise average for the selected columns
average_accuracy_df['average_f1'] = f1_columns.mean(axis=1)

# Apply the lambda function to the 'score' column
average_accuracy_df['memory_dist'] = abs(average_accuracy_df['memory'] - tr_dict.get(model_name))

average_accuracy_df.to_csv(f'experiment2/{model_name}_{sample_size}_avg.csv')

latex_df = average_accuracy_df[['algo', 'average_f1', 'average_perplexity', 'average_acc', 'total_extractable', '_runtime', 'memory_dist']]
latex_df.to_csv(f'experiment2/{model_name}_{sample_size}_filter.csv')