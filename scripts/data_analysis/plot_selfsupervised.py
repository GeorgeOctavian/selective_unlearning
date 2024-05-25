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


from scipy.stats import t



def calculate_confidence_intervals(grouped_df, confidence_level=0.95):
    # Number of observations (assuming sample size is 5)
    n = 5

    # Degrees of freedom
    df = n - 1

    # t-value for the given confidence level and degrees of freedom
    t_value = t.ppf((1 + confidence_level) / 2, df)

    # Compute confidence intervals for each column
    for column in grouped_df.columns.levels[0]:
        if column != 'algo':
            mean_col = grouped_df[column]['mean']
            std_col = grouped_df[column]['std']

            # Standard error of the mean
            sem = std_col / np.sqrt(n)

            # Margin of error
            margin_of_error = t_value * sem

            # Confidence interval
            lower_bound = mean_col - margin_of_error
            upper_bound = mean_col + margin_of_error

            # Update the DataFrame with confidence interval columns
            grouped_df[column, 'lower_bound'] = lower_bound
            grouped_df[column, 'upper_bound'] = upper_bound

    return grouped_df



# from plot_spider import plot_spider
api = wandb.Api()

def plot_spider(df, name):

    plt.clf()
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 4 groups makes the chart unreadable

    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']  # Extend for more line styles

    # Use a color palette suitable for color-blind viewers
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']

    for i in range(len(df)):
        values = df.loc[i].drop('algo').values.flatten().tolist()
        values += values[:1]

        algo_label = df.loc[i]['algo']

        # Use a different linestyle for each algorithm
        linestyle = line_styles[i % len(line_styles)]

        # Use a different color for each algorithm
        color = colors[i % len(colors)]

        # Increase line width
        linewidth = 2.0

        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle, color=color, label=f"{algo_label}")
        ax.fill(angles, values, alpha=0)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Show the graph
    plt.show()

    plt.savefig(f'experiment1/{name}.pdf')

tr_dict = {'gpt-neo-2.7b': 34.02, 'gpt-neo-1.3b':33.27, 'gpt-neo-125M': 29.94}

sample_size =  'lm_extraction_128_'
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
    
    if 'sga+ta' in name: 
        return 'TAU'
    elif 'sga' in name:
        return 'SGA'
    elif 'ufl' in name:
        return 'GA'
    elif 'air' in name:
        return 'TA'
    # elif 'dp' in name:
    #     return 'DPD'


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
                acc_idx_array = data.get(f"acc_{formatted_index}", [])

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

mem_df = process_directories("llmu_results", model_name, ["sga+ta_unlearn_air_"+sample_size, "_unlearn_ufl_"+sample_size, "sga_unlearn_ufl_"+sample_size, "_unlearn_air_"+sample_size])
runs_df = pd.merge(runs_df, mem_df, on='name') 

runs_df.rename(columns={'target/acc': 'memory'}, inplace=True)
runs_df['memory'] = runs_df['memory'] *100
# runs_df['mem_difference'] = runs_df.apply(lambda row: abs(row['value'] - v) if k in row['name'].lower() else None, axis=1)

# Use the apply function to create a new column based on the custom function
runs_df['algo'] = runs_df['name'].apply(lambda x: create_algo(x))

# Filter columns and create a new DataFrame
runs_df = runs_df.drop(columns=[col for col in runs_df.columns if 'lambada' in col])

runs_df.to_csv(f'experiment1/{model_name}_{sample_size}_full.csv')

# Identify numerical columns for which you want to calculate the mean
numerical_columns = runs_df.select_dtypes(include=['number']).columns

std_accuracy_df = runs_df.groupby('algo')[numerical_columns].std().reset_index()

std_acc_columns = std_accuracy_df.filter(regex='/acc$', axis=1)
std_loss_columns = std_accuracy_df.filter(regex='/loss$', axis=1)
std_f1_columns = std_accuracy_df.filter(regex='/f1$', axis=1)

std_accuracy_df['avg_std_acc'] = std_acc_columns.mean(axis=1)
std_accuracy_df['avg_std_ppl'] = std_loss_columns.mean(axis=1)
std_accuracy_df['avg_std_f1'] = std_f1_columns.mean(axis=1)

std_accuracy_df.to_csv(f'experiment1/{model_name}_{sample_size}_std.csv')

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

average_accuracy_df.to_csv(f'experiment1/{model_name}_{sample_size}_avg.csv')

spider_df = average_accuracy_df[['algo', 'average_f1', 'average_perplexity', 'average_acc']]

spider_df.loc[:, 'average_f1'] = spider_df['average_f1'] * 100
spider_df.loc[:, 'average_perplexity'] = spider_df['average_perplexity']
spider_df.loc[:, 'average_acc'] = spider_df['average_acc'] * 10


latex_df = average_accuracy_df[['algo', 'average_f1', 'average_perplexity', 'average_acc', 'total_extractable', '_runtime', 'memory_dist']]
latex_df.to_csv(f'experiment1/{model_name}_{sample_size}_filter.csv')

plot_spider(spider_df, f'{model_name}_{sample_size}')

f1_column_names =  ['algo'] + list(f1_columns) 

f1_and_avg = f1_column_names + ['average_f1']
f1_df = average_accuracy_df[f1_and_avg]

f1_df.to_csv(f'experiment1/{model_name}_{sample_size}_individ_f1.csv')

plot_spider(f1_df, f'{model_name}_{sample_size}_f1')


acc_column_names =  ['algo'] + list(acc_columns) 

acc_df = average_accuracy_df[acc_column_names]

plot_spider(acc_df, f'{model_name}_{sample_size}_acc')


loss_column_names =  ['algo'] + list(loss_columns) 

loss_df = average_accuracy_df[loss_column_names]

plot_spider(loss_df, f'{model_name}_{sample_size}_loss')


mem_columns = ['below_015_count','above_060_count', 'total_extractable', 'Memory_std']
numerical_df = runs_df[['algo'] + list(f1_columns) + list(acc_columns) + mem_columns]


grouped_stats = numerical_df.groupby('algo').agg(['mean', 'std'])

# Reset index to make 'algo' a regular column
grouped_stats.reset_index(inplace=True)


grouped_stats = calculate_confidence_intervals(grouped_stats)

grouped_stats.to_csv(f'experiment1/{model_name}_{sample_size}_confidence.csv')
