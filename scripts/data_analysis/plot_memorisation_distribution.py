import json
import matplotlib.pyplot as plt

file_path = "llmu_results/_unlearn_ufl_lm_extraction_32_0.csv_gpt-neo-1.3b_seed_42_coefs_-3.0_0_epc_00_bs_32_lr_5e-05/val_data_11.json"

with open(file_path, 'r') as file:
    data = json.load(file)

acc_init_value = data.get('acc_11', None)

if acc_init_value is not None:
    # Plot the distribution
    plt.hist(acc_init_value, edgecolor='black', alpha=0.7)

    # Highlight values larger than 0.6 in red
    plt.axvline(x=0.6, color='red', linestyle='--', label='Threshold: 0.6')

    # Highlight values lower than 0.15 in blue
    plt.axvline(x=0.15, color='blue', linestyle='--', label='Threshold: 0.15')

    plt.title('Distribution of Memorisation Scores', fontsize=16)
    plt.xlabel('Memorisation Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    # Increase font size of tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)
    plt.savefig('memory_dist.pdf')
else:
    print("acc_init not found in the JSON file.")