import json
import matplotlib.pyplot as plt

save_dir = 'llmu_results'
json_file = 'val_data_init.json'

parent_1 = 'MA_verification_unlearn_ufl_mia_lm_extraction_128_0.csv_gpt-neo-1.3b_seed_42_coefs_-3_3_epc_00_bs_32_lr_5e-05'
parent_2 = 'MA_verification_unlearn_ufl_mia_lm_extraction_128_0.csv_gpt-neo-125M_seed_42_coefs_-3_3_epc_00_bs_32_lr_5e-05'

fp1 = f'{save_dir}/{parent_1}/{json_file}'
fp2 = f'{save_dir}/{parent_2}/{json_file}'



with open(fp1, 'r') as json_file:
    data_1 = json.load(json_file)


with open(fp2, 'r') as json_file:
    data_2 = json.load(json_file)


plt.clf()

plt.figure(figsize=(10, 6))
plt.scatter(data_1[0]['acc'], data_1[0]['ll_dist'], alpha=0.5, color='blue', label='1.3b')
plt.scatter(data_2[0]['acc'], data_2[0]['ll_dist'], alpha=0.5, color='red', label='125M')


plt.legend()

# Add grid for better visibility of quadrants
plt.grid(True)

plt.xlabel('Verbatim memory')
plt.ylabel('Counterfactual memory')

plt.title(f"Verbatim vs Counterfactual (Pile)")
plt.savefig(f"scripts/memory_vs_counterfactual_vs_scale.png")