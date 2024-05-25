import json
import matplotlib.pyplot as plt
import numpy as np

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

# Extract 'acc' and 'll_dist' values
acc_1 = data_1[0]['acc']
ll_dist_1 = data_1[0]['ll_dist']

acc_2 = data_2[0]['acc']
ll_dist_2 = data_2[0]['ll_dist']

# Calculate medians
median_acc = np.median(np.concatenate([acc_1, acc_2]))
median_ll_dist = np.median(np.concatenate([ll_dist_1, ll_dist_2]))

# Determine quadrants
quadrant_1 = (acc_1 >= median_acc) & (ll_dist_1 >= median_ll_dist)
quadrant_2 = (acc_1 < median_acc) & (ll_dist_1 >= median_ll_dist)
quadrant_3 = (acc_1 < median_acc) & (ll_dist_1 < median_ll_dist)
quadrant_4 = (acc_1 >= median_acc) & (ll_dist_1 < median_ll_dist)

# Plot the scatter plot
plt.clf()
plt.figure(figsize=(10, 6))

# Plot points for dataset 1
plt.scatter(np.array(acc_1)[quadrant_1], np.array(ll_dist_1)[quadrant_1], alpha=0.5, color='blue', label='1.3b')
plt.scatter(np.array(acc_1)[quadrant_2], np.array(ll_dist_1)[quadrant_2], alpha=0.5, color='blue')
plt.scatter(np.array(acc_1)[quadrant_3], np.array(ll_dist_1)[quadrant_3], alpha=0.5, color='blue')
plt.scatter(np.array(acc_1)[quadrant_4], np.array(ll_dist_1)[quadrant_4], alpha=0.5, color='blue')

# Plot points for dataset 2
plt.scatter(np.array(acc_2)[quadrant_1], np.array(ll_dist_2)[quadrant_1], alpha=0.5, color='red', label='125M')
plt.scatter(np.array(acc_2)[quadrant_2], np.array(ll_dist_2)[quadrant_2], alpha=0.5, color='red')
plt.scatter(np.array(acc_2)[quadrant_3], np.array(ll_dist_2)[quadrant_3], alpha=0.5, color='red')
plt.scatter(np.array(acc_2)[quadrant_4], np.array(ll_dist_2)[quadrant_4], alpha=0.5, color='red')

# Color the background based on quadrants
# Color the background based on quadrants
plt.axhline(y=median_ll_dist, color='gray', linestyle='--')
plt.axvline(x=median_acc, color='gray', linestyle='--')

plt.fill_betweenx(y=[0, median_ll_dist], x1=0, x2=median_acc, color='lightgray', alpha=0.5)
plt.fill_betweenx(y=[median_ll_dist, 2], x1=0, x2=median_acc, color='darkgray', alpha=0.5)

plt.fill_betweenx(y=[0, median_ll_dist], x1=median_acc, x2=1, color='lightgray', alpha=0.5)
plt.fill_betweenx(y=[median_ll_dist, 2], x1=median_acc, x2=1, color='darkgray', alpha=0.5)


plt.legend()

# Add grid for better visibility of quadrants
plt.grid(True)

plt.xlabel('Verbatim memory')
plt.ylabel('Counterfactual memory')

plt.title(f"Verbatim vs Counterfactual (Pile)")
plt.savefig(f"scripts/ds_vs_scale.png")
plt.show()

