import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def plot_scatter_acc_vs_conterfactual(memory, counterfactual, save_dir, name):

    plt.clf()

    plt.figure(figsize=(10, 6))

    correlation_coefficient, p_value = scipy.stats.pearsonr(memory, counterfactual)


    # Compute median values
    median_x = np.median(memory)
    median_y = np.median(counterfactual)

    # Classify points into quadrants
    quadrant_labels = ['Quadrant I', 'Quadrant II', 'Quadrant III', 'Quadrant IV']
    quadrant_colors = ['r', 'g', 'b', 'y']

    quadrant = []

    for xi, yi in zip(memory, counterfactual):
        if xi > median_x and yi > median_y:
            quadrant.append(0)
        elif xi <= median_x and yi > median_y:
            quadrant.append(1)
        elif xi <= median_x and yi <= median_y:
            quadrant.append(2)
        elif xi > median_x and yi <= median_y:
            quadrant.append(3)

    # Plot points with different colors for each quadrant
    for q in range(4):
        plt.scatter([memory[i] for i in range(len(memory)) if quadrant[i] == q],
                    [counterfactual[i] for i in range(len(counterfactual)) if quadrant[i] == q],
                    label=quadrant_labels[q],
                    color=quadrant_colors[q])

    # Add legend
    plt.legend()

    # Add grid for better visibility of quadrants
    plt.grid(True)

    plt.xlabel('Verbatim memory')
    plt.ylabel('Counterfactual memory')

    plt.title(f"Verbatim vs Counterfactual (Pile). Correlation coefficient: {correlation_coefficient}, P-value {p_value}")
    plt.savefig(f"{save_dir}/memory_vs_counterfactual_{name}.png")
    

def plot_counterfactual_histogram(data, save_dir, histo_name):
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.hist(data, color='blue', edgecolor='black')

    plt.xlabel('Distance to neighbours (counterfactual)')
    plt.ylabel('Frequency')
    plt.title('Histogram for approximate counterfactual memory (Pile)')
    plt.savefig(f"{save_dir}/counterfactual_hist_{histo_name}.png")

def plot_accuracy_histogram(data, save_dir, histo_name):
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.hist(data, color='blue', edgecolor='black')

    plt.xlabel('Verbatim memory')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for verbatim memorisation (Pile). Memory {np.mean(data)}')
    plt.savefig(f"{save_dir}/memo_hist_{histo_name}.png")
    
def save_logl_histograms(member_out, nonmember_out, save_dir, histo_name):
    # first, clear plt
    plt.clf()

    # for experiment in experiments:
    try:
        # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
        bins = 20  # You can adjust this value according to your needs

        # Set the bin edges explicitly to ensure they are the same for both histograms
        bin_edges = np.linspace(min(min(member_out['og_ll']), min(member_out['pert_mean_ll'])),
                                max(max(member_out['og_ll']), max(member_out['pert_mean_ll'])), bins+1)


        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        plt.hist([r for r in member_out['og_ll']], alpha=0.5, bins=bin_edges, label='member')
        plt.hist([r for r in member_out['pert_mean_ll']], alpha=0.5, bins=bin_edges, label='perturbed member')
        plt.xlabel("log likelihood")
        plt.ylabel('count')
        plt.legend(loc='upper right')

        bin_edges = np.linspace(min(min(nonmember_out['og_ll']), min(nonmember_out['pert_mean_ll'])),
                                max(max(nonmember_out['og_ll']), max(nonmember_out['pert_mean_ll'])), bins+1)
        
        plt.subplot(1, 2, 2)
        plt.hist([r for r in nonmember_out['og_ll']], alpha=0.5, bins='auto', label='nonmember')
        plt.hist([r for r in nonmember_out['pert_mean_ll']], alpha=0.5, bins='auto', label='perturbed nonmember')
        plt.xlabel("log likelihood")
        plt.ylabel('count')
        plt.legend(loc='upper right')
        plt.savefig(f"{save_dir}/ll_histograms_{histo_name}.png")
    except Exception as e:
        print(f"Error: {e}")
        pass


def save_llr_histograms(member_out, nonmember_out, save_dir, histo_name):
    # Clear any existing plots
    plt.clf()

    try:
        # Plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
        plt.figure(figsize=(10, 6))

        # Choose the number of bins based on your data distribution
        bins = 20  # You can adjust this value according to your needs

        # Set the bin edges explicitly to ensure they are the same for both histograms
        bin_edges = np.linspace(min(min(member_out['ll_dist']), min(nonmember_out['ll_dist'])),
                                max(max(member_out['ll_dist']), max(nonmember_out['ll_dist'])), bins+1)

        # Plot histograms with specified bin edges
        plt.hist(member_out['ll_dist'], alpha=0.5, bins=bin_edges, label='member')
        plt.hist(nonmember_out['ll_dist'], alpha=0.5, bins=bin_edges, label='nonmember')

        # Set labels and legend
        plt.xlabel("log likelihood ratio")
        plt.ylabel('count')
        plt.legend(loc='upper right')

        # Save the figure
        plt.savefig(f"{save_dir}/llr_histograms_{histo_name}.png")

        # Optionally, you may want to show the plot
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")

def save_roc_curves(fpr, tpr, roc_auc, save_dir, curve_name):
    # first, clear plt
    plt.clf()

    # for experiment, color in zip(experiments, COLORS):
    # metrics = experiment["metrics"]
    plt.plot(fpr, tpr, label=f"roc_auc={roc_auc:.3f}", color=COLORS[0])
    # print roc_auc for this experiment
    print(f"roc_auc: {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves_{curve_name}')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_dir}/roc_curves_{curve_name}.png")



