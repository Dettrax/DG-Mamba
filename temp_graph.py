import numpy as np
from matplotlib import pyplot as plt

# Updated data for each model and metric
models = ['lb = 1', 'lb = 2', 'lb = 3', 'lb = 4', 'lb = 5']
x_pos = np.arange(len(models))
width = 0.3

# TransformerG2G
MAP_transg2g = [0.6204, 0.6143, 0.5927, 0.6096, 0.6097]
std_transg2g = [0.0386, 0.0274, 0.0192, 0.0360, 0.0104]

# MambaG2G
MAP_Mam = [0.6896, 0.6934, 0.6848, 0.6920, 0.6873]
std_Mam = [0.0013, 0.0012, 0.0005, 0.0001, 0.0019]

# MambaG2G V2
MAP_MamV2 = [0.6928, 0.6828, 0.6863, 0.6872, 0.6857]
std_MamV2 = [0.0042, 0.0045, 0.0013, 0.0035, 0.0061]

# Plotting
fig, ax = plt.subplots()
ax.bar(x_pos - width, MAP_transg2g, yerr=std_transg2g, align='center', alpha=0.5, ecolor='black', capsize=10, width=width, label='TransformerG2G')
ax.bar(x_pos, MAP_Mam, yerr=std_Mam, align='center', alpha=0.5, ecolor='black', capsize=10, width=width, label='MambaG2G')
ax.bar(x_pos + width, MAP_MamV2, yerr=std_MamV2, align='center', alpha=0.5, ecolor='black', capsize=10, width=width, label='MambaG2G V2')

# Labels and Title
ax.set_ylabel('MAP', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)
ax.set_title('SBM', fontsize=16)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.ylim([0, 0.9])
plt.legend()
plt.savefig('updated_bar_plot_with_error_bars.png')
plt.show()
