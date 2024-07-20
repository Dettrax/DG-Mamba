import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['MambaG2G', 'TransformerG2G']
metrics = ['MAP', 'MRR']
l_values = [1, 2, 3, 4, 5]

# MambaG2G values
mambaG2G_MAP = [0.1108, 0.1228, 0.09776, 0.0979, 0.1206]
mambaG2G_MAP_err = [0.0061, 0.0087, 0.0012, 0.0015, 0.0126]
mambaG2G_MRR = [0.4228, 0.4615, 0.4800, 0.4761, 0.4324]
mambaG2G_MRR_err = [0.0034, 0.0007, 0.0056, 0.0001, 0.0008]

# TransformerG2G values
transformerG2G_MAP = [0.0241, 0.2057, 0.0340, 0.0342, 0.0495]
transformerG2G_MAP_err = [0.0152, 0.0356, 0.0256, 0.0244, 0.0107]
transformerG2G_MRR = [0.2616, 0.2947, 0.2905, 0.3042, 0.3447]
transformerG2G_MRR_err = [0.0879, 0.1087, 0.1121, 0.1115, 0.0196]

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot MAP
axs[0].errorbar(l_values, mambaG2G_MAP, yerr=mambaG2G_MAP_err, fmt='-o', label='MambaG2G')
axs[0].errorbar(l_values, transformerG2G_MAP, yerr=transformerG2G_MAP_err, fmt='-o', label='TransformerG2G')
axs[0].set_title('MAP vs. l')
axs[0].set_xlabel('l')
axs[0].set_ylabel('MAP')
axs[0].legend()
axs[0].grid(True)

# Plot MRR
axs[1].errorbar(l_values, mambaG2G_MRR, yerr=mambaG2G_MRR_err, fmt='-o', label='MambaG2G')
axs[1].errorbar(l_values, transformerG2G_MRR, yerr=transformerG2G_MRR_err, fmt='-o', label='TransformerG2G')
axs[1].set_title('MRR vs. l')
axs[1].set_xlabel('l')
axs[1].set_ylabel('MRR')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
