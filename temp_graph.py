import numpy as np
from matplotlib import pyplot as plt

models = ['lb = 1', 'lb = 2', 'lb = 3', 'lb = 4', 'lb = 5']
x_pos = np.arange(len(models))
MAP_transg2g = [0.0241, 0.0347, 0.0340, 0.0342, 0.0495]
std_transg2g = [0.0152, 0.0222, 0.0256, 0.0244, 0.0107]

MAP_Mam = [0.0766, 0.1244, 0.131, 0.1425, 0.0938]
std_Mam = [0.0034, 0.0013, 0.0015, 0.0019, 0.002]


fig, ax = plt.subplots()
width = 0.4
# ax.bar(x_pos[:1], MAP[:1], yerr=std[:1], align='center', alpha=0.5, ecolor='black', capsize=10, width = width)
ax.bar(x_pos[0:]-width/2, MAP_transg2g[0:], yerr=std_transg2g[0:], align='center', alpha=0.5, ecolor='black', capsize=10, width = width)
ax.bar(x_pos[0:]+width/2, MAP_Mam, yerr=std_Mam, align='center', alpha=0.5, ecolor='black', capsize=10, width = width)
ax.set_ylabel('MAP',fontsize = 12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize = 12)
ax.set_title('UCI', fontsize = 16)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.ylim([0, 0.15])
plt.savefig('bar_plot_with_error_bars.png')
plt.legend(['TransformerG2G', 'MambaG2G'])
plt.show()

