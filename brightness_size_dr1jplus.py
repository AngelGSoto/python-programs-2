import glob
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
#import StringIO
from astropy.table import Table
from astropy.io import ascii
import sys
import numpy as np
import seaborn as sns

tab = Table.read("17519.tab", format="ascii.tab")

r_mag = tab['rSDSS_ISO_GAUSS']
area = tab['ISOAREA_WORLD']
size = tab['R_EFF']

#brightness
b = r_mag + 2.5*np.log10(area)

plotfile = "brightnes-size.pdf"
fig = plt.figure(figsize=(15, 9.5))
ax = fig.add_subplot(1,1,1)
plt.tick_params(axis='x', labelsize=42) 
plt.tick_params(axis='y', labelsize=42)
#ax.set_xlim(xmin=3000, xmax=9700)
#ax.set_ylim(ymin=17.5,ymax=23)
#ax1.set_xlabel(r'$\lambda$')
ax.set_xlabel(r'R_eff (pixel)', fontsize = 44)
ax.set_ylabel(r'Surface brightness', fontsize = 44)
ax.scatter(size, b, color = sns.xkcd_rgb["cerulean"], edgecolor='black', marker='o', s=40, cmap='viridis')
#ax.errorbar(wl1, mag, yerr=mag_err, marker='.', fmt='.', color=colors, ecolor=colors, elinewidth=5.9, markeredgewidth=5.2,  capsize=20)
#plt.subplots_adjust(bottom=0.19)
#plt.legend(fontsize=20.0)
plt.tight_layout()
plt.savefig(plotfile)
plt.clf()

