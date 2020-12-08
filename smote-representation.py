'''
Read the file from J-PLUS IDR table to make the colour-colour diagrams
'''
from __future__ import print_function
import pylab
import random
import seaborn as sns
import numpy as np
from astropy.io import fits
import os
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
#import StringIO
from astropy.table import Table
import seaborn as sns
import sys
from scipy.optimize import fsolve
import colours

sampleSize = 16

x = []
y = []

for i in range(sampleSize):
    newVal = random.normalvariate(10,1)
    x.append(newVal)
    y.append(newVal / 2.0 + random.normalvariate(1,1))



x1 = np.array([13,  14])
y1 = np.array([3, 1])
x2 = np.array([13, 13])
y2 = np.array([3, -0.5])
x3 =  np.array([14, 13])
y3 =  np.array([1, -0.5]) 
x4 =  np.array([13, 14])
y4 =  np.array([1, 1])


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

fig = plt.figure() #para cv
ax = fig.add_subplot(1,1,1)
plt.scatter(x,y, c= sns.xkcd_rgb["light green"], marker="o", s=120,  edgecolor='black', label="Class 0")
plt.scatter(x1,y1, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black', label="Class 1")

plt.scatter(x2,y2, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')


plt.scatter(x2,y1, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')

plt.scatter(x3,y3, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')

plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
# plt.xlabel("X" , fontsize= 30)
# plt.ylabel("Y", fontsize= 30)
plt.legend(scatterpoints=1, ncol=1, fontsize=10.0, loc='upper right')
plt.tight_layout()
plt.tight_layout()
pltfile = 'smote-example1.pdf'
save_path = '../../Dropbox/JPAS/Tesis/Fig/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

fig = plt.figure() #para cv
ax = fig.add_subplot(1,1,1)
plt.scatter(x,y, c= sns.xkcd_rgb["light green"], marker="o", s=120,  edgecolor='black', label="Class 0")
plt.scatter(x1,y1, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black', label="Class 1 - original")
plt.scatter(13+0.2,  3-0.4, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black', label="Class 1 - SMOTE")
plt.scatter(13+0.45,  3-0.9, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13+0.67,  3-1.3, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')

plt.scatter(x2,y2, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')
plt.scatter(13,  3-0.4, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
#plt.scatter(13,  3-0.9, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13,  3-1.3, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')

plt.scatter(x2,y1, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')
#plt.scatter(13,  3-1.5, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13, 0.5, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13, 0.1, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')

plt.scatter(x3,y3, c= sns.xkcd_rgb["light red"], marker="o", s=120,  edgecolor='black')
plt.scatter(13.2,  -0.2, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13.4,  0.15, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13.8,  0.7, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')

#plt.scatter(13.2,  1, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13.4,  1, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')
plt.scatter(13.7,  1, c= sns.xkcd_rgb["pale yellow"], marker="o", s=60,  edgecolor='black')


plt.plot(x1,y1, "b-")
plt.plot(x2,y2, "b-")
plt.plot(x4,y4, "b-")
plt.plot(x3,y3, "b-")
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
# plt.xlabel("X" , fontsize= 30)
# plt.ylabel("Y", fontsize= 30)
plt.legend(scatterpoints=1, ncol=1, fontsize=10.0, loc='upper right')
plt.tight_layout()
plt.tight_layout()
pltfile = 'smote-example2.pdf'
save_path = '../../Dropbox/JPAS/Tesis/Fig/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
