'''
Principal component analysis (J-PLUS)
'''
#from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns
import os.path
from collections import OrderedDict
from scipy.stats import gaussian_kde
import pandas as pd

g=[]
n=[]


pattern =  "BB*-spectros/*-JPLUS17-magnitude.json"
file_list = glob.glob(pattern)

for i in range(len(file_list)):
    n.append(i)
for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        g.append(data["F480_g_sdss"])
        #print(len(str(data["F480_g_sdss"])))
        
#Save files with flux
asciifile = "g-band-PNModels.dat"
file=open(asciifile,'w') #create file  
for x,y in zip(n, g):
    file.write('%s %s\n'%(x,y))     #assume you separate columns by tabs  
file.close()     #close fil
