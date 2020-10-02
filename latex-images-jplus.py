'''
Create file.tex with several figures (table)
'''
from __future__ import print_function
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


#Read de files
pattern = "figs-pca/*mask-broad_MAG_APER_6_0.pdf"
file_list = glob.glob(pattern)

pattern1 = "Field*/1000001-JPLUS*J0861*-RGB.pdf"           #1000001-JPLUS-00588-v202006_iSDSS_62617-9746-RGB
file_list1 = glob.glob(pattern1)

pattern2 = "Field*/1000001-JPLUS*iSDSS*-RGB.pdf"               
file_list2 = glob.glob(pattern2)

latex_columns = ['Aper (6 \")', 'J0515+J0660+J0861', 'gSDSS+rSDSS+iSDSS']
    
#print('\n'.join(map(lambda x: 'Test{0:04}'.format(x), range(0, 10000))))

ap3, apa, app = [], [], []
for a, b, c in zip(file_list, file_list1, file_list2):
    ap3.append("\includegraphics[width=0.3\linewidth, clip]{"+a+"}")
    apa.append("\includegraphics[width=0.3\linewidth, clip]{"+b+"}")
    app.append("\includegraphics[width=0.3\linewidth, clip]{"+c+"}")

ap3.sort()
apa.sort()
app.sort()
table_fig = Table([ap3, apa, app],  names=('Aper (6 \")', 'J0515+J0660+J0861', 'gSDSS+rSDSS+iSDSS'), meta={'name': 'first table'})
    #table_fig.sort('Auto')
table_fig.write('table-images.tex', format = "ascii.latex", overwrite=True) 
