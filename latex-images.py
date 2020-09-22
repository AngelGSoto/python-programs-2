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
pattern = "*match-2007_aper.pdf"
file_list = glob.glob(pattern)

pattern1 = "MC*/MC*F861*-RGB.pdf"              #MC0133_F861_025038-RGB.pdf
file_list1 = glob.glob(pattern1)

pattern2 = "MC*/MC*I*-RGB.pdf"                 #MC0133_I_025038-RGB.pdf 
file_list2 = glob.glob(pattern2)

latex_columns = ['Aper_3', 'F515+F660+F861', 'G+R+I']
    
#print('\n'.join(map(lambda x: 'Test{0:04}'.format(x), range(0, 10000))))

ap3, apa, app = [], [], []
for a, b, c in zip(file_list, file_list1, file_list2):
    ap3.append("\includegraphics[width=0.3\linewidth, clip]{"+a+"}")
    apa.append("\includegraphics[width=0.3\linewidth, clip]{"+b+"}")
    app.append("\includegraphics[width=0.3\linewidth, clip]{"+c+"}")

ap3.sort()
apa.sort()
app.sort()
table_fig = Table([ap3, apa, app],  names=('Aper_3', 'F515+F660+F861', 'G+R+I'), meta={'name': 'first table'})
    #table_fig.sort('Auto')
table_fig.write('table-images.tex', format = "ascii.latex", overwrite=True) 
