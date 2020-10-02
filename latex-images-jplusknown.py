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

nownk_pn1 = ["64343-69243", "70242-18193", "65251-18195", "65750-23436"]
nownk_pn = ["64343-69243", "70242-18193", "65251-18195", "65750-23436", "67905-25070", "62617-9746", "63864-47481", "70145-19210", "64078-73697"]
spect, im1, im2 = [], [], []


for y in x:
    pattern1 = str(y)+"/1000001-JPLUS*J0861*-RGB.pdf"           
    file_list1 = glob.glob(pattern1)

    pattern2 = str(y)+"/1000001-JPLUS*iSDSS*-RGB.pdf"               
    file_list2 = glob.glob(pattern2)
    
    for i, b, c in zip(nownk_pn, file_list1, file_list2):
        spect.append("\includegraphics[width=0.3\linewidth, clip]{figs-pca/photospectrum_"+i+"-PN-pc-Halpha_emitters_threeerror-cleaning-limfilter-limcolor-flags-mask-broad_MAG_APER_6_0.pdf}")
        spect.append("\includegraphics[width=0.3\linewidth, clip]{figs-pca/photospectrum_"+i+"-Missing-pne-allinf-v2_MAG_APER_6_0.pdf}")

        print(i)
        im1.append("\includegraphics[width=0.3\linewidth, clip]{"+b+"}")
        im2.append("\includegraphics[width=0.3\linewidth, clip]{"+c+"}")

    spect.sort()
    im1.sort()
    im2.sort()
    table_fig = Table([spect, im1, im2],  names=('Aper (6 \")', 'J0515+J0660+J0861', 'gSDSS+rSDSS+iSDSS'), meta={'name': 'first table'})
    #table_fig.sort('Auto')
    table_fig.write('table-images-pnknown.tex', format = "ascii.latex", overwrite=True) 
