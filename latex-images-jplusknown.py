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

nownk_pn1 = ["64343-69243", "70242-18193", "65251-18195", "65750-23821"]
nownk_pn = ["67905-25070", "62617-9746", "63864-47481", "70145-19210", "64078-73697"]
file_list1 = []
file_list2 = []

file_list11 = []
file_list22 = []

spect, im1, im2 = [], [], []
spect_, im1_, im2_ = [], [], []

x = [i.split("-")[0] for i in nownk_pn]
for y in x:
    pattern1 = "Field_"+y+"/1000001-JPLUS*J0861*-RGB.pdf"
    file_list1.append(glob.glob(pattern1))

    pattern2 = "Field_"+str(y)+"/1000001-JPLUS*iSDSS*-RGB.pdf"
    file_list2.append(glob.glob(pattern2))

x1 = [i.split("-")[0] for i in nownk_pn1]
for y in x1:
    pattern11 = "Field_"+str(y)+"/1000001-JPLUS*J0861*-RGB.pdf"           
    file_list11.append(glob.glob(pattern11))

    pattern22 = "Field_"+str(y)+"/1000001-JPLUS*iSDSS*-RGB.pdf"               
    file_list22.append(glob.glob(pattern22))
    
for a, b, c in zip(nownk_pn, file_list1, file_list2):
    spect.append("\includegraphics[width=0.3\linewidth, clip]{figs-pca/photospectrum_"+a+"-PN-pc-Halpha_emitters_threeerror-cleaning-limfilter-limcolor-flags-mask-broad_MAG_APER_6_0.pdf}")
    im1.append("\includegraphics[width=0.3\linewidth, clip]{"+str(b).split("['")[-1].split("']")[0]+"}")
    im2.append("\includegraphics[width=0.3\linewidth, clip]{"+str(c).split("['")[-1].split("']")[0]+"}")

for aa, bb, cc in zip(nownk_pn1, file_list11, file_list22):
    spect_.append("\includegraphics[width=0.3\linewidth, clip]{figs-pca/photospectrum_"+aa+"-Missing-pne-allinf-v2_MAG_APER_6_0.pdf}")
    im1_.append("\includegraphics[width=0.3\linewidth, clip]{"+str(bb).split("['")[-1].split("']")[0]+"}")
    im2_.append("\includegraphics[width=0.3\linewidth, clip]{"+str(cc).split("['")[-1].split("']")[0]+"}")

spectTotal = spect + spect_
im1Total = im1 + im1_
im2Total = im2 + im2_
spectTotal.sort()
im1Total.sort()
im2Total.sort()
table_fig = Table([spectTotal, im1Total, im2Total],  names=("\\textbf{Aper (6 \")}", "\\textbf{J0515+J0660+J0861}", "\\textbf{gSDSS+rSDSS+iSDSS}"), meta={'name': 'first table'})
#table_fig.sort('Auto')
table_fig.write('table-images-pnknown.tex', format = "ascii.latex", overwrite=True) 
