'''
Getting the PNe from S-PLUS catalog (DR2)
'''
from __future__ import print_function
from astropy.io import fits
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from astropy.table import Table
import argparse
from io import StringIO
import os

n=129
mag_aper = [[] for _ in range(n)]
mag_petro = [[] for _ in range(n)]
mag_auto = [[] for _ in range(n)]

def select(magg):
    ef1 = dta['euJAVA_'+magg] <= 3.0
    ef2 = dta['eF378_'+magg] <= 3.0
    ef3 =  dta['eF395_'+magg] <= 3.0
    ef4 =  dta['eF410_'+magg] <= 3.0
    ef5 =  dta['eF430_'+magg] <= 3.0
    ef6 =  dta['eg_'+magg] <= 3.0
    ef7 =  dta['eF515_'+magg] <= 3.0
    ef8 =  dta['er_'+magg] <= 0.2
    ef9 =  dta['eF660_'+magg] <= 0.2
    ef10 =  dta['ei_'+magg] <= 0.2
    ef11 =  dta['eF861_'+magg] <= 3.0
    ef12 =  dta['ez_'+magg] <= 3.0

    mask = ef1 & ef2 & ef3 & ef5 & ef6 & ef7 & ef8 & ef9 & ef10 & ef11 & ef12
    #mask = ef8 & ef9 & ef10 
    
    #Mask
    #Mask filters
    #q = (dta['F515_'+magg] - dta['F660_'+magg]) <= 5.0
    #q1 = (data['J0515'] - data['J0861']) >= -3.0
    #q2 = (dta['r_'+magg] - dta['F660_'+magg]) <= 4.0
    #q3 = (dta['r_'+magg] - dta['i_'+magg]) >= -4.0
    #q4 = (dta['g_'+magg] - dta['F515_'+magg]) >= -3.2
    #q5 = (dta['g_'+magg] - dta['i_'+magg]) >= -3.0
    #q6 = (dta['F410_'+magg] - dta['F660_'+magg]) <= 6.0
    #q7 = (dta['z_'+magg] - dta['g_'+magg]) <= 4.0


    m = dta['PhotoFlag'] == 0.0
    #m = data['Class star'] > 0.0 
    m1 = dta['F515_'+magg] - dta['F660_'+magg]>=0.2
    
    total_m = mask & m1 
    
    table = Table(dta[total_m])
    print(len(table["RA"]))
    return table

   
parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
fitsfile = cmd_args.source + ".fits"
hdulist= fits.open(fitsfile, dtype='uint8', mode='denywrite', memmap=True)
dta = hdulist[1].data
# m = dta['PhotoFlag'] == 0.0
# print(dta[m])

#table_aper = select('aper')
#table_petro = select('petro')
table_auto = select('auto')

# # #Saving resultated table
#######
#aper#
#######
#asciifile = fitsfile.replace(".fits", "-aper-Halpha.tab")
#try:
    #table_aper.write(asciifile, format='ascii.tab', overwrite=True)
#except TypeError:
#    table_aper.write(asciifile, format='ascii.tab')

#######
#petro#
#######
#asciifile_petro = fitsfile.replace(".fits", "-petro-Halpha.tab")
#try:
#    table_petro.write(asciifile_petro, format='ascii.tab', overwrite=True)
#except TypeError:
#    table_petro.write(asciifile_petro, format='ascii.tab')

######
#auto#
######
asciifile_auto = fitsfile.replace(".fits", "-auto-Halpha.tab")
try:
    table_auto.write(asciifile_auto, format='ascii.tab', overwrite=True)
except TypeError:
    table_auto.write(asciifile_auto, format='ascii.tab')
