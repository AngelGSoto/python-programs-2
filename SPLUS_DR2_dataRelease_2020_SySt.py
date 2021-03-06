'''
Getting the SySt from S-PLUS catalog (DR2)
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
    #f1 = dta['euJAVA_'+magg] <= 0.2
    #f2 = dta['eF378_'+magg] <= 0.2
    #f3 = dta['eF395_'+magg] < 0.2
    #f4 = dta['eF410_'+magg] <= 0.2
    #f5 = dta['eF430_'+magg] < 0.2
    f6 = dta['eg_'+magg] <= 0.2
    #f7 = dta['eF515_'+magg] <= 0.2
    f8 = dta['er_'+magg] <= 0.2
    f9 = dta['eF660_'+magg] <= 0.2
    f10 = dta['ei_'+magg] <= 0.2
    #f11 = dta['eF861_'+magg] <= 0.2
    #f12 = dta['ez_'+magg] <= 0.2

    #mask =  f1 & f2 & f3 & f5 & f6 & f7 & f8 & f9 & f10 & f11 & f12
    mask =   f6 & f8 & f9 & f10 
    # # #Mask 1
    # q = (dta['F515_'+magg] - dta['F660_'+magg]) <= 5.0
    # q1 = (dta['F515_'+magg] - dta['F861_'+magg]) >= -3.0
    # q2 = (dta['r_'+magg] - dta['F660_'+magg]) <= 4.0
    # q3 = (dta['r_'+magg] - dta['i_'+magg]) >= -4.0
    # q4 = (dta['g_'+magg] - dta['F515_'+magg]) >= -3.2
    # q5 = (dta['g_'+magg] - dta['i_'+magg]) >= -3.0
    # q6 = (dta['F410_'+magg] - dta['F660_'+magg]) <= 6.0
    # q7 = (dta['z_'+magg] - dta['g_'+magg]) <= 4.0

    #mask1 = q & q1 & q2 & q3 & q4 & q5 & q6 & q7 
    #Mask
    Y = 5.5*(dta['F515_'+magg] - dta['F861_'+magg]) - 6.45
    Y1 = 0.98*(dta['F515_'+magg] - dta['F861_'+magg]) - 0.16
    Y2 = -1.96*(dta['z_'+magg] - dta['g_'+magg]) - 3.15
    Y3 = 0.2*(dta['z_'+magg] - dta['g_'+magg]) + 0.44 
    Y4 = -220*(dta['r_'+magg] - dta['i_'+magg]) + 40.4
    Y44 = 0.39*(dta['r_'+magg] - dta['i_'+magg]) + 0.73
    Y5 = -4.7*(dta['g_'+magg] - dta['i_'+magg]) + 10.60
    Y6 = 2.13*(dta['g_'+magg] - dta['i_'+magg]) - 1.43
    Y7 = -0.19*(dta['F660_'+magg] - dta['r_'+magg]) - 0.05
    Y8 = -2.66*(dta['F660_'+magg] - dta['r_'+magg]) - 2.2

    #m = dta['PROB_STAR'] >= 0.0
    m = dta['PhotoFlag'] == 0.0
    m1 = dta['F515_'+magg] - dta['F660_'+magg]<= Y
    m2 = dta['F515_'+magg] - dta['F660_'+magg]>= Y1
    m3 = dta['z_'+magg] - dta['F660_'+magg]<= Y2
    m4 = dta['z_'+magg] - dta['F660_'+magg]>= Y3
    m5 = dta['r_'+magg] - dta['F660_'+magg]>= Y4
    m6 = dta['r_'+magg] - dta['F660_'+magg]>= Y44
    m7 = dta['F410_'+magg] - dta['F660_'+magg]>= Y5
    m8 = dta['F410_'+magg] - dta['F660_'+magg]>= Y6
    m9 = dta['g_'+magg] - dta['F515_'+magg]>= Y7
    m10 = dta['g_'+magg] - dta['F515_'+magg]<= Y8
    total_m = m & m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8 & m9 & m10 & mask 
    
    table = Table(dta[total_m])
    return table
   
# pattern = "*.cat"
# file_list = glob.glob(pattern)

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
fitsfile = cmd_args.source + ".fits"

#datadir = "../"
#hdulist= fits.open(os.path.join(datadir, fitsfile), dtype='uint8', mode='denywrite', memmap=True)
hdulist= fits.open(fitsfile, mode='denywrite', memmap=True)
dta = hdulist[1].data
# m = dta['PhotoFlag'] == 0.0
# print(dta[m])

table_aper = select('aper')
table_petro = select('petro')
table_auto = select('auto')

# # #Saving resultated table
#######
#aper#
#######
asciifile = fitsfile.replace(".fits", "-aper-SySt.tab")
try:
    table_aper.write(asciifile, format='ascii.tab', overwrite=True)
except TypeError:
    table_aper.write(asciifile, format='ascii.tab')

#######
#petro#
#######
asciifile_petro = fitsfile.replace(".fits", "-petro-SySt.tab")
try:
    table_petro.write(asciifile_petro, format='ascii.tab', overwrite=True)
except TypeError:
    table_petro.write(asciifile_petro, format='ascii.tab')

######
#auto#
######
asciifile_auto = fitsfile.replace(".fits", "-auto-SySt.tab")
try:
    table_auto.write(asciifile_auto, format='ascii.tab', overwrite=True)
except TypeError:
    table_auto.write(asciifile_auto, format='ascii.tab')

