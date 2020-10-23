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
    ef1 = hdulist[1].data['e_U_'+magg] <= 0.2
    ef2 = hdulist[1].data['e_F378_'+magg] <= 0.2
    ef3 =  hdulist[1].data['e_F395_'+magg] < 0.2
    ef4 =  hdulist[1].data['e_F410_'+magg] <= 0.2
    ef5 =  hdulist[1].data['e_F430_'+magg] < 0.2
    ef6 =  hdulist[1].data['e_G_'+magg] <= 0.2
    ef7 =  hdulist[1].data['e_F515_'+magg] <= 0.2
    ef8 =  hdulist[1].data['e_R_'+magg] <= 0.2
    ef9 =  hdulist[1].data['e_F660_'+magg] <= 0.2
    ef10 =  hdulist[1].data['e_I_'+magg] <= 0.2
    ef11 =  hdulist[1].data['e_F861_'+magg] <= 0.2
    ef12 =  hdulist[1].data['e_Z_'+magg] <= 0.2

    f1 = hdulist[1].data['U_'+magg] != 99.0
    f2 = hdulist[1].data['F378_'+magg] != 99.0
    f3 =  hdulist[1].data['F395_'+magg] != 99.0
    f4 =  hdulist[1].data['F410_'+magg] != 99.0
    f5 =  hdulist[1].data['F430_'+magg] != 99.0
    f6 =  hdulist[1].data['G_'+magg] != 99.0
    f7 =  hdulist[1].data['F515_'+magg] != 99.0
    f8 =  hdulist[1].data['R_'+magg] != 99.0
    f9 =  hdulist[1].data['F660_'+magg] != 99.0
    f10 =  hdulist[1].data['I_'+magg] != 99.0
    f11 =  hdulist[1].data['F861_'+magg] != 99.0
    f12 =  hdulist[1].data['Z_'+magg] != 99.0

    #m = hdulist[1].data['PhotoFlag'] == 0.0
    
    #m1 = hdulist[1].data['r_'+magg] <=19
    # y = 0.108*(hdulist[1].data['r_'+magg] - hdulist[1].data['i_'+magg]) + 0.5
    # m = hdulist[1].data['r_'+magg] - hdulist[1].data['eF660_'+magg] >= y
    
    #mask = ef1 & ef2 & ef3 & ef4 & ef5 & ef6 & ef7 & ef8 & ef9 & ef10 & ef11 & ef12 & f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10 & f11 & f12 & m1
    #mask =   ef8 & ef9 & ef10 & f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10 & f11 & f12 & m1 
    mask =  f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10 & f11 & f12 

    table = Table(hdulist[1].data[mask])
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
hdulist= fits.open(fitsfile, mode='denywrite', memmap=True)
  
#table_aper = select('aper', mag_aper)
#table_petro = select('petro', mag_petro)
table_auto = select('PStotal')

# # #Saving resultated table
#######
#aper#
#######
# asciifile = fitsfile.replace(".fits", "-aper-HalphaVii.tab")
# try:
#     table_aper.write(asciifile, format='ascii.tab', overwrite=True)
# except TypeError:
#     table_aper.write(asciifile, format='ascii.tab')

# #######
# #petro#
# #######
# asciifile_petro = fitsfile.replace(".fits", "-petro-HalphaVii.tab")
# try:
#     table_petro.write(asciifile_petro, format='ascii.tab', overwrite=True)
# except TypeError:
#     table_petro.write(asciifile_petro, format='ascii.tab')

######
#auto#
######
asciifile_auto = fitsfile.replace(".fits", "-99.tab")
#asciifile_auto = fitsfile.replace(".fits", ".tab")
try:
    table_auto.write(asciifile_auto, format='ascii.tab', overwrite=True)
except TypeError:
    table_auto.write(asciifile_auto, format='ascii.tab')

