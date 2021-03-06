from astropy.io import fits
import os
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

def select(magg):
    ef1 = tab['uJAVA_'+magg+'_err'] <= 0.2
    ef2 = tab['J0378_'+magg+'_err'] <= 0.2
    ef3 =  tab['J0395_'+magg+'_err'] <= 0.2
    ef4 =  tab['J0410_'+magg+'_err'] <= 0.2
    ef5 =  tab['J0430_'+magg+'_err'] <= 0.2
    ef6 =  tab['gSDSS_'+magg+'_err'] <= 0.2
    ef7 =  tab['J0515_'+magg+'_err'] <= 0.2
    ef8 =  tab['rSDSS_'+magg+'_err'] <= 0.2
    ef9 =  tab['J0660_'+magg+'_err'] <= 0.2
    ef10 =  tab['iSDSS_'+magg+'_err'] <= 0.2
    ef11 =  tab['J0861_'+magg+'_err'] <= 0.2
    ef12 =  tab['zSDSS_'+magg+'_err'] <= 0.2

    f1 = tab['uJAVA_'+magg] != 99.0
    f2 = tab['J0378_'+magg] != 99.0
    f3 =  tab['J0395_'+magg] != 99.0
    f4 =  tab['J0410_'+magg] != 99.0
    f5 =  tab['J0430_'+magg] != 99.0
    f6 =  tab['gSDSS_'+magg] != 99.0
    f7 =  tab['J0515_'+magg] != 99.0
    f8 =  tab['rSDSS_'+magg] != 99.0
    f9 =  tab['J0660_'+magg] != 99.0
    f10 =  tab['iSDSS_'+magg] != 99.0
    f11 =  tab['J0861_'+magg] != 99.0
    f12 =  tab['zSDSS_'+magg] != 99.0

    flag1 = tab['rSDSS_FLAGS'] <= 3.0
    flag2 = tab['gSDSS_FLAGS'] <= 3.0
    flag3 = tab['iSDSS_FLAGS'] <= 3.0
    flag4 = tab['zSDSS_FLAGS'] <= 3.0
    flag5 = tab['uJAVA_FLAGS'] <= 3.0
    flag6 = tab['J0378_FLAGS'] <= 3.0
    flag7 = tab['J0395_FLAGS'] <= 3.0
    flag8 = tab['J0410_FLAGS'] <= 3.0
    flag9 = tab['J0430_FLAGS'] <= 3.0
    flag10 = tab['J0515_FLAGS'] <= 3.0
    flag11 = tab['J0660_FLAGS'] <= 3.0
    flag12 = tab['J0861_FLAGS'] <= 3.0

    mask_flag1 = tab['rSDSS_MASK_FLAGS'] == 0.0
    mask_flag2 = tab['gSDSS_MASK_FLAGS'] == 0.0
    mask_flag3 = tab['iSDSS_MASK_FLAGS'] == 0.0
    mask_flag4 = tab['zSDSS_MASK_FLAGS'] == 0.0
    mask_flag5 = tab['uJAVA_MASK_FLAGS'] == 0.0
    mask_flag6 = tab['J0378_MASK_FLAGS'] == 0.0
    mask_flag7 = tab['J0395_MASK_FLAGS'] == 0.0
    mask_flag8 = tab['J0410_MASK_FLAGS'] == 0.0
    mask_flag9 = tab['J0430_MASK_FLAGS'] == 0.0
    mask_flag10 = tab['J0515_MASK_FLAGS'] == 0.0
    mask_flag11 = tab['J0660_MASK_FLAGS'] == 0.0
    mask_flag12 = tab['J0861_MASK_FLAGS'] == 0.0

    #color
    col1 = (tab['rSDSS_'+magg] - tab['iSDSS_'+magg]) <= 2.5
    col2 = (tab['rSDSS_'+magg] - tab['iSDSS_'+magg]) >= -2.5
    col3 = (tab['rSDSS_'+magg] - tab['J0660_'+magg]) <= 2.5

    total_col = col1 & col2 & col3
    
    total_flags = flag1 & flag2  & flag3  & flag4 & flag5  & flag6  & flag7 & flag8  & flag9 & flag10 & flag11  & flag12
    total_mask_flags = mask_flag1 & mask_flag2 & mask_flag3 & mask_flag4 & mask_flag5 & mask_flag6 & mask_flag7 & mask_flag8 & mask_flag9 & mask_flag10 & mask_flag11 & mask_flag12
    #errors = ef1 & ef2 & ef3 & ef4 & ef5 & ef6 & ef7 & ef8 & ef9 & ef10 & ef11 & ef12
    errors = ef1 & ef6 & ef8 & ef10 & ef12
    
    #mask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & total_flags & total_col & errors & total_mask_flags

    mask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10 & f11 & f12 & total_col & total_flags 
    
    table = Table(tab[mask])
    return table
    
# pattern = "*.cat"
# file_list = glob.glob(pattern)

parser = argparse.ArgumentParser(
    description="""Make a table from the J-PLUS catalogs""")

parser.add_argument("source", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
fitsfile = cmd_args.source + ".tab"
tab = Table.read(fitsfile, format="ascii.tab")
  
#table_aper = select('aper', mag_aper)
#table_petro = select('petro', mag_petro)MAG_APER_6_0
table_Aper = select('MAG_APER_6_0')
table_auto = select('auto')

# # #Saving resultated table

######
#Aper#
######
asciifile_Aper = fitsfile.replace(".tab", "-cleaning-limfilter-limcolor-flags.tab")
#asciifile_auto = fitsfile.replace(".fits", ".tab")
try:
    table_Aper.write(asciifile_Aper, format='ascii.tab', overwrite=True)
except TypeError:
    table_Aper.write(asciifile_Aper, format='ascii.tab')
