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

#create list with the magnitudes
n = 77
magnitude = [[] for _ in range(n)]

parser = argparse.ArgumentParser(
    description="""Write the file""")

parser.add_argument("source", type=str,
                    default="table-6mil-obj-jplus",
                    help="Name of source, taken the prefix ")

parser.add_argument("--savefile", action="store_true",
                    help="Save ascii file showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
file_ = args.source + ".dat"

def mag(type_mag):
    dta = tab[type_mag]
    dta = pd.Series(dta)
    dta_exp = dta.str.split(' ', expand=True)
    return dta_exp

#Read files
tab = ascii.read(file_)

Tile = tab["TILE_ID"]
Number = tab["NUMBER"]
RA = tab["ALPHA_J2000"]
DEC = tab["DELTA_J2000"]
petro_radius = tab["PETRO_RADIUS"]
kron_radius = tab["KRON_RADIUS"]
r_auto = mag("MAG_AUTO")[0]
g_auto = mag("MAG_AUTO")[1]
i_auto = mag("MAG_AUTO")[2]
z_auto = mag("MAG_AUTO")[3]
uJAVA_auto = mag("MAG_AUTO")[4]
J0378_auto = mag("MAG_AUTO")[5]
J0395_auto = mag("MAG_AUTO")[6]
J0410_auto = mag("MAG_AUTO")[7]
J0430_auto = mag("MAG_AUTO")[8]
J0515_auto = mag("MAG_AUTO")[9]
J0660_auto = mag("MAG_AUTO")[10]
J0861_auto = mag("MAG_AUTO")[11]

r_MAG_PETRO = mag("MAG_PETRO")[0]
g_MAG_PETRO = mag("MAG_PETRO")[1]
i_MAG_PETRO = mag("MAG_PETRO")[2]
z_MAG_PETRO = mag("MAG_PETRO")[3]
uJAVA_MAG_PETRO = mag("MAG_PETRO")[4]
J0378_MAG_PETRO = mag("MAG_PETRO")[5]
J0395_MAG_PETRO = mag("MAG_PETRO")[6]
J0410_MAG_PETRO = mag("MAG_PETRO")[7]
J0430_MAG_PETRO = mag("MAG_PETRO")[8]
J0515_MAG_PETRO = mag("MAG_PETRO")[9]
J0660_MAG_PETRO = mag("MAG_PETRO")[10]
J0861_MAG_PETRO = mag("MAG_PETRO")[11]

r_MAG_APER_6_0 = mag("MAG_APER_6_0")[0]
g_MAG_APER_6_0 = mag("MAG_APER_6_0")[1]
i_MAG_APER_6_0 = mag("MAG_APER_6_0")[2]
z_MAG_APER_6_0 = mag("MAG_APER_6_0")[3]
uJAVA_MAG_APER_6_0 = mag("MAG_APER_6_0")[4]
J0378_MAG_APER_6_0 = mag("MAG_APER_6_0")[5]
J0395_MAG_APER_6_0 = mag("MAG_APER_6_0")[6]
J0410_MAG_APER_6_0 = mag("MAG_APER_6_0")[7]
J0430_MAG_APER_6_0 = mag("MAG_APER_6_0")[8]
J0515_MAG_APER_6_0 = mag("MAG_APER_6_0")[9]
J0660_MAG_APER_6_0 = mag("MAG_APER_6_0")[10]
J0861_MAG_APER_6_0 = mag("MAG_APER_6_0")[11]

#ERROR
r_auto_err = mag("MAG_ERR_AUTO")[0]
g_auto_err = mag("MAG_ERR_AUTO")[1]
i_auto_err = mag("MAG_ERR_AUTO")[2]
z_auto_err = mag("MAG_ERR_AUTO")[3]
uJAVA_auto_err = mag("MAG_ERR_AUTO")[4]
J0378_auto_err = mag("MAG_ERR_AUTO")[5]
J0395_auto_err = mag("MAG_ERR_AUTO")[6]
J0410_auto_err = mag("MAG_ERR_AUTO")[7]
J0430_auto_err = mag("MAG_ERR_AUTO")[8]
J0515_auto_err = mag("MAG_ERR_AUTO")[9]
J0660_auto_err = mag("MAG_ERR_AUTO")[10]
J0861_auto_err = mag("MAG_ERR_AUTO")[11]

r_MAG_PETRO_ERR  = mag("MAG_ERR_PETRO")[0]
g_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[1]
i_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[2]
z_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[3]
# try:
#     uJAVA_MAG_PETRO_ERR = mag("MAG_ERR_ISO_GAUS")[4]
# except KeyError:
#     None
uJAVA_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[4]
J0378_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[5]
J0395_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[6]
J0410_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[7]
J0430_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[8]
J0515_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[9]
J0660_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[10]
J0861_MAG_PETRO_ERR = mag("MAG_ERR_PETRO")[11]

r_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[0]
g_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[1]
i_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[2]
z_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[3]
uJAVA_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[4]
J0378_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[5]
J0395_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[6]
J0410_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[7]
J0430_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[8]
J0515_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[9]
J0660_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[10]
J0861_MAG_APER_6_0_err = mag("MAG_ERR_APER_6_0")[11]

#FLAGS
r_FLAGS = mag("FLAGS")[0]
g_FLAGS = mag("FLAGS")[1]
i_FLAGS = mag("FLAGS")[2]
z_FLAGS = mag("FLAGS")[3]
uJAVA_FLAGS = mag("FLAGS")[4]
J0378_FLAGS = mag("FLAGS")[5]
J0395_FLAGS = mag("FLAGS")[6]
J0410_FLAGS = mag("FLAGS")[7]
J0430_FLAGS = mag("FLAGS")[8]
J0515_FLAGS = mag("FLAGS")[9]
J0660_FLAGS = mag("FLAGS")[10]
J0861_FLAGS = mag("FLAGS")[11]

#MASK
r_MASK_FLAGS = mag("MASK_FLAGS")[0]
g_MASK_FLAGS = mag("MASK_FLAGS")[1]
i_MASK_FLAGS = mag("MASK_FLAGS")[2]
z_MASK_FLAGS = mag("MASK_FLAGS")[3]
uJAVA_MASK_FLAGS = mag("MASK_FLAGS")[4]
J0378_MASK_FLAGS = mag("MASK_FLAGS")[5]
J0395_MASK_FLAGS = mag("MASK_FLAGS")[6]
J0410_MASK_FLAGS = mag("MASK_FLAGS")[7]
J0430_MASK_FLAGS = mag("MASK_FLAGS")[8]
J0515_MASK_FLAGS = mag("MASK_FLAGS")[9]
J0660_MASK_FLAGS = mag("MASK_FLAGS")[10]
J0861_MASK_FLAGS = mag("MASK_FLAGS")[11]

#Creating table
table = Table([Tile, Number, RA, DEC, petro_radius, kron_radius, r_auto, g_auto, i_auto, z_auto, uJAVA_auto, J0378_auto, J0395_auto,  J0410_auto , J0430_auto, J0515_auto, J0660_auto, J0861_auto, r_MAG_PETRO, g_MAG_PETRO, i_MAG_PETRO, z_MAG_PETRO, uJAVA_MAG_PETRO, J0378_MAG_PETRO, J0395_MAG_PETRO,  J0410_MAG_PETRO , J0430_MAG_PETRO, J0515_MAG_PETRO, J0660_MAG_PETRO, J0861_MAG_PETRO, r_MAG_APER_6_0, g_MAG_APER_6_0, i_MAG_APER_6_0, z_MAG_APER_6_0, uJAVA_MAG_APER_6_0, J0378_MAG_APER_6_0, J0395_MAG_APER_6_0,  J0410_MAG_APER_6_0 , J0430_MAG_APER_6_0, J0515_MAG_APER_6_0, J0660_MAG_APER_6_0, J0861_MAG_APER_6_0, r_auto_err, g_auto_err, i_auto_err, z_auto_err, uJAVA_auto_err, J0378_auto_err, J0395_auto_err,  J0410_auto_err, J0430_auto_err, J0515_auto_err, J0660_auto_err, J0861_auto_err, r_MAG_PETRO_ERR, g_MAG_PETRO_ERR, i_MAG_PETRO_ERR, z_MAG_PETRO_ERR, uJAVA_MAG_PETRO_ERR, J0378_MAG_PETRO_ERR, J0395_MAG_PETRO_ERR, J0410_MAG_PETRO_ERR , J0430_MAG_PETRO_ERR, J0515_MAG_PETRO_ERR, J0660_MAG_PETRO_ERR, J0861_MAG_PETRO_ERR, r_MAG_APER_6_0_err, g_MAG_APER_6_0_err, i_MAG_APER_6_0_err, z_MAG_APER_6_0_err, uJAVA_MAG_APER_6_0_err, J0378_MAG_APER_6_0_err, J0395_MAG_APER_6_0_err,  J0410_MAG_APER_6_0_err , J0430_MAG_APER_6_0_err, J0515_MAG_APER_6_0_err, J0660_MAG_APER_6_0_err, J0861_MAG_APER_6_0_err, r_FLAGS, g_FLAGS, i_FLAGS, z_FLAGS, uJAVA_FLAGS, J0378_FLAGS, J0395_FLAGS, J0410_FLAGS, J0430_FLAGS, J0515_FLAGS, J0660_FLAGS, J0861_FLAGS, r_MASK_FLAGS, g_MASK_FLAGS, i_MASK_FLAGS, z_MASK_FLAGS, uJAVA_MASK_FLAGS, J0378_MASK_FLAGS, J0395_MASK_FLAGS, J0410_MASK_FLAGS, J0430_MASK_FLAGS, J0515_MASK_FLAGS, J0660_MASK_FLAGS, J0861_MASK_FLAGS], names=('Tile', 'Number', 'RA', 'Dec', 'petro_radius', 'kron_radius', 'rSDSS_auto', 'gSDSS_auto', 'iSDSS_auto', 'zSDSS_auto', 'uJAVA_auto', 'J0378_auto', 'J0395_auto', 'J0410_auto', 'J0430_auto', 'J0515_auto', 'J0660_auto', 'J0861_auto', 'rSDSS_PETRO', 'gSDSS_PETRO', 'iSDSS_PETRO', 'zSDSS_PETRO', 'uJAVA_PETRO', 'J0378_PETRO', 'J0395_PETRO', 'J0410_PETRO', 'J0430_PETRO', 'J0515_PETRO', 'J0660_PETRO', 'J0861_PETRO', 'rSDSS_MAG_APER_6_0', 'gSDSS_MAG_APER_6_0', 'iSDSS_MAG_APER_6_0', 'zSDSS_MAG_APER_6_0', 'uJAVA_MAG_APER_6_0', 'J0378_MAG_APER_6_0', 'J0395_MAG_APER_6_0', 'J0410_MAG_APER_6_0', 'J0430_MAG_APER_6_0', 'J0515_MAG_APER_6_0', 'J0660_MAG_APER_6_0', 'J0861_MAG_APER_6_0', 'rSDSS_auto_err', 'gSDSS_auto_err', 'iSDSS_auto_err', 'zSDSS_auto_err', 'uJAVA_auto_err', 'J0378_auto_err', 'J0395_auto_err', 'J0410_auto_err', 'J0430_auto_err', 'J0515_auto_err', 'J0660_auto_err', 'J0861_auto_err',  'rSDSS_PETRO_err', 'gSDSS_PETRO_err', 'iSDSS_PETRO_err', 'zSDSS_PETRO_err', 'uJAVA_PETRO_err', 'J0378_PETRO_err', 'J0395_PETRO_err', 'J0410_PETRO_err', 'J0430_PETRO_err', 'J0515_PETRO_err', 'J0660_PETRO_err', 'J0861_PETRO_err','rSDSS_MAG_APER_6_0_err', 'gSDSS_MAG_APER_6_0_err', 'iSDSS_MAG_APER_6_0_err', 'zSDSS_MAG_APER_6_0_err', 'uJAVA_MAG_APER_6_0_err', 'J0378_MAG_APER_6_0_err', 'J0395_MAG_APER_6_0_err', 'J0410_MAG_APER_6_0_err', 'J0430_MAG_APER_6_0_err', 'J0515_MAG_APER_6_0_err', 'J0660_MAG_APER_6_0_err', 'J0861_MAG_APER_6_0_err', 'rSDSS_FLAGS', 'gSDSS_FLAGS', 'iSDSS_FLAGS', 'zSDSS_FLAGS', 'uJAVA_FLAGS', 'J0378_FLAGS', 'J0395_FLAGS', 'J0410_FLAGS', 'J0430_FLAGS', 'J0515_FLAGS', 'J0660_FLAGS', 'J0861_FLAGS', 'rSDSS_MASK_FLAGS', 'gSDSS_MASK_FLAGS', 'iSDSS_MASK_FLAGS', 'zSDSS_MASK_FLAGS', 'uJAVA_MASK_FLAGS', 'J0378_MASK_FLAGS', 'J0395_MASK_FLAGS', 'J0410_MASK_FLAGS', 'J0430_MASK_FLAGS', 'J0515_MASK_FLAGS', 'J0660_MASK_FLAGS', 'J0861_MASK_FLAGS'), meta={'name': 'first table'})  

#Saving resultated table
if args.savefile:
    asciifile = file_.replace(".dat", ".tab")
    table.write(asciifile, format="ascii.tab")


