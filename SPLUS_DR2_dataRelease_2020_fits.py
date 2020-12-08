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

def select(magg, mag):
    ef1 = hdulist[1].data['euJAVA_'+magg] <= 0.2
    ef2 = hdulist[1].data['eF378_'+magg] <= 0.2
    ef3 =  hdulist[1].data['eF395_'+magg] < 0.2
    ef4 =  hdulist[1].data['eF410_'+magg] <= 0.2
    ef5 =  hdulist[1].data['eF430_'+magg] < 0.2
    ef6 =  hdulist[1].data['eg_'+magg] <= 0.2
    ef7 =  hdulist[1].data['eF515_'+magg] <= 0.2
    ef8 =  hdulist[1].data['er_'+magg] <= 0.2
    ef9 =  hdulist[1].data['eF660_'+magg] <= 0.2
    ef10 =  hdulist[1].data['ei_'+magg] <= 0.2
    ef11 =  hdulist[1].data['eF861_'+magg] <= 0.2
    ef12 =  hdulist[1].data['ez_'+magg] <= 0.2

    mask = ef1 & ef2 & ef3 & ef5 & ef6 & ef7 & ef8 & ef9 & ef10 & ef11 & ef12

    #Mask
    #Mask filters
    q = (hdulist[1].data['F515_'+magg] - hdulist[1].data['F660_'+magg]) <= 5.0
    #q1 = (data['J0515'] - data['J0861']) >= -3.0
    q2 = (hdulist[1].data['r_'+magg] - hdulist[1].data['F660_'+magg]) <= 4.0
    q3 = (hdulist[1].data['r_'+magg] - hdulist[1].data['i_'+magg]) >= -4.0
    q4 = (hdulist[1].data['g_'+magg] - hdulist[1].data['F515_'+magg]) >= -3.2
    q5 = (hdulist[1].data['g_'+magg] - hdulist[1].data['i_'+magg]) >= -3.0
    q6 = (hdulist[1].data['F410_'+magg] - hdulist[1].data['F660_'+magg]) <= 6.0
    q7 = (hdulist[1].data['z_'+magg] - hdulist[1].data['g_'+magg]) <= 4.0

    mask1 = q & q2 & q3 & q4 & q5 & q6 & q7
    #Mask
    Y = 2.7*(hdulist[1].data['F515_'+magg] - hdulist[1].data['F861_'+magg]) + 2.15
    Y1 = 0.2319*(hdulist[1].data['z_'+magg] - hdulist[1].data['g_'+magg]) + 0.85
    Y2 = -1.3*(hdulist[1].data['z_'+magg] - hdulist[1].data['g_'+magg]) + 1.7
    Y3 = 1.559*(hdulist[1].data['F660_'+magg] - hdulist[1].data['r_'+magg]) + 0.58
    Y4 = 0.12*(hdulist[1].data['F660_'+magg] - hdulist[1].data['r_'+magg]) - 0.01
    Y44 = -1.1*(hdulist[1].data['F660_'+magg] - hdulist[1].data['r_'+magg]) - 1.07
    Y5 = 8.0*(hdulist[1].data['g_'+magg] - hdulist[1].data['i_'+magg]) + 4.5
    Y6 = 0.8*(hdulist[1].data['g_'+magg] - hdulist[1].data['i_'+magg]) + 0.55
    Y7 = 0.43*(hdulist[1].data['r_'+magg] - hdulist[1].data['i_'+magg]) + 0.65
    Y8 = -6.8*(hdulist[1].data['r_'+magg] - hdulist[1].data['i_'+magg]) - 1.3

    m = hdulist[1].data['PhotoFlag'] == 0.0
    #m = data['Class star'] > 0.0 
    m1 = hdulist[1].data['F660_'+magg] - hdulist[1].data['r_'+magg]<=-1.0 
    m2 = hdulist[1].data['F515_'+magg] - hdulist[1].data['F660_'+magg]>=0.3
    m3 = hdulist[1].data['z_'+magg] - hdulist[1].data['F660_'+magg]>=Y1
    m4 = hdulist[1].data['z_'+magg] - hdulist[1].data['F660_'+magg]>=Y2
    m5 = hdulist[1].data['F515_'+magg] - hdulist[1].data['F660_'+magg]>=Y
    m6 = hdulist[1].data['F660_'+magg] - hdulist[1].data['g_'+magg]>=Y3
    m7 = hdulist[1].data['g_'+magg] - hdulist[1].data['F515_'+magg]<=Y4
    m8 = hdulist[1].data['g_'+magg] - hdulist[1].data['F515_'+magg]<=Y44
    m9 = hdulist[1].data['F410_'+magg] - hdulist[1].data['F660_'+magg]>=Y5
    m10 = hdulist[1].data['F410_'+magg] - hdulist[1].data['F660_'+magg]>=Y6
    m11 = (hdulist[1].data['r_'+magg] - hdulist[1].data['F660_'+magg]) >= Y7
    m12 = (hdulist[1].data['r_'+magg] - hdulist[1].data['F660_'+magg]) <= Y8
    total_m = m & m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8 & m9 & m10 & m11 & m12 & mask & mask1

    for i in hdulist[1].data['FIELD'][total_m]:
        mag[0].append(i)
    for i in hdulist[1].data['ID'][total_m]:
        mag[1].append(i)
    for i in hdulist[1].data['RA'][total_m]:
        mag[2].append(i)
    for i in hdulist[1].data['Dec'][total_m]:
        mag[3].append(i)
    for i in hdulist[1].data['X'][total_m]:
        mag[4].append(i)
    for i in hdulist[1].data['Y'][total_m]:
        mag[5].append(i)
    for i in hdulist[1].data['ISOarea'][total_m]:
        mag[6].append(i)
    for i in hdulist[1].data['s2nDet'][total_m]:
        mag[7].append(i)
    for i in hdulist[1].data['PhotoFlag'][total_m]:
        mag[8].append(i)
    for i in hdulist[1].data['FWHM'][total_m]:
        mag[9].append(i)
    for i in hdulist[1].data['FWHM_n'][total_m]:
        mag[10].append(i)
    for i in hdulist[1].data['MUMAX'][total_m]:
        mag[11].append(i)
    for i in hdulist[1].data['A'][total_m]:
        mag[12].append(i)
    for i in hdulist[1].data['B'][total_m]:
        mag[13].append(i)
    for i in hdulist[1].data['THETA'][total_m]:
        mag[14].append(i)
    for i in hdulist[1].data['FlRadDet'][total_m]:
        mag[15].append(i)
    for i in hdulist[1].data['KrRadDet'][total_m]:
        mag[16].append(i)
    for i in hdulist[1].data['nDet_auto'][total_m]:
        mag[17].append(i)
    for i in hdulist[1].data['nDet_petro'][total_m]:
        mag[18].append(i)
    for i in hdulist[1].data['nDet_aper'][total_m]:         
        mag[19].append(i)
    for i in hdulist[1].data['uJAVA_auto'][total_m]:
        mag[20].append(i)
    for i in hdulist[1].data['euJAVA_auto'][total_m]:
        mag[21].append(i)
    for i in hdulist[1].data['s2n_uJAVA_auto'][total_m]:
        mag[22].append(i)
    for i in hdulist[1].data['uJAVA_petro'][total_m]:
        mag[23].append(i)
    for i in hdulist[1].data['euJAVA_petro'][total_m]:
        mag[24].append(i)
    for i in hdulist[1].data['s2n_uJAVA_petro'][total_m]:
        mag[25].append(i)
    for i in hdulist[1].data['uJAVA_aper'][total_m]:
        mag[26].append(i)
    for i in hdulist[1].data['euJAVA_aper'][total_m]:
        mag[27].append(i)
    for i in hdulist[1].data['S2n_uJAVA_aper'][total_m]:
        mag[28].append(i)
    for i in hdulist[1].data['F378_auto'][total_m]:
        mag[29].append(i)
    for i in hdulist[1].data['eF378_auto'][total_m]:
        mag[30].append(i)
    for i in hdulist[1].data['s2n_F378_auto'][total_m]:
        mag[31].append(i)
    for i in hdulist[1].data['F378_petro'][total_m]:
        mag[32].append(i)
    for i in hdulist[1].data['eF378_petro'][total_m]:
        mag[33].append(i)
    for i in hdulist[1].data['s2n_F378_petro'][total_m]:
        mag[34].append(i)
    for i in hdulist[1].data['F378_aper'][total_m]:
        mag[35].append(i)
    for i in hdulist[1].data['eF378_aper'][total_m]:
        mag[36].append(i)
    for i in hdulist[1].data['s2n_F378_aper'][total_m]:
        mag[37].append(i)
    for i in hdulist[1].data['F395_auto'][total_m]:
        mag[38].append(i)
    for i in hdulist[1].data['eF395_auto'][total_m]:
        mag[39].append(i)
    for i in hdulist[1].data['s2n_F395_auto'][total_m]:
        mag[40].append(i)
    for i in hdulist[1].data['F395_petro'][total_m]:
        mag[41].append(i)
    for i in hdulist[1].data['eF395_petro'][total_m]:
        mag[42].append(i)
    for i in hdulist[1].data['s2n_F395_petro'][total_m]:
        mag[43].append(i)
    for i in hdulist[1].data['F395_aper'][total_m]:
        mag[44].append(i)
    for i in hdulist[1].data['eF395_aper'][total_m]:
        mag[45].append(i)
    for i in hdulist[1].data['s2n_F395_aper'][total_m]:
        mag[46].append(i)
    for i in hdulist[1].data['F410_auto'][total_m]:
        mag[47].append(i)
    for i in hdulist[1].data['eF410_auto'][total_m]:
        mag[48].append(i)
    for i in hdulist[1].data['s2n_F410_auto'][total_m]:
        mag[49].append(i)
    for i in hdulist[1].data['F410_petro'][total_m]:
        mag[50].append(i)
    for i in hdulist[1].data['eF410_petro'][total_m]:
        mag[51].append(i)
    for i in hdulist[1].data['s2n_F410_petro'][total_m]:
        mag[52].append(i)
    for i in hdulist[1].data['F410_aper'][total_m]: 
        mag[53].append(i)
    for i in hdulist[1].data['eF410_aper'][total_m]:
        mag[54].append(i)
    for i in hdulist[1].data['s2n_F410_aper'][total_m]:
        mag[55].append(i)
    for i in hdulist[1].data['F430_auto'][total_m]:
        mag[56].append(i)
    for i in hdulist[1].data['eF430_auto'][total_m]:
        mag[57].append(i)
    for i in hdulist[1].data['s2n_F430_auto'][total_m]:
        mag[58].append(i)
    for i in hdulist[1].data['F430_petro'][total_m]:
        mag[59].append(i)
    for i in hdulist[1].data['eF430_petro'][total_m]:
        mag[60].append(i)
    for i in hdulist[1].data['s2n_F430_petro'][total_m]:
        mag[61].append(i)
    for i in hdulist[1].data['F430_aper'][total_m]:
        mag[62].append(i)
    for i in hdulist[1].data['eF430_aper'][total_m]:
        mag[63].append(i)
    for i in hdulist[1].data['s2n_F430_aper'][total_m]:
        mag[64].append(i)
    for i in hdulist[1].data['g_auto'][total_m]:
        mag[65].append(i)
    for i in hdulist[1].data['eg_auto'][total_m]:
        mag[66].append(i)
    for i in hdulist[1].data['s2n_g_auto'][total_m]:
        mag[67].append(i)
    for i in hdulist[1].data['g_petro'][total_m]:
        mag[68].append(i)
    for i in hdulist[1].data['eg_petro'][total_m]:
        mag[69].append(i)
    for i in hdulist[1].data['s2n_g_petro'][total_m]:
        mag[70].append(i)
    for i in hdulist[1].data['g_aper'][total_m]:
        mag[71].append(i)
    for i in hdulist[1].data['eg_aper'][total_m]:
        mag[72].append(i)
    for i in hdulist[1].data['s2n_g_aper'][total_m]:
        mag[73].append(i)
    for i in hdulist[1].data['F515_auto'][total_m]:
        mag[74].append(i)
    for i in hdulist[1].data['eF515_auto'][total_m]:
        mag[75].append(i)
    for i in hdulist[1].data['s2n_F515_auto'][total_m]:
        mag[76].append(i)
    for i in hdulist[1].data['F515_petro'][total_m]:
        mag[77].append(i)
    for i in hdulist[1].data['eF515_petro'][total_m]:
        mag[78].append(i)
    for i in hdulist[1].data['s2n_F515_petro'][total_m]:
        mag[79].append(i)
    for i in hdulist[1].data['F515_aper'][total_m]: 
        mag[80].append(i)
    for i in hdulist[1].data['eF515_aper'][total_m]:
        mag[81].append(i)
    for i in hdulist[1].data['s2n_F515_aper'][total_m]:
        mag[82].append(i)
    for i in hdulist[1].data['r_auto'][total_m]:
        mag[83].append(i)
    for i in hdulist[1].data['er_auto'][total_m]:
        mag[84].append(i)
    for i in hdulist[1].data['s2n_r_auto'][total_m]:
        mag[85].append(i)
    for i in hdulist[1].data['r_petro'][total_m]:
        mag[86].append(i)
    for i in hdulist[1].data['er_petro'][total_m]:
        mag[87].append(i)
    for i in hdulist[1].data['s2n_r_petro'][total_m]:
        mag[88].append(i)
    for i in hdulist[1].data['r_aper'][total_m]:
        mag[89].append(i)
    for i in hdulist[1].data['er_aper'][total_m]:
        mag[90].append(i)
    for i in hdulist[1].data['s2n_r_aper'][total_m]:
        mag[91].append(i)
    for i in hdulist[1].data['F660_auto'][total_m]:
        mag[92].append(i)
    for i in hdulist[1].data['eF660_auto'][total_m]:
        mag[93].append(i)
    for i in hdulist[1].data['s2n_F660_auto'][total_m]:
        mag[94].append(i)
    for i in hdulist[1].data['F660_petro'][total_m]:
        mag[95].append(i)
    for i in hdulist[1].data['eF660_petro'][total_m]:
        mag[96].append(i)
    for i in hdulist[1].data['s2n_F660_petro'][total_m]:
        mag[97].append(i)
    for i in hdulist[1].data['F660_aper'][total_m]:
        mag[98].append(i)
    for i in hdulist[1].data['eF660_aper'][total_m]:
        mag[99].append(i)
    for i in hdulist[1].data['s2n_F660_aper'][total_m]:
        mag[100].append(i)
    for i in hdulist[1].data['i_auto'][total_m]:
        mag[101].append(i)
    for i in hdulist[1].data['ei_auto'][total_m]:
        mag[102].append(i)
    for i in hdulist[1].data['s2n_i_auto'][total_m]:
        mag[103].append(i)
    for i in hdulist[1].data['i_petro'][total_m]:
        mag[104].append(i)
    for i in hdulist[1].data['ei_petro'][total_m]:
        mag[105].append(i)
    for i in hdulist[1].data['s2n_i_petro'][total_m]:
        mag[106].append(i)
    for i in hdulist[1].data['i_aper'][total_m]: 
        mag[107].append(i)
    for i in hdulist[1].data['ei_aper'][total_m]:
        mag[108].append(i)
    for i in hdulist[1].data['s2n_i_aper'][total_m]:
        mag[109].append(i)
    for i in hdulist[1].data['F861_auto'][total_m]:
        mag[110].append(i)
    for i in hdulist[1].data['eF861_auto'][total_m]:
        mag[111].append(i)
    for i in hdulist[1].data['s2n_F861_auto'][total_m]:
        mag[112].append(i)
    for i in hdulist[1].data['F861_petro'][total_m]:
        mag[113].append(i)
    for i in hdulist[1].data['eF861_petro'][total_m]:
        mag[114].append(i)
    for i in hdulist[1].data['s2n_F861_petro'][total_m]:
        mag[115].append(i)
    for i in hdulist[1].data['F861_aper'][total_m]:
        mag[116].append(i)
    for i in hdulist[1].data['eF861_aper'][total_m]:
        mag[117].append(i)
    for i in hdulist[1].data['s2n_F861_aper'][total_m]:
        mag[118].append(i)
    for i in hdulist[1].data['z_auto'][total_m]:
        mag[119].append(i)
    for i in hdulist[1].data['ez_auto'][total_m]:
        mag[120].append(i)
    for i in hdulist[1].data['s2n_z_auto'][total_m]:
        mag[121].append(i)
    for i in hdulist[1].data['z_petro'][total_m]: 
        mag[122].append(i)
    for i in hdulist[1].data['ez_petro'][total_m]:
        mag[123].append(i)
    for i in hdulist[1].data['s2n_z_petro'][total_m]:
        mag[124].append(i)
    for i in hdulist[1].data['z_aper'][total_m]:
        mag[125].append(i)
    for i in hdulist[1].data['ez_aper'][total_m]:
        mag[126].append(i)
    for i in hdulist[1].data['s2n_z_aper'][total_m]:
        mag[127].append(i)
    table = Table([mag[0], mag[1], mag[2], mag[3], mag[4], mag[5], mag[6], mag[7], mag[8], mag[9], mag[10], mag[11], mag[12], mag[13], mag[14], mag[15], mag[16], mag[17], mag[18], mag[19], mag[20], mag[21], mag[22], mag[23], mag[24], mag[25], mag[26], mag[27], mag[28], mag[29], mag[30], mag[31], mag[32], mag[33], mag[34], mag[35], mag[36], mag[37], mag[38], mag[39], mag[40], mag[41], mag[42], mag[43], mag[44], mag[45], mag[46], mag[47], mag[48], mag[49], mag[50], mag[51], mag[52], mag[53], mag[54], mag[55], mag[56], mag[57], mag[58], mag[59], mag[60], mag[61], mag[62], mag[63], mag[64], mag[65], mag[66], mag[67], mag[68], mag[69], mag[70], mag[71], mag[72], mag[73], mag[74], mag[75], mag[76], mag[77], mag[78], mag[79], mag[80], mag[81], mag[82], mag[83], mag[84], mag[85], mag[86], mag[87], mag[88], mag[89], mag[90], mag[91], mag[92], mag[93], mag[94], mag[95], mag[96], mag[97], mag[98], mag[99], mag[100], mag[101], mag[102], mag[103], mag[104], mag[105], mag[106], mag[107], mag[108], mag[109], mag[110], mag[111], mag[112], mag[113], mag[114], mag[115], mag[116], mag[117], mag[118], mag[119], mag[120], mag[121], mag[122], mag[123], mag[124], mag[125], mag[126], mag[127]],  names=('Field', 'ID', 'RA', 'Dec', 'X', 'Y', 'Aperture', 's2nDet', 'PhotoFlag', 'FWHM', 'FWHM_n', 'MUMAX', 'A', 'B','THETA','FlRadDet','KrRadDet', 'nDet_auto', 'nDet_petro', 'nDet_aper', 'uJAVA_auto','euJAVA_auto', 's2n_uJAVA_auto', 'uJAVA_petro', 'euJAVA_petro', 's2n_uJAVA_petro ', 'uJAVA_aper', 'euJAVA_aper', 'S2n_uJAVA_aper', 'F0378_auto', 'eF0378_auto', 's2n_F0378_auto', 'F0378_petro', 'eF0378_petro', 's2n_F0378_petro', 'F0378_aper', 'eF0378_aper', 's2n_F0378_aper', 'F0395_auto','eF0395_auto', 's2n_F0395_auto', 'F0395_petro', 'eF0395_petro','s2n_F0395_petro', 'F0395_aper','eF0395_aper','s2n_F0395_aper','F0410_auto','eF0410_auto', 's2n_F0410_auto', 'F0410_petro', 'eF0410_petro', 's2n_F0410_petro', 'F0410_aper', 'eF0410_aper', 's2n_F0410_aper', 'F0430_auto', 'eF0430_auto','s2n_F0430_auto','F0430_petro','eF0430_petro', 's2n_F0430_petro', 'F0430_aper', 'eF0430_aper', 's2n_F0430_aper','g_auto','eg_auto','s2n_g_auto', 'g_petro', 'eg_petro', 's2n_g_petro', 'g_aper', 'eg_aper', 's2n_g_aper', 'F0515_auto', 'eF0515_auto', 's2n_F0515_auto', 'F0515_petro','eF0515_petro','s2n_F0515_petro','F0515_aper','eF0515_aper', 's2n_F0515_aper','r_auto','er_auto','s2n_r_auto','r_petro','er_petro','s2n_r_petro', 'r_aper', 'er_aper', 's2n_r_aper', 'F0660_auto', 'eF0660_auto', 's2n_F0660_auto', 'F0660_petro', 'eF0660_petro', 's2n_F0660_petro','F0660_aper', 'eF0660_aper', 's2n_F0660_aper', 'i_auto','ei_auto','s2n_i_auto','i_petro', 'ei_petro','s2n_i_petro','i_aper','ei_aper', 's2n_i_aper', 'F0861_auto', 'eF0861_auto','s2n_F0861_auto', 'F0861_petro', 'eF0861_petro', 's2n_F0861_petro', 'F0861_aper', 'eF0861_aper', 's2n_F0861_aper', 'z_auto', 'ez_auto', 's2n_z_auto', 'z_petro','dz_petro', 's2n_z_petro', 'z_aper', 'ez_aper', 's2n_z_aper'), meta={'name': 'first table'})
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
  
table_aper = select('aper', mag_aper)
table_petro = select('petro', mag_petro)
table_auto = select('auto', mag_auto)

# # #Saving resultated table
#######
#aper#
#######
asciifile = fitsfile.replace(".fits", "-aper-PN.tab")
try:
    table_aper.write(asciifile, format='ascii.tab', overwrite=True)
except TypeError:
    table_aper.write(asciifile, format='ascii.tab')

#######
#petro#
#######
asciifile_petro = fitsfile.replace(".fits", "-petro-PN.tab")
try:
    table_petro.write(asciifile_petro, format='ascii.tab', overwrite=True)
except TypeError:
    table_petro.write(asciifile_petro, format='ascii.tab')

######
#auto#
######
asciifile_auto = fitsfile.replace(".fits", "-auto-PN.tab")
try:
    table_auto.write(asciifile_auto, format='ascii.tab', overwrite=True)
except TypeError:
    table_auto.write(asciifile_auto, format='ascii.tab')

