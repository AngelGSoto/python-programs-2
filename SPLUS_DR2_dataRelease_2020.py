'''
Getting the S-PLUS catalog (DR2)
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from astropy.table import Table
import argparse

n=129
mag = [[] for _ in range(n)]

# pattern = "*.cat"
# file_list = glob.glob(pattern)

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default=" teste-program",
                    help="Name of catalog, taken the prefix ")


#file_list = "teste-program.cat"

dt = np.dtype([('FIELD', 'S25'), ('ID', 'S25'), ('RA', 'f4'), ('Dec', 'f4'),('X', 'f4'),('Y', 'f4'),('Aperture', 'f4'),('s2nDet', 'f4'),('PhotoFlag', 'f4'), ('FWHM', 'f4'), ('FWHM_n', 'f4'),  ('MUMAX', 'f4'),('A', 'f4'),('B', 'f4'),('THETA', 'f4'),('FlRadDet', 'f4'),('KrRadDet', 'f4'), ('nDet_auto', 'f4'), ('nDet_petro', 'f4'), ('nDet_aper', 'f4'), ('uJAVA_auto', 'f4'),('euJAVA_auto', 'f4'), ('s2n_uJAVA_auto', 'f4'), ('uJAVA_petro', 'f4'), ('euJAVA_petro', 'f4'), ('s2n_uJAVA_petro', 'f4'), ('uJAVA_aper', 'f4'), ('euJAVA_aper', 'f4'), ('S2n_uJAVA_aper', 'f4'), ('F0378_auto', 'f4'),('eF0378_auto', 'f4'),('s2n_F0378_auto', 'f4'),('F0378_petro', 'f4'),('eF0378_petro', 'f4'),('s2n_F0378_petro', 'f4'),('F0378_aper', 'f4'),('eF0378_aper', 'f4'),('s2n_F0378_aper', 'f4'),('F0395_auto', 'f4'),('eF0395_auto', 'f4'),('s2n_F0395_auto', 'f4'),('F0395_petro', 'f4'),('eF0395_petro', 'f4'),('s2n_F0395_petro', 'f4'),('F0395_aper', 'f4'),('eF0395_aper', 'f4'),('s2n_F0395_aper', 'f4'),('F0410_auto', 'f4'),('eF0410_auto', 'f4'),('s2n_F0410_auto', 'f4'),('F0410_petro', 'f4'),('eF0410_petro', 'f4'),('s2n_F0410_petro', 'f4'),('F0410_aper', 'f4'),('eF0410_aper', 'f4'),('s2n_F0410_aper', 'f4'),('F0430_auto', 'f4'),('eF0430_auto', 'f4'),('s2n_F0430_auto', 'f4'),('F0430_petro', 'f4'),('eF0430_petro', 'f4'),('s2n_F0430_petro', 'f4'),('F0430_aper', 'f4'),('eF0430_aper', 'f4'),('s2n_F0430_aper', 'f4'),('G_auto', 'f4'),('dG_auto', 'f4'),('s2n_G_auto', 'f4'),('G_petro', 'f4'),('dG_petro', 'f4'),('s2n_G_petro', 'f4'),('G_aper', 'f4'),('dG_aper', 'f4'),('s2n_G_aper', 'f4'),('F0515_auto', 'f4'),('eF0515_auto', 'f4'),('s2n_F0515_auto', 'f4'),('F0515_petro', 'f4'),('eF0515_petro', 'f4'),('s2n_F0515_petro', 'f4'),('F0515_aper', 'f4'),('eF0515_aper', 'f4'),('s2n_F0515_aper', 'f4'),('R_auto', 'f4'),('dR_auto', 'f4'),('s2n_R_auto', 'f4'),('R_petro', 'f4'),('dR_petro', 'f4'),('s2n_R_petro', 'f4'),('R_aper', 'f4'),('dR_aper', 'f4'),('s2n_R_aper', 'f4'),('F0660_auto', 'f4'),('eF0660_auto', 'f4'),('s2n_F0660_auto', 'f4'),('F0660_petro', 'f4'),('eF0660_petro', 'f4'),('s2n_F0660_petro', 'f4'),('F0660_aper', 'f4'),('eF0660_aper', 'f4'),('s2n_F0660_aper', 'f4'),('I_auto', 'f4'),('dI_auto', 'f4'),('s2n_I_auto', 'f4'),('I_petro', 'f4'),('dI_petro', 'f4'),('s2n_I_petro', 'f4'),('I_aper', 'f4'),('dI_aper', 'f4'),('s2n_I_aper', 'f4'),('F0861_auto', 'f4'),('eF0861_auto', 'f4'),('s2n_F0861_auto', 'f4'),('F0861_petro', 'f4'),('eF0861_petro', 'f4'),('s2n_F0861_petro', 'f4'),('F0861_aper', 'f4'),('eF0861_aper', 'f4'),('s2n_F0861_aper', 'f4'),('Z_auto', 'f4'),('dZ_auto', 'f4'),('s2n_Z_auto', 'f4'),('Z_petro', 'f4'), ('dZ_petro', 'f4'),('s2n_Z_petro', 'f4'),('Z_aper', 'f4'),('dZ_aper', 'f4'),('s2n_Z_aper', 'f4')])

cmd_args = parser.parse_args()
file_ = cmd_args.source + ".cat"

data = np.loadtxt(file_, dtype=dt)

for i in data['FIELD']:
    mag[0].append(i)
for i in data['ID']:
    mag[1].append(i)
for i in data['RA']:
    mag[2].append(i)
for i in data['Dec']:
    mag[3].append(i)
for i in data['X']:
    mag[4].append(i)
for i in data['Y']:
    mag[5].append(i)
for i in data['Aperture']:
    mag[6].append(i)
for i in data['s2nDet']:
    mag[7].append(i)
for i in data['PhotoFlag']:
    mag[8].append(i)
for i in data['FWHM']:
    mag[9].append(i)
for i in data['FWHM_n']:
    mag[10].append(i)
for i in data['MUMAX']:
    mag[11].append(i)
for i in data['A']:
    mag[12].append(i)
for i in data['B']:
    mag[13].append(i)
for i in data['THETA']:
    mag[14].append(i)
for i in data['FlRadDet']:
    mag[15].append(i)
for i in data['KrRadDet']:
    mag[16].append(i)
for i in data['nDet_auto']:
    mag[17].append(i)
for i in data['nDet_petro']:
    mag[18].append(i)
for i in data['nDet_aper']:         
    mag[19].append(i)
for i in data['uJAVA_auto']:
    mag[20].append(i)
for i in data['euJAVA_auto']:
    mag[21].append(i)
for i in data['s2n_uJAVA_auto']:
    mag[22].append(i)
for i in data['uJAVA_petro']:
    mag[23].append(i)
for i in data['euJAVA_petro']:
    mag[24].append(i)
for i in data['s2n_uJAVA_petro']:
    mag[25].append(i)
for i in data['uJAVA_aper']:
    mag[26].append(i)
for i in data['euJAVA_aper']:
    mag[27].append(i)
for i in data['S2n_uJAVA_aper']:
    mag[28].append(i)
for i in data['F0378_auto']:
    mag[29].append(i)
for i in data['eF0378_auto']:
    mag[30].append(i)
for i in data['s2n_F0378_auto']:
    mag[31].append(i)
for i in data['F0378_petro']:
    mag[32].append(i)
for i in data['eF0378_petro']:
    mag[33].append(i)
for i in data['s2n_F0378_petro']:
    mag[34].append(i)
for i in data['F0378_aper']:
    mag[35].append(i)
for i in data['eF0378_aper']:
    mag[36].append(i)
for i in data['s2n_F0378_aper']:
    mag[37].append(i)
for i in data['F0395_auto']:
    mag[38].append(i)
for i in data['eF0395_auto']:
    mag[39].append(i)
for i in data['s2n_F0395_auto']:
    mag[40].append(i)
for i in data['F0395_petro']:
    mag[41].append(i)
for i in data['eF0395_petro']:
    mag[42].append(i)
for i in data['s2n_F0395_petro']:
    mag[43].append(i)
for i in data['F0395_aper']:
    mag[44].append(i)
for i in data['eF0395_aper']:
    mag[45].append(i)
for i in data['s2n_F0395_aper']:
    mag[46].append(i)
for i in data['F0410_auto']:
    mag[47].append(i)
for i in data['eF0410_auto']:
    mag[48].append(i)
for i in data['s2n_F0410_auto']:
    mag[49].append(i)
for i in data['F0410_petro']:
    mag[50].append(i)
for i in data['eF0410_petro']:
    mag[51].append(i)
for i in data['s2n_F0410_petro']:
    mag[52].append(i)
for i in data['F0410_aper']: 
    mag[53].append(i)
for i in data['eF0410_aper']:
    mag[54].append(i)
for i in data['s2n_F0410_aper']:
    mag[55].append(i)
for i in data['F0430_auto']:
    mag[56].append(i)
for i in data['eF0430_auto']:
    mag[57].append(i)
for i in data['s2n_F0430_auto']:
    mag[58].append(i)
for i in data['F0430_petro']:
    mag[59].append(i)
for i in data['eF0430_petro']:
    mag[60].append(i)
for i in data['s2n_F0430_petro']:
    mag[61].append(i)
for i in data['F0430_aper']:
    mag[62].append(i)
for i in data['eF0430_aper']:
    mag[63].append(i)
for i in data['s2n_F0430_aper']:
    mag[64].append(i)
for i in data['G_auto']:
    mag[65].append(i)
for i in data['dG_auto']:
    mag[66].append(i)
for i in data['s2n_G_auto']:
    mag[67].append(i)
for i in data['G_petro']:
    mag[68].append(i)
for i in data['dG_petro']:
    mag[69].append(i)
for i in data['s2n_G_petro']:
    mag[70].append(i)
for i in data['G_aper']:
    mag[71].append(i)
for i in data['dG_aper']:
    mag[72].append(i)
for i in data['s2n_G_aper']:
    mag[73].append(i)
for i in data['F0515_auto']:
    mag[74].append(i)
for i in data['eF0515_auto']:
    mag[75].append(i)
for i in data['s2n_F0515_auto']:
    mag[76].append(i)
for i in data['F0515_petro']:
    mag[77].append(i)
for i in data['eF0515_petro']:
    mag[78].append(i)
for i in data['s2n_F0515_petro']:
    mag[79].append(i)
for i in data['F0515_aper']: 
    mag[80].append(i)
for i in data['eF0515_aper']:
    mag[81].append(i)
for i in data['s2n_F0515_aper']:
    mag[82].append(i)
for i in data['R_auto']:
    mag[83].append(i)
for i in data['dR_auto']:
    mag[84].append(i)
for i in data['s2n_R_auto']:
    mag[85].append(i)
for i in data['R_petro']:
    mag[86].append(i)
for i in data['dR_petro']:
    mag[87].append(i)
for i in data['s2n_R_petro']:
    mag[88].append(i)
for i in data['R_aper']:
    mag[89].append(i)
for i in data['dR_aper']:
    mag[90].append(i)
for i in data['s2n_R_aper']:
    mag[91].append(i)
for i in data['F0660_auto']:
    mag[92].append(i)
for i in data['eF0660_auto']:
    mag[93].append(i)
for i in data['s2n_F0660_auto']:
    mag[94].append(i)
for i in data['F0660_petro']:
    mag[95].append(i)
for i in data['eF0660_petro']:
    mag[96].append(i)
for i in data['s2n_F0660_petro']:
    mag[97].append(i)
for i in data['F0660_aper']:
    mag[98].append(i)
for i in data['eF0660_aper']:
    mag[99].append(i)
for i in data['s2n_F0660_aper']:
    mag[100].append(i)
for i in data['I_auto']:
    mag[101].append(i)
for i in data['dI_auto']:
    mag[102].append(i)
for i in data['s2n_I_auto']:
    mag[103].append(i)
for i in data['I_petro']:
    mag[104].append(i)
for i in data['dI_petro']:
    mag[105].append(i)
for i in data['s2n_I_petro']:
    mag[106].append(i)
for i in data['I_aper']: 
    mag[107].append(i)
for i in data['dI_aper']:
    mag[108].append(i)
for i in data['s2n_I_aper']:
    mag[109].append(i)
for i in data['F0861_auto']:
    mag[110].append(i)
for i in data['eF0861_auto']:
    mag[111].append(i)
for i in data['s2n_F0861_auto']:
    mag[112].append(i)
for i in data['F0861_petro']:
    mag[113].append(i)
for i in data['eF0861_petro']:
    mag[114].append(i)
for i in data['s2n_F0861_petro']:
    mag[115].append(i)
for i in data['F0861_aper']:
    mag[116].append(i)
for i in data['eF0861_aper']:
    mag[117].append(i)
for i in data['s2n_F0861_aper']:
    mag[118].append(i)
for i in data['Z_auto']:
    mag[119].append(i)
for i in data['dZ_auto']:
    mag[120].append(i)
for i in data['s2n_Z_auto']:
    mag[121].append(i)
for i in data['Z_petro']: 
    mag[122].append(i)
for i in data['dZ_petro']:
    mag[123].append(i)
for i in data['s2n_Z_petro']:
    mag[124].append(i)
for i in data['Z_aper']:
    mag[125].append(i)
for i in data['dZ_aper']:
    mag[126].append(i)
for i in data['s2n_Z_aper']:
    mag[127].append(i)
  
table = Table([mag[0], mag[1], mag[2], mag[3], mag[4], mag[5], mag[6], mag[7], mag[8], mag[9], mag[10], mag[11], mag[12], mag[13], mag[14], mag[15], mag[16], mag[17], mag[18], mag[19], mag[20], mag[21], mag[22], mag[23], mag[24], mag[25], mag[26], mag[27], mag[28], mag[29], mag[30], mag[31], mag[32], mag[33], mag[34], mag[35], mag[36], mag[37], mag[38], mag[39], mag[40], mag[41], mag[42], mag[43], mag[44], mag[45], mag[46], mag[47], mag[48], mag[49], mag[50], mag[51], mag[52], mag[53], mag[54], mag[55], mag[56], mag[57], mag[58], mag[59], mag[60], mag[61], mag[62], mag[63], mag[64], mag[65], mag[66], mag[67], mag[68], mag[69], mag[70], mag[71], mag[72], mag[73], mag[74], mag[75], mag[76], mag[77], mag[78], mag[79], mag[80], mag[81], mag[82], mag[83], mag[84], mag[85], mag[86], mag[87], mag[88], mag[89], mag[90], mag[91], mag[92], mag[93], mag[94], mag[95], mag[96], mag[97], mag[98], mag[99], mag[100], mag[101], mag[102], mag[103], mag[104], mag[105], mag[106], mag[107], mag[108], mag[109], mag[110], mag[111], mag[112], mag[113], mag[114], mag[115], mag[116], mag[117], mag[118], mag[119], mag[120], mag[121], mag[122], mag[123], mag[124], mag[125], mag[126], mag[127]],  names=('Field', 'ID', 'RA', 'Dec', 'X', 'Y', 'Aperture', 's2nDet', 'PhotoFlag', 'FWHM', 'FWHM_n', 'MUMAX', 'A', 'B','THETA','FlRadDet','KrRadDet', 'nDet_auto', 'nDet_petro', 'nDet_aper', 'uJAVA_auto','euJAVA_auto', 's2n_uJAVA_auto', 'uJAVA_petro', 'euJAVA_petro', 's2n_uJAVA_petro ', 'uJAVA_aper', 'euJAVA_aper', 'S2n_uJAVA_aper', 'F0378_auto', 'eF0378_auto', 's2n_F0378_auto', 'F0378_petro', 'eF0378_petro', 's2n_F0378_petro', 'F0378_aper', 'eF0378_aper', 's2n_F0378_aper', 'F0395_auto','eF0395_auto', 's2n_F0395_auto', 'F0395_petro', 'eF0395_petro','s2n_F0395_petro', 'F0395_aper','eF0395_aper','s2n_F0395_aper','F0410_auto','eF0410_auto', 's2n_F0410_auto', 'F0410_petro', 'eF0410_petro', 's2n_F0410_petro', 'F0410_aper', 'eF0410_aper', 's2n_F0410_aper', 'F0430_auto', 'eF0430_auto','s2n_F0430_auto','F0430_petro','eF0430_petro', 's2n_F0430_petro', 'F0430_aper', 'eF0430_aper', 's2n_F0430_aper','G_auto','dG_auto','s2n_G_auto', 'G_petro', 'eG_petro', 's2n_G_petro', 'G_aper', 'eG_aper', 's2n_G_aper', 'F0515_auto', 'eF0515_auto', 's2n_F0515_auto', 'F0515_petro','eF0515_petro','s2n_F0515_petro','F0515_aper','eF0515_aper', 's2n_F0515_aper','R_auto','eR_auto','s2n_R_auto','R_petro','eR_petro','s2n_R_petro', 'R_aper', 'eR_aper', 's2n_R_aper', 'F0660_auto', 'eF0660_auto', 's2n_F0660_auto', 'F0660_petro', 'eF0660_petro', 's2n_F0660_petro','F0660_aper', 'eF0660_aper', 's2n_F0660_aper', 'I_auto','eI_auto','s2n_I_auto','I_petro', 'eI_petro','s2n_I_petro','I_aper','dI_aper', 's2n_I_aper', 'F0861_auto', 'eF0861_auto','s2n_F0861_auto', 'F0861_petro', 'eF0861_petro', 's2n_F0861_petro', 'F0861_aper', 'eF0861_aper', 's2n_F0861_aper', 'Z_auto', 'eZ_auto', 's2n_Z_auto', 'Z_petro','dZ_petro', 's2n_Z_petro', 'Z_aper', 'eZ_aper', 's2n_Z_aper'), meta={'name': 'first table'})

#print(table)
# # #Saving resultated table
# # #asciifile = "SPLUS_STRIPE82_Photometry-Datarelease-Junio18.tab"
#asciifile = "teste-program.tab"
asciifile = file_.replace(".cat", 
                  ".tab")
try:
    table.write(asciifile, format='ascii.tab', overwrite=True)
except TypeError:
    table.write(asciifile, format='ascii.tab')
