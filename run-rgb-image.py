'''
Creating command lines to run rgb_image-v2.py
'''
from __future__ import print_function
import glob
from astropy.table import Table
import sys

pattern1 = "MC*/MC*I*swp-crop.fits"
file_list1 = glob.glob(pattern1)

pattern2 = "MC*/MC*R*swp-crop.fits"
file_list2 = glob.glob(pattern2)

pattern3 = "MC*/MC*G*swp-crop.fits"
file_list3 = glob.glob(pattern3)

pattern_region = "MC*/*.reg"
file_list_region = glob.glob(pattern_region)

file_1 = []
file_2 = []
file_3 = []
file_region = []
for a, b, c, d in zip(file_list1, file_list2, file_list3, file_list_region):
    file_1.append(a.split('.fit')[0])
    file_2.append(b.split('.fit')[0])
    file_3.append(c.split('.fit')[0])
    file_region.append(d.split('.re')[0])

file_1.sort()
file_2.sort()
file_3.sort()
file_region.sort()
tab = Table([file_1, file_2, file_3, file_region],  names=('File1', 'File2', 'File3', 'File_region'), meta={'name': 'first table'})
    #table_fig.sort('Auto')
for aa, bb, cc, dd in zip(tab['File1'], tab['File2'], tab['File3'], tab['File_region']):
    file1 = aa
    file2 = bb
    file3 = cc
    file4 = dd
    print("python", "../varios-programas/rgb_image-v2.py", file1, file2, file3, "--position", file4, "--debug")
   
