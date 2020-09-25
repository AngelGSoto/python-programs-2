'''
Read the file from J-PLUS DR2 to make the colour-colour diagrams
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
import argparse

#The equation the represent the criteria
def findIntersection(m, y, m1, y1, x0):
    x = np.linspace(-10.0, 15.5, 200)
    return fsolve(lambda x : (m*x + y) - (m1*x + y1), x0)

def colour(Aper, f1, f2, f3, f4):
    xcolour = Aper[f1] - Aper[f2]
    ycolour = Aper[f3] - Aper[f4]
    return xcolour, ycolour

#Error
def errormag(Aper, ef1, ef2, ef3, ef4):
    excolour = np.sqrt(Aper[ef1]**2 + Aper[ef1]**2)
    eycolour = np.sqrt(Aper[ef3]**2 + Aper[ef4]**2)
    return excolour, eycolour


parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("filesource", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
file_ = cmd_args.filesource + ".tab"

#Creating  dictionary
col_names = ["r-i", "r-ha", "F515 - F861", "'F515 - F660", 
             "z - g", "z - F660", "F660 - r", "g - F515", "g - i", 
             "F410 - F660", "e(r-i)", "e(r-ha)", "e(F515 - F861)", "e(F515 - F660)", 
             "e(z - g)", "e(z - F660)", "e(F660 - r)", "e(g - F515)", "e(g - i)", 
             "e(F410 - F660)"]

nsourse = len(file_)
colours0 = {c: [np.empty((nsourse,))] for c in col_names}

colours_auto = {c: [np.empty((nsourse,))] for c in col_names}

#Read de files
#hdu = fits.open(fitsfile)
#tab_ = hdu[1].data
tab_ = Table.read(file_, format="ascii.tab")

#########################################################################
#Color Aper#############################################################
#########################################################################MAG_APER_6_0
#Color vironen
x1_MAG_APER_aper, y1_MAG_APER_aper = colour(tab_, 'rSDSS_MAG_APER_6_0', 'iSDSS_MAG_APER_6_0', 'rSDSS_MAG_APER_6_0', 'J0660_MAG_APER_6_0')
colours0["r-i"] = x1_MAG_APER_aper
colours0["r-ha"] = y1_MAG_APER_aper
#Color
x2_MAG_APER_aper, y2_MAG_APER_aper = colour(tab_, 'J0515_MAG_APER_6_0', 'J0861_MAG_APER_6_0', 'J0515_MAG_APER_6_0', 'J0660_MAG_APER_6_0')
colours0["F515 - F861"] = x2_MAG_APER_aper
colours0["F515 - F660"] = y2_MAG_APER_aper
#Color
x3_MAG_APER_aper, y3_MAG_APER_aper = colour(tab_, 'zSDSS_MAG_APER_6_0', 'gSDSS_MAG_APER_6_0', 'zSDSS_MAG_APER_6_0', 'J0660_MAG_APER_6_0')
colours0["z - g"] = x3_MAG_APER_aper
colours0["z - F660"] = y3_MAG_APER_aper
#Color
x4_MAG_APER_aper, y4_MAG_APER_aper = colour(tab_, 'J0660_MAG_APER_6_0', 'rSDSS_MAG_APER_6_0', 'gSDSS_MAG_APER_6_0', 'J0515_MAG_APER_6_0')
colours0["F660 - r"] = x4_MAG_APER_aper
colours0["g - F515"] = y4_MAG_APER_aper
#Color
x5_MAG_APER_aper, y5_MAG_APER_aper = colour(tab_, 'gSDSS_MAG_APER_6_0', 'iSDSS_MAG_APER_6_0', 'J0410_MAG_APER_6_0', 'J0660_MAG_APER_6_0')
colours0["g - i"] = x5_MAG_APER_aper
colours0["F410 - F660"] = y5_MAG_APER_aper
#########################################################
#ERROR ##################################################
#########################################################
#Error Color vironen
ex1_MAG_APER_aper, ey1_MAG_APER_aper = errormag(tab_, 'rSDSS_MAG_APER_6_0_err', 'iSDSS_MAG_APER_6_0_err', 'rSDSS_MAG_APER_6_0_err', 'J0660_MAG_APER_6_0_err')
colours0["e(r-i)"] = ex1_MAG_APER_aper
colours0["e(r-ha)"] = ey1_MAG_APER_aper
#Color
ex2_MAG_APER_aper, ey2_MAG_APER_aper = errormag(tab_, 'J0515_MAG_APER_6_0_err', 'J0861_MAG_APER_6_0_err', 'J0515_MAG_APER_6_0_err', 'J0660_MAG_APER_6_0_err')
colours0["e(F515 - F861)"] = ex2_MAG_APER_aper
colours0["e(F515 - F660)"] = ey2_MAG_APER_aper
#Color
ex3_MAG_APER_aper, ey3_MAG_APER_aper = errormag(tab_, 'zSDSS_MAG_APER_6_0_err', 'gSDSS_MAG_APER_6_0_err', 'zSDSS_MAG_APER_6_0_err', 'J0660_MAG_APER_6_0_err')
colours0["e(z - g)"] = ex3_MAG_APER_aper
colours0["e(z - F660)"] = ey3_MAG_APER_aper
#Color
ex4_MAG_APER_aper, ey4_MAG_APER_aper = errormag(tab_, 'J0660_MAG_APER_6_0_err', 'rSDSS_MAG_APER_6_0_err', 'gSDSS_MAG_APER_6_0_err', 'J0515_MAG_APER_6_0_err')
colours0["e(F660 - r)"] = ex4_MAG_APER_aper
colours0["e(g - F515)"] = ey4_MAG_APER_aper
#Color
ex5_MAG_APER_aper, ey5_MAG_APER_aper = errormag(tab_, 'gSDSS_MAG_APER_6_0_err', 'iSDSS_MAG_APER_6_0', 'J0410_MAG_APER_6_0_err', 'J0660_MAG_APER_6_0_err')
colours0["e(g - i)"] = ex5_MAG_APER_aper
colours0["e(F410 - F660)"] = ey5_MAG_APER_aper

#########################################################################
#Color Auto#############################################################
#########################################################################

#Color vironen
x1_MAG_APER_auto, y1_MAG_APER_auto = colour(tab_, 'rSDSS_auto', 'iSDSS_auto', 'rSDSS_auto', 'J0660_auto')
colours_auto["r-i"] = x1_MAG_APER_auto
colours_auto["r-ha"] = y1_MAG_APER_auto
#Color
x2_MAG_APER_auto, y2_MAG_APER_auto = colour(tab_, 'J0515_auto', 'J0861_auto', 'J0515_auto', 'J0660_auto')
colours_auto["F515 - F861"] = x2_MAG_APER_auto
colours_auto["F515 - F660"] = y2_MAG_APER_auto
#Color
x3_MAG_APER_auto, y3_MAG_APER_auto = colour(tab_, 'zSDSS_auto', 'gSDSS_auto', 'zSDSS_auto', 'J0660_auto')
colours_auto["z - g"] = x3_MAG_APER_auto
colours_auto["z - F660"] = y3_MAG_APER_auto
#Color
x4_MAG_APER_auto, y4_MAG_APER_auto = colour(tab_, 'J0660_auto', 'rSDSS_auto', 'gSDSS_auto', 'J0515_auto')
colours_auto["F660 - r"] = x4_MAG_APER_auto
colours_auto["g - F515"] = y4_MAG_APER_auto
#Color
x5_MAG_APER_auto, y5_MAG_APER_auto = colour(tab_, 'gSDSS_auto', 'iSDSS_auto', 'J0410_auto', 'J0660_auto')
colours_auto["g - i"] = x5_MAG_APER_auto
colours_auto["F410 - F660"] = y5_MAG_APER_auto
#########################################################
#ERROR ##################################################
#########################################################
#Error Color vironen
ex1_MAG_APER_auto, ey1_MAG_APER_auto = errormag(tab_, 'rSDSS_auto_err', 'iSDSS_auto_err', 'rSDSS_auto_err', 'J0660_auto_err')
colours_auto["e(r-i)"] = ex1_MAG_APER_auto
colours_auto["e(r-ha)"] = ey1_MAG_APER_auto
#Color
ex2_MAG_APER_auto, ey2_MAG_APER_auto = errormag(tab_, 'J0515_auto_err', 'J0861_auto_err', 'J0515_auto_err', 'J0660_auto_err')
colours_auto["e(F515 - F861)"] = ex2_MAG_APER_auto
colours_auto["e(F515 - F660)"] = ey2_MAG_APER_auto
#Color
ex3_MAG_APER_auto, ey3_MAG_APER_auto = errormag(tab_, 'zSDSS_auto_err', 'gSDSS_auto_err', 'zSDSS_auto_err', 'J0660_auto_err')
colours0["e(z - g)"] = ex3_MAG_APER_auto
colours_auto["e(z - F660)"] = ey3_MAG_APER_auto
#Color
ex4_MAG_APER_auto, ey4_MAG_APER_auto = errormag(tab_, 'J0660_auto_err', 'rSDSS_auto_err', 'gSDSS_auto_err', 'J0515_auto_err')
colours_auto["e(F660 - r)"] = ex4_MAG_APER_auto
colours_auto["e(g - F515)"] = ey4_MAG_APER_auto
#Color
ex5_MAG_APER_auto, ey5_MAG_APER_auto = errormag(tab_, 'gSDSS_auto_err', 'iSDSS_auto', 'J0410_auto_err', 'J0660_auto_err')
colours_auto["e(g - i)"] = ex5_MAG_APER_auto
colours_auto["e(F410 - F660)"] = ey5_MAG_APER_auto

print("The number of objects is:", len(colours0["e(F410 - F660)"]))

# print(colours)
################################
#Definition to make of plots####
################################
current_palette = sns.color_palette()
sns.palplot(current_palette)
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.set_xlim(left=-3.7,right=3.7)
ax.set_ylim(bottom=-2.4,top=2.8)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.xlabel('$r - i$', size =22)
plt.ylabel('$r - J0660$', size =22) #edgecolor='black'
ax.scatter(colours0["r-i"], colours0["r-ha"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, zorder=211.0, label='SMC PNe')
#ax.errorbar(colours0["r-i"], colours0["r-ha"], xerr=colours0["e(r-i)"], yerr=colours0["e(r-ha)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)
    
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
y0 =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3
ax.plot(x_new, y0, color='k', linestyle='-.')
ax.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax.plot(x_new_s, y_s, color='r', linestyle='--')
# ax.plot(x_new2_s, yy_s , color='r', linestyle='--')
plt.text(0.05, 0.92, 'hPN zone',
         transform=ax.transAxes, fontsize=22)
ax.minorticks_on()

# plt.text(0.56, 0.92, 'SySt Zone',
#          transform=ax.transAxes, color="red", fontsize=22)
# ax.minorticks_on()

#ax1.grid(which='minor')#, lw=0.3)
ax.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig1-{}-vironen.pdf'.format(file_.split('.ta')[0])
# save_path = '../../../../../Dropbox/paper-pne/Fig/'
# file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
#############################################################################################

'''
J0515 - J0861 vs J0515 - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(left=-5.8,right=6.4)
ax1.set_ylim(bottom=-4.5,top=5.7)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$J0515 - J0861$', size = 35)
plt.ylabel('$J0515 - J0660$', size = 35)
ax1.scatter(colours0["F515 - F861"], colours0["F515 - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax1.errorbar(colours0["F515 - F861"], colours0["F515 - F660"], xerr=colours0["e(F515 - F861)"], yerr=colours0["e(F515 - F660)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

# Region where are located the PNe
result = findIntersection(2.7, 2.15, 0.0, 0.22058956, 0.0)
result_y = 2.7*result + 2.15

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)
x_new3 = np.linspace(-10.0, result, 200)
y = 2.7*x_new + 2.15
yy = 0.0*x_new2 + 0.22058956

ax1.plot(x_new, y, color='k', linestyle='-.')
ax1.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(5.5, -6.45, 0.98, -0.16, 0.0)
x_new_s = np.linspace(result1, 15.5, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = 5.5*x_new_s - 6.45
yy_s = 0.98*x_new2_s - 0.16

ax1.plot(x_new_s, y_s, color='r', linestyle='--')
ax1.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax1, "", x1_np_MAG_APER_6_0_0, y1_np_MAG_APER_6_0_0, dx=-45)
# source_label(ax1, "LEDA 2790884", x1_np_MAG_APER_6_0_3, y1_np_MAG_APER_6_0_3, dx=8)
# source_label(ax1, "LEDA 101538", x1_np_MAG_APER_6_0_1, y1_np_MAG_APER_6_0_1, dx=-72)
# source_label(ax1, "PN Sp 4-1", x1_np_MAG_APER_6_0_2, y1_np_MAG_APER_6_0_2, dx=-50)
# source_label_hash(ax1, "TK 1", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 6034, dx=4, dy=-10)
# source_label_hash(ax1, "Kn J1857.7+3931", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 3014, dx=-50, dy=13)
# source_label_hash(ax1, "KnPa J1848.6+4151", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 45492, dy=10)
# source_label_hash(ax1, "Jacoby 1", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 5598, dx=-42, dy=6)
# source_label_hash_s(ax1, "Fr 2-21", x1_np_hast_ISO_GAUSS_s, y1_np_hast_ISO_GAUSS_s, dx=-36, dy=-7)

plt.text(0.05, 0.91, 'hPN zone',
         transform=ax1.transAxes, fontsize=22)
# plt.text(0.56, 0.91, 'SySt Zone',
#          transform=ax1.transAxes, color="red", fontsize=22)
# ax1.minorticks_on()

ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax1.grid()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig2-{}-J0515_J0660.pdf'.format(file_.split('.ta')[0])
#save_path = '../../../../../Dropbox/JPAS/paper-phot/'
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)

plt.clf()
###########################################################
#############################################################
'''
z - g vs z - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111)
ax2.set_xlim(left=-5.9,right=3.9)
ax2.set_ylim(bottom=-5.,top=5.0)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$z - g$', size =35)
plt.ylabel('$z - J0660$', size =35)
ax2.scatter(colours0["z - g"], colours0["z - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax2.errorbar(colours0["z - g"], colours0["z - F660"], xerr=colours0["e(z - g)"], yerr=colours0["e(z - F660)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

result = findIntersection(0.2319, 0.85, -1.3, 1.7, 0.0)
result_y = 0.2319*result + 0.85

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)

y = 0.2319*x_new + 0.85
yy = -1.3*x_new2 + 1.7
#Mask
#mask = y >= result_y - 0.5
ax2.plot(x_new, y, color='k', linestyle='-.')
ax2.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars=>
result1 = findIntersection(-1.96, -3.15, 0.2, 0.44, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(-15.5, result1, 200)
y_s = -1.96*x_new_s - 3.15
yy_s = 0.2*x_new2_s + 0.44
# ax2.plot(x_new_s, y_s, color='r', linestyle='--')
# ax2.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax2, "", x2_np_MAG_APER_6_0_0, y2_np_MAG_APER_6_0_0, dx=-42)
# source_label(ax2, "LEDA 2790884", x2_np_MAG_APER_6_0_3, y2_np_MAG_APER_6_0_3, dx=-75, dy=7)
# source_label(ax2, "LEDA 101538", x2_np_MAG_APER_6_0_1, y2_np_MAG_APER_6_0_1, dy=-8)
# source_label(ax2, "PN Sp 4-1", x2_np_MAG_APER_6_0_2, y2_np_MAG_APER_6_0_2, dx=7, dy=-5)
# source_label_hash(ax2, "TK 1", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 6034)
# source_label_hash(ax2, "Kn J1857.7+3931", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 3014, dx=-85, dy=-5)#, dx=-85, dy=5)
# source_label_hash(ax2, "KnPa J1848.6+4151", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 45492, dy=-10)
# source_label_hash(ax2, "Jacoby 1", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 5598, dx=4, dy=-10)#, dx=-45, dy=-5)
# source_label_hash_s(ax2, "Fr 2-21", x2_np_hast_ISO_GAUSS_s, y2_np_hast_ISO_GAUSS_s, dx=-36, dy=7) 

plt.text(0.58, 0.92, 'hPN zone',
         transform=ax2.transAxes, fontsize=22)
# plt.text(0.03, 0.7, 'SySt Zone',
#          transform=ax2.transAxes, color="red", fontsize=22)
# ax2.minorticks_on()

ax2.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax2.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig3-{}-z.pdf'.format(file_.split('.ta')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
plt.clf()
###############################################################
###############################################################
'''
J0660 - r vs g - J0515
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax4 = fig.add_subplot(111)
ax4.set_xlim(left=-2.7,right=0.8)
ax4.set_ylim(bottom=-3.2,top=1.8)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$J0660 - r$', size =35)
plt.ylabel('$g - J0515$', size =35)
ax4.scatter(colours0["F660 - r"], colours0["g - F515"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax4.errorbar(colours0["F660 - r"], colours0["g - F515"], xerr=colours0["e(F660 - r)"], yerr=colours0["e(g - F515)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

# Region where are located the PNe
result = findIntersection(0.12, -0.01, -1.1, -1.07, 0.0)
result_y = 0.12*result - 0.01

x_new = np.linspace(-15.5, result,  200)
x_new2 = np.linspace(result, 10.0, 200)
x_new3 = np.linspace(-10.0, 1.1, 200)
y = 0.12*x_new - 0.01
yy = -1.1*x_new2 - 1.07
#Mask
#mask = y >= result_y - 0.5
ax4.plot(x_new, y, color='k', linestyle='-.')
ax4.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-0.19, -0.05, -2.66, -2.2, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(-15.0, result1, 200)
y_s = -0.19*x_new_s - 0.09
yy_s = -2.66*x_new2_s - 2.2

plt.text(0.05, 0.07, 'hPN zone',
         transform=ax4.transAxes, fontsize=22)
plt.text(0.05, 0.92, 'SySt Zone',
         transform=ax4.transAxes, color="red", fontsize=22)
# ax4.minorticks_on()

ax4.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax4.legend(scatterpoints=1, fontsize=15.0, loc="lower right", **lgd_kws)
#ax4.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig4-{}-g.pdf'.format(file_.split('.tab')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
plt.clf()

####################################################################################
####################################################################################
'''
g - i vs J0410 - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax5 = fig.add_subplot(111)
ax5.set_xlim(left=-3.0,right=5.0)
ax5.set_ylim(bottom=-2.0,top=6.0)

plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$g - i$', size=35)
plt.ylabel('$J0410 - J0660$', size =35)
ax5.scatter(colours0["g - i"], colours0["F410 - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, s=100,  zorder=211.0, label='PN candidate')
#ax5.errorbar(colours0["g - i"], colours0["F410 - F660"], xerr=colours0["e(g - i)"], yerr=colours0["e(F410 - F660)"],  marker='.', fmt='.', color= sns.xkcd_rgb["light orange"], elinewidth=0.9, markeredgewidth=0.9, alpha=0.3, capsize=3)

# Region where are located the PNe
result = findIntersection(8.0, 4.50, 0.8, 0.55, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)
x_new3 = np.linspace(-10.0, 1.1, 200)
y =  8.0*x_new + 4.50
yy = 0.8*x_new2 + 0.55
#Mask
#mask = y >= result_y - 0.5
ax5.plot(x_new, y, color='k', linestyle='-.')
ax5.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-5.2, +10.60, 2.13, -1.43, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -5.2*x_new_s + 10.60
yy_s = 2.13*x_new2_s - 1.43

ax5.plot(x_new_s, y_s, color='r', linestyle='--')
ax5.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax5, "", x5_np_MAG_APER_6_0_0, y5_np_MAG_APER_6_0_0, dy=-4.5)
# source_label(ax5, "LEDA 2790884", x5_np_MAG_APER_6_0_3, y5_np_MAG_APER_6_0_3, dx=10, dy=-4.5)
# source_label(ax5, "LEDA 101538", x5_np_MAG_APER_6_0_1, y5_np_MAG_APER_6_0_1, dx=-65, dy=-4.5)
# source_label(ax5, "PN Sp 4-1", x5_np_MAG_APER_6_0_2, y5_np_MAG_APER_6_0_2, dx= -50, dy=-4.5)
# source_label_hash(ax5, "TK 1", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 6034, dy=-5)
# source_label_hash(ax5, "Kn J1857.7+3931", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 3014)#, dx=-85)
# source_label_hash(ax5, "KnPa J1848.6+4151", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 45492, dy=10)
# source_label_hash(ax5, "Jacoby 1", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 5598, dx=-46, dy=-5)
# source_label_hash_s(ax5, "Fr 2-21", x5_np_hast_ISO_GAUSS_s, y5_np_hast_ISO_GAUSS_s, dx=-36, dy=8)

plt.text(0.03, 0.90, 'hPN zone',
         transform=ax5.transAxes, fontsize=22)

# plt.text(0.5, 0.93, 'SySt Zone',
#          transform=ax5.transAxes,color="red", fontsize=22)

ax5.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax5.legend(scatterpoints=1, fontsize=15.0, loc='lower right', **lgd_kws)
#ax5.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig5-{}-gi.pdf'.format(file_.split('.ta')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)

###########################################
#Auto ##################################
###########################################

urrent_palette = sns.color_palette()
sns.palplot(current_palette)
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.set_xlim(left=-3.7,right=3.7)
ax.set_ylim(bottom=-2.4,top=2.8)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.xlabel('$r - i$', size =22)
plt.ylabel('$r - J0660$', size =22) #edgecolor='black'
ax.scatter(colours_auto["r-i"], colours_auto["r-ha"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, zorder=211.0, label='SMC PNe')
#ax.errorbar(colours_auto["r-i"], colours_auto["r-ha"], xerr=colours_auto["e(r-i)"], yerr=colours_auto["e(r-ha)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)
    
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
y0 =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3
ax.plot(x_new, y0, color='k', linestyle='-.')
ax.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax.plot(x_new_s, y_s, color='r', linestyle='--')
# ax.plot(x_new2_s, yy_s , color='r', linestyle='--')
plt.text(0.05, 0.92, 'hPN zone',
         transform=ax.transAxes, fontsize=22)
ax.minorticks_on()

# plt.text(0.56, 0.92, 'SySt Zone',
#          transform=ax.transAxes, color="red", fontsize=22)
# ax.minorticks_on()

#ax1.grid(which='minor')#, lw=0.3)
ax.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig1-{}-vironen-auto.pdf'.format(file_.split('.ta')[0])
# save_path = '../../../../../Dropbox/paper-pne/Fig/'
# file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
#############################################################################################

'''
J0515 - J0861 vs J0515 - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(left=-5.8,right=6.4)
ax1.set_ylim(bottom=-4.5,top=5.7)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$J0515 - J0861$', size = 35)
plt.ylabel('$J0515 - J0660$', size = 35)
ax1.scatter(colours_auto["F515 - F861"], colours_auto["F515 - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax1.errorbar(colours_auto["F515 - F861"], colours_auto["F515 - F660"], xerr=colours_auto["e(F515 - F861)"], yerr=colours_auto["e(F515 - F660)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

# Region where are located the PNe
result = findIntersection(2.7, 2.15, 0.0, 0.22058956, 0.0)
result_y = 2.7*result + 2.15

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)
x_new3 = np.linspace(-10.0, result, 200)
y = 2.7*x_new + 2.15
yy = 0.0*x_new2 + 0.22058956

ax1.plot(x_new, y, color='k', linestyle='-.')
ax1.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(5.5, -6.45, 0.98, -0.16, 0.0)
x_new_s = np.linspace(result1, 15.5, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = 5.5*x_new_s - 6.45
yy_s = 0.98*x_new2_s - 0.16

ax1.plot(x_new_s, y_s, color='r', linestyle='--')
ax1.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax1, "", x1_np_MAG_APER_6_0_0, y1_np_MAG_APER_6_0_0, dx=-45)
# source_label(ax1, "LEDA 2790884", x1_np_MAG_APER_6_0_3, y1_np_MAG_APER_6_0_3, dx=8)
# source_label(ax1, "LEDA 101538", x1_np_MAG_APER_6_0_1, y1_np_MAG_APER_6_0_1, dx=-72)
# source_label(ax1, "PN Sp 4-1", x1_np_MAG_APER_6_0_2, y1_np_MAG_APER_6_0_2, dx=-50)
# source_label_hash(ax1, "TK 1", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 6034, dx=4, dy=-10)
# source_label_hash(ax1, "Kn J1857.7+3931", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 3014, dx=-50, dy=13)
# source_label_hash(ax1, "KnPa J1848.6+4151", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 45492, dy=10)
# source_label_hash(ax1, "Jacoby 1", x1_np_hast_MAG_APER_6_0, y1_np_hast_MAG_APER_6_0, 5598, dx=-42, dy=6)
# source_label_hash_s(ax1, "Fr 2-21", x1_np_hast_ISO_GAUSS_s, y1_np_hast_ISO_GAUSS_s, dx=-36, dy=-7)

plt.text(0.05, 0.91, 'hPN zone',
         transform=ax1.transAxes, fontsize=22)
# plt.text(0.56, 0.91, 'SySt Zone',
#          transform=ax1.transAxes, color="red", fontsize=22)
# ax1.minorticks_on()

ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax1.grid()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig2-{}-J0515_J0660-auto.pdf'.format(file_.split('.ta')[0])
#save_path = '../../../../../Dropbox/JPAS/paper-phot/'
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)

plt.clf()
###########################################################
#############################################################
'''
z - g vs z - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111)
ax2.set_xlim(left=-5.9,right=3.9)
ax2.set_ylim(bottom=-5.,top=5.0)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$z - g$', size =35)
plt.ylabel('$z - J0660$', size =35)
ax2.scatter(colours_auto["z - g"], colours_auto["z - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax2.errorbar(colours_auto["z - g"], colours_auto["z - F660"], xerr=colours_auto["e(z - g)"], yerr=colours_auto["e(z - F660)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

result = findIntersection(0.2319, 0.85, -1.3, 1.7, 0.0)
result_y = 0.2319*result + 0.85

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)

y = 0.2319*x_new + 0.85
yy = -1.3*x_new2 + 1.7
#Mask
#mask = y >= result_y - 0.5
ax2.plot(x_new, y, color='k', linestyle='-.')
ax2.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars=>
result1 = findIntersection(-1.96, -3.15, 0.2, 0.44, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(-15.5, result1, 200)
y_s = -1.96*x_new_s - 3.15
yy_s = 0.2*x_new2_s + 0.44
# ax2.plot(x_new_s, y_s, color='r', linestyle='--')
# ax2.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax2, "", x2_np_MAG_APER_6_0_0, y2_np_MAG_APER_6_0_0, dx=-42)
# source_label(ax2, "LEDA 2790884", x2_np_MAG_APER_6_0_3, y2_np_MAG_APER_6_0_3, dx=-75, dy=7)
# source_label(ax2, "LEDA 101538", x2_np_MAG_APER_6_0_1, y2_np_MAG_APER_6_0_1, dy=-8)
# source_label(ax2, "PN Sp 4-1", x2_np_MAG_APER_6_0_2, y2_np_MAG_APER_6_0_2, dx=7, dy=-5)
# source_label_hash(ax2, "TK 1", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 6034)
# source_label_hash(ax2, "Kn J1857.7+3931", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 3014, dx=-85, dy=-5)#, dx=-85, dy=5)
# source_label_hash(ax2, "KnPa J1848.6+4151", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 45492, dy=-10)
# source_label_hash(ax2, "Jacoby 1", x2_np_hast_MAG_APER_6_0, y2_np_hast_MAG_APER_6_0, 5598, dx=4, dy=-10)#, dx=-45, dy=-5)
# source_label_hash_s(ax2, "Fr 2-21", x2_np_hast_ISO_GAUSS_s, y2_np_hast_ISO_GAUSS_s, dx=-36, dy=7) 

plt.text(0.58, 0.92, 'hPN zone',
         transform=ax2.transAxes, fontsize=22)
# plt.text(0.03, 0.7, 'SySt Zone',
#          transform=ax2.transAxes, color="red", fontsize=22)
# ax2.minorticks_on()

ax2.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax2.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig3-{}-z-auto.pdf'.format(file_.split('.ta')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
plt.clf()
###############################################################
###############################################################
'''
J0660 - r vs g - J0515
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax4 = fig.add_subplot(111)
ax4.set_xlim(left=-2.7,right=0.8)
ax4.set_ylim(bottom=-3.2,top=1.8)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$J0660 - r$', size =35)
plt.ylabel('$g - J0515$', size =35)
ax4.scatter(colours_auto["F660 - r"], colours_auto["g - F515"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100,  zorder=211.0, label='PN candidate')
#ax4.errorbar(colours_auto["F660 - r"], colours_auto["g - F515"], xerr=colours_auto["e(F660 - r)"], yerr=colours_auto["e(g - F515)"], marker='.', fmt='.', color= sns.xkcd_rgb["black"], elinewidth=0.9, markeredgewidth=0.9, capsize=3)

# Region where are located the PNe
result = findIntersection(0.12, -0.01, -1.1, -1.07, 0.0)
result_y = 0.12*result - 0.01

x_new = np.linspace(-15.5, result,  200)
x_new2 = np.linspace(result, 10.0, 200)
x_new3 = np.linspace(-10.0, 1.1, 200)
y = 0.12*x_new - 0.01
yy = -1.1*x_new2 - 1.07
#Mask
#mask = y >= result_y - 0.5
ax4.plot(x_new, y, color='k', linestyle='-.')
ax4.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-0.19, -0.05, -2.66, -2.2, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(-15.0, result1, 200)
y_s = -0.19*x_new_s - 0.09
yy_s = -2.66*x_new2_s - 2.2

plt.text(0.05, 0.07, 'hPN zone',
         transform=ax4.transAxes, fontsize=22)
plt.text(0.05, 0.92, 'SySt Zone',
         transform=ax4.transAxes, color="red", fontsize=22)
# ax4.minorticks_on()

ax4.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax4.legend(scatterpoints=1, fontsize=15.0, loc="lower right", **lgd_kws)
#ax4.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig4-{}-g-auto.pdf'.format(file_.split('.ta')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
plt.clf()

####################################################################################
####################################################################################
'''
g - i vs J0410 - J0660
'''
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax5 = fig.add_subplot(111)
ax5.set_xlim(left=-3.0,right=5.0)
ax5.set_ylim(bottom=-2.0,top=6.0)

plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$g - i$', size=35)
plt.ylabel('$J0410 - J0660$', size =35)
ax5.scatter(colours_auto["g - i"], colours_auto["F410 - F660"], c=sns.xkcd_rgb['cerulean'], alpha=0.7, s=100,  zorder=211.0, label='PN candidate')
#ax5.errorbar(colours_auto["g - i"], colours_auto["F410 - F660"], xerr=colours_auto["e(g - i)"], yerr=colours_auto["e(F410 - F660)"],  marker='.', fmt='.', color= sns.xkcd_rgb["light orange"], elinewidth=0.9, markeredgewidth=0.9, alpha=0.3, capsize=3)

# Region where are located the PNe
result = findIntersection(8.0, 4.50, 0.8, 0.55, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)
x_new3 = np.linspace(-10.0, 1.1, 200)
y =  8.0*x_new + 4.50
yy = 0.8*x_new2 + 0.55
#Mask
#mask = y >= result_y - 0.5
ax5.plot(x_new, y, color='k', linestyle='-.')
ax5.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-5.2, +10.60, 2.13, -1.43, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -5.2*x_new_s + 10.60
yy_s = 2.13*x_new2_s - 1.43

ax5.plot(x_new_s, y_s, color='r', linestyle='--')
ax5.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax5, "", x5_np_MAG_APER_6_0_0, y5_np_MAG_APER_6_0_0, dy=-4.5)
# source_label(ax5, "LEDA 2790884", x5_np_MAG_APER_6_0_3, y5_np_MAG_APER_6_0_3, dx=10, dy=-4.5)
# source_label(ax5, "LEDA 101538", x5_np_MAG_APER_6_0_1, y5_np_MAG_APER_6_0_1, dx=-65, dy=-4.5)
# source_label(ax5, "PN Sp 4-1", x5_np_MAG_APER_6_0_2, y5_np_MAG_APER_6_0_2, dx= -50, dy=-4.5)
# source_label_hash(ax5, "TK 1", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 6034, dy=-5)
# source_label_hash(ax5, "Kn J1857.7+3931", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 3014)#, dx=-85)
# source_label_hash(ax5, "KnPa J1848.6+4151", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 45492, dy=10)
# source_label_hash(ax5, "Jacoby 1", x5_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 5598, dx=-46, dy=-5)
# source_label_hash_s(ax5, "Fr 2-21", x5_np_hast_ISO_GAUSS_s, y5_np_hast_ISO_GAUSS_s, dx=-36, dy=8)

plt.text(0.03, 0.90, 'hPN zone',
         transform=ax5.transAxes, fontsize=22)

# plt.text(0.5, 0.93, 'SySt Zone',
#          transform=ax5.transAxes,color="red", fontsize=22)

ax5.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax5.legend(scatterpoints=1, fontsize=15.0, loc='lower right', **lgd_kws)
#ax5.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'figs-pca-v1/Fig5-{}-gi-auto.pdf'.format(file_.split('.ta')[0])
#file_save = os.path.join(save_path, pltfile)
plt.savefig(pltfile)
