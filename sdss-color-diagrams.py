# -*- coding: utf-8 -*-
'''
Make color-color diagrams for Ramses
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import gaussian_kde
import pandas as pd
from astropy.table import Table
#import StringIO
from sympy import S, symbols
from scipy.optimize import fsolve
import os
from astropy.io import fits
import scipy.stats as st
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#Find the point inteception between two lines     
def findIntersection(m, y, m1, y1, x0):
    x = np.linspace(-10.0, 15.5, 200)
    return fsolve(lambda x : (m*x + y) - (m1*x + y1), x0)

#reading the files .json

pattern = "*-spectros/*-SDSS-magnitude.json"
#file_ = "jacoby_library_stellar/*-RAMSESGMOSS-magnitude.json"
file_list0 = glob.glob(pattern)
#file_list1 = glob.glob(file_)
file_list = file_list0 

pattern1 = "BB_*-spectros/*-SDSS-magnitude.json"
file_list2 = glob.glob(pattern1)

#reddenign vector
def redde_vector(x0, y0, x1, y1, a, b, c, d):
    plt.arrow(x0+a, y0+b, (x1+a)-(x0+a), (y1+b)-(y0+b),  fc="k", ec="k", width=0.01,
              head_width=0.05, head_length=0.12) #head_width=0.05, head_length=0.1)
    plt.text(x0+a+c, y0+b+d, 'A$_\mathrm{V}=2$', va='center', fontsize='x-large')


def filter_mag(e, s, f1, f2, f3, f4):
    '''
    Calculate the colors using any of set of filters
    '''
    col, col0 = [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            filter4 = data[f4]
            diff = filter1 - filter2
            diff0 = filter3 - filter4
            col.append(diff)
            col0.append(diff0)
    return col, col0

def plot_mag(f1, f2, f3, f4):
    x0, y0 = filter_mag("HPNe", "", f1, f2, f3, f4)
    x1, y1 = filter_mag("CV", "", f1, f2, f3, f4)
    x2, y2 = filter_mag("sys", "", f1, f2, f3, f4)
    x3, y3 = filter_mag("YSOs", "", f1, f2, f3, f4)
    x4, y4 = filter_mag("Be", "", f1, f2, f3, f4)
    x5, y5 = filter_mag("galaxy", "", f1, f2, f3, f4)
    x6, y6 = filter_mag("ExtHII", "", f1, f2, f3, f4)
    x7, y7 = filter_mag("STAR", "", f1, f2, f3, f4)
   
    for a, b in zip(x0, y0):
        A1[0].append(a)
        B1[0].append(b)
    for a, b in zip(x1, y1):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x2, y2):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x3, y3):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x4, y4):
        A1[4].append(a)
        B1[4].append(b)
    for a, b in zip(x5, y5):
        A1[5].append(a)
        B1[5].append(b)
    for a, b in zip(x6, y6):
        A1[6].append(a)
        B1[6].append(b)
    for a, b in zip(x7, y7):
        A1[7].append(a)
        B1[7].append(b)

def plot_mag_PN(f1, f2, f3, f4):
    x00, y00 = filter_mag("100", "", f1, f2, f3, f4)
    x11, y11 = filter_mag("300", "", f1, f2, f3, f4)
    x22, y22 = filter_mag("600", "", f1, f2, f3, f4)
    for a, b in zip(x00, y00):
        PN_x.append(a)
        PN_y.append(b)
    for a, b in zip(x11, y11):
        PN_x.append(a)
        PN_y.append(b)
    for a, b in zip(x22, y22):
        PN_x.append(a)
        PN_y.append(b)

label = []

n = 8
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("SLOAN_SDSS_g", "SLOAN_SDSS_r", "SLOAN_SDSS_r", "SLOAN_SDSS_i")

for file_name2 in file_list2:
    with open(file_name2) as f2:
        data = json.load(f2)
        plot_mag_PN("SLOAN_SDSS_g", "SLOAN_SDSS_r", "SLOAN_SDSS_r", "SLOAN_SDSS_i")

##################################################################################
# Galaxy SDSS              #######################################################
##################################################################################
datadir = "SDSS/"
datadir1 = "HIIGalaxies-amanda/"
fitsfile = "Skyserver_CrossID11_16_2020_8_50_57_PM.fits"
#hdulist= fits.open(os.path.join(datadir, fitsfile))

fitsfile_model = "Skyserver_CrossID11_17_2020_6_14_53_PM_modelMag_HIIGalaxies.fits"
hdulist = fits.open(os.path.join(datadir, fitsfile_model))

def sdss(hdulist, filter_1, filter_2, filter_3, filter_4):
    band1 = hdulist[1].data[filter_1]
    band2 = hdulist[1].data[filter_2]
    band3 = hdulist[1].data[filter_3]
    band4 = hdulist[1].data[filter_4]
    return (band1 - band2), (band3 - band4)

# Galaxias
gr, ri = sdss(hdulist, "modelMag_g", "modelMag_r", "modelMag_r", "modelMag_i")

#WD from SDSS
fitsfile1 = "TAP_1_J_apj_167_40_catalog_WD.fits"
hdulist1 = fits.open(os.path.join(datadir, fitsfile1))

gr_wd, ri_wd = sdss(hdulist1, "gmag", "rmag", "rmag", "imag")

#HII galaxy from Amanda sample
fitsfile2 = "Skyserver_CrossID11_21_2020_HIIGalaxies.fits"
hdulist2 = fits.open(os.path.join(datadir1, fitsfile2))
gr_hii, ri_hii = sdss(hdulist2, "modelMag_g", "modelMag_r", "modelMag_r", "modelMag_i")

#################################################################################
AB = np.vstack([PN_x, PN_y])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(PN_x), 'y': np.array(PN_y) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(PN_x)[idx], np.array(PN_y)[idx], z[idx]
################################################################################
#plots
################################################################################
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
#sns.set_style("white")
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-3.7,xmax=3.7)
ax1.set_xlim(-2.0,2.0)
ax1.set_ylim(-2.5,1.0)
#ax1.axis('equal')
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$g - r$', fontsize= 25)
plt.ylabel(r'$r - i$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax1.scatter(x, y, c=z, s=50, alpha=0.4, edgecolor='')
ax1.scatter(A1[0], B1[0], c = sns.xkcd_rgb["aqua"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='HPNe')
ax1.scatter(A1[2], B1[2], c = sns.xkcd_rgb["bright red"], alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt')
ax1.scatter(A1[1], B1[1], c = sns.xkcd_rgb["pale yellow"], alpha=0.8, s=60, marker='o', facecolors='none', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='CV')
ax1.scatter(A1[3], B1[3], c = sns.xkcd_rgb["light salmon"], alpha=0.8, s=110, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='YSOs')
ax1.scatter(A1[4], B1[4], c = sns.xkcd_rgb['azure'], alpha=0.8, s=80, cmap=plt.cm.hot, marker='^', edgecolor='black', zorder=111, label='B[e] stars')
ax1.scatter(A1[6], B1[6], c = sns.xkcd_rgb['light orange'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='HII regions')
ax1.scatter(A1[5], B1[5], c = sns.xkcd_rgb['neon purple'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='BCDs')
ax1.scatter(gr, ri, c = sns.xkcd_rgb['neon purple'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110)
ax1.scatter(gr_wd, ri_wd, c = sns.xkcd_rgb['green'], alpha=0.6, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110, label='WDs')
ax1.scatter(A1[7], B1[7], c = sns.xkcd_rgb['mint green'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='Normal stars')
ax1.scatter(gr_hii, ri_hii, c = sns.xkcd_rgb['cerulean'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110, label='HII galaxies')
#plt.axhline(y=-0.4, c="k", linestyle='-.', zorder=120)

# Region where are located the PNe
result = findIntersection(0.0, -0.7, -0.75, -0.7, 0.0)

x_new = np.linspace(-15.5, result,  200)
x_new2 = np.linspace(result, 10.0, 200)
y = 0*x_new - 0.7
yy = -0.75*x_new2 - 0.7
#Mask
#mask = y >= result_y - 0.5
ax1.plot(x_new, y, color='k', zorder=300, linestyle='-.')
ax1.plot(x_new2, yy , color='k', zorder=300, linestyle='-.')
#ax1.scatter(A1[6], B1[6],  c=sns.xkcd_rgb['greyish'], alpha=0.8, s=70, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=90, label='MS and Giant stars')
#ax1 = sns.kdeplot(A1[6], B1[6],
                 #cmap="Blues", shade=True, shade_lowest=False)

#################################################################

# for label_, x, y in zip(label, B1[0], A1[0]):
#     ax1.annotate(label_, (x, y), alpha=5, size=8,
#                    xytext=(-5.0, -10.0), textcoords='offset points', ha='right', va='bottom',)
# ###################################################################
# bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.5, pad=0.1)
# ax1.annotate("H4-1", (np.array(B1[15]), np.array(A1[15])), alpha=15, size=10.0,
#                    xytext=(-7, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
# ax1.annotate("PNG 135.9+55.9", (np.array(B1[16]), np.array(A1[16])), alpha=15, size=10,
#                    xytext=(90, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150)
# ax1.annotate("DdDm-1", (np.array(B1[17]), np.array(A1[17])), alpha=10, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax1.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax1.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

#Region Halpha Emitters
plt.text(0.1, 0.04, 'CLOUDY modelled PNe',
         transform=ax1.transAxes, fontsize=13.8)
# plt.text(0.75, 0.48, 'CV',
#          transform=ax1.transAxes, fontsize=13.8)

#reddening vector
#redde_vector(-1.26260948181, -0.875335025787, -0.909874053228, -0.814979598636, 0.7, 1.2, -0.2, -0.1) #E=0.7
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, ncol=1, fontsize=11.0, loc='upper left', **lgd_kws)
#ax1.legend(scatterpoints=1, frameon=False, labelspacing=1)
#ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig1-ri_vs_gr.pdf'
save_path = 'SDSS/plots-sdss/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
###################################################################################
# Halpha -HalphaC vs OVi - OVICo###################################################
###################################################################################

n = 8
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("SLOAN_SDSS_u", "SLOAN_SDSS_g", "SLOAN_SDSS_r", "SLOAN_SDSS_i")

#PN model
for file_name2 in file_list2:
    with open(file_name2) as f2:
        data = json.load(f2)
        plot_mag_PN("SLOAN_SDSS_u", "SLOAN_SDSS_g", "SLOAN_SDSS_r", "SLOAN_SDSS_i")

#Galaxies from SDSS
ug, ri = sdss(hdulist, "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i")

#WD SDSS
ug_wd, ri_wd = sdss(hdulist1, "umag", "gmag", "rmag", "imag")

#HII galaxy from Amanda sample
ug_hii, ri_hiii = sdss(hdulist2, "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i")

#################################################################################

AB = np.vstack([PN_x, PN_y])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(PN_x), 'y': np.array(PN_y) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(PN_x)[idx], np.array(PN_y)[idx], z[idx]
        
###############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True}#, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111)
ax2.set_xlim(-2.0,3.0)
ax2.set_ylim(-2.5,1.0)
#ax2.axis('equal')
#ax2.set_ylim(ymin=-0.5,ymax=0.2)
# ax2.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$u - g$', fontsize= 25)
plt.ylabel(r'$r - i$', fontsize= 25)
ax2.scatter(x, y, c=z, s=50, alpha=0.4, edgecolor='')
ax2.scatter(A1[0], B1[0], c = sns.xkcd_rgb["aqua"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='HPNe')
ax2.scatter(A1[2], B1[2], c = sns.xkcd_rgb["bright red"], alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt')
ax2.scatter(A1[1], B1[1], c = sns.xkcd_rgb["pale yellow"], alpha=0.8, s=60, marker='o', facecolors='none', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='CV')
ax2.scatter(A1[3], B1[3], c = sns.xkcd_rgb["light salmon"], alpha=0.8, s=110, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='YSOs')
ax2.scatter(A1[4], B1[4], c = sns.xkcd_rgb['azure'], alpha=0.8, s=80, cmap=plt.cm.hot, marker='^', edgecolor='black', zorder=111, label='B[e] stars')
ax2.scatter(A1[6], B1[6], c = sns.xkcd_rgb['light orange'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='HII regions')
ax2.scatter(A1[5], B1[5], c = sns.xkcd_rgb['neon purple'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='BCDs')
ax2.scatter(ug, ri, c = sns.xkcd_rgb['neon purple'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='BCDs')
ax2.scatter(ug_wd, ri_wd, c = sns.xkcd_rgb['green'], alpha=0.6, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110, label='WDs')
ax2.scatter(A1[7], B1[7], c = sns.xkcd_rgb['mint green'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='Normal stars')
ax2.scatter(ug_hii, ri_hiii, c = sns.xkcd_rgb['cerulean'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110, label='HII galaxies')
# plt.axhline(y=-0.4, c="k", linestyle='-.', zorder=120)
# plt.axvline(x=-0.2, c="k", linestyle=':', zorder=120)

# Region where are located the PNe
result = findIntersection(0.0, -0.7, -0.45, -0.42, 0.0)

x_new = np.linspace(-15.5, result,  200)
x_new2 = np.linspace(result, 10.0, 200)
y = 0*x_new - 0.7
yy = -0.45*x_new2 - 0.42
#Mask
#mask = y >= result_y - 0.5
ax2.plot(x_new, y, color='k', zorder=300, linestyle='-.')
ax2.plot(x_new2, yy , color='k', zorder=300, linestyle='-.')

#ax2.scatter(A1[6], B1[6],  c=sns.xkcd_rgb['greyish'], alpha=0.8, s=70, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=90, label='MS and Giant stars')
#ax2 = sns.kdeplot(A1[6], B1[6],
                 #cmap="Blues", shade=True, shade_lowest=False)

#################################################################

# for label_, x, y in zip(label, B1[0], A1[0]):
#     ax2.annotate(label_, (x, y), alpha=5, size=8,
#                    xytext=(-5.0, -10.0), textcoords='offset points', ha='right', va='bottom',)
# ###################################################################
# bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.5, pad=0.1)
# ax2.annotate("H4-1", (np.array(B1[15]), np.array(A1[15])), alpha=15, size=10.0,
#                    xytext=(-7, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
# ax2.annotate("PNG 135.9+55.9", (np.array(B1[16]), np.array(A1[16])), alpha=15, size=10,
#                    xytext=(90, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150)
# ax2.annotate("DdDm-1", (np.array(B1[17]), np.array(A1[17])), alpha=10, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax2.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax2.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

#Region Halpha Emitters

# plt.text(0.01, 0.8, 'CLOUDY modelled hPNe',
#          transform=ax2.transAxes, fontsize=13.8)
# plt.text(0.75, 0.765, 'CV',
#          transform=ax2.transAxes, fontsize=13.8)

#reddening vector
#redde_vector(-0.109506845474, -0.875335025787, 0.140656153361, -0.814979598636, 0.15, -0.99, 0.1, -0.1) #E=0.7

ax2.minorticks_on()
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=1, fontsize=11.0, loc='lower right', **lgd_kws)
#ax2.grid()
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig2-ri_vs_ug.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()

