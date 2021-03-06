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

pattern = "*-spectros/*-RAMSESGMOSS-magnitude.json"
file_ = "jacoby_library_stellar/*-RAMSESGMOSS-magnitude.json"
file_list0 = glob.glob(pattern)
file_list1 = glob.glob(file_)
file_list = file_list0 + file_list1 

pattern1 = "BB_*-spectros/*-RAMSESGMOSS-magnitude.json"
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
    x0, y0 = filter_mag("CV", "", f1, f2, f3, f4)
    x1, y1 = filter_mag("sys", "", f1, f2, f3, f4)
    x2, y2 = filter_mag("sys-IPHAS", "", f1, f2, f3, f4) 
    x3, y3 = filter_mag("extr-SySt", '', f1, f2, f3, f4)
    x4, y4 = filter_mag("ngc185", "", f1, f2, f3, f4)
    x5, y5 = filter_mag("SySt-ic10", "", f1, f2, f3, f4)
    x6, y6 = filter_mag("YSOs", "", f1, f2, f3, f4)
    x7, y7 = filter_mag("Be", "", f1, f2, f3, f4)
    x8, y8 = filter_mag("OVI", "", f1, f2, f3, f4)
    x9, y9 = filter_mag("Giant", "", f1, f2, f3, f4)
   
    for a, b in zip(x0, y0):
        A1[0].append(a)
        B1[0].append(b)
    for a, b in zip(x1, y1):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x2, y2):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x3, y3):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x4, y4):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x5, y5):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x6, y6):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x7, y7):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x8, y8):
        A1[4].append(a)
        B1[4].append(b)
    for a, b in zip(x9, y9):
        A1[5].append(a)
        B1[5].append(b)

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

n = 6
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("HeII_G0340", "HeIIC_G0341", "Ha_G0336", "HaC_G0337")

for file_name2 in file_list2:
    with open(file_name2) as f2:
        data = json.load(f2)
        plot_mag_PN("HeII_G0340", "HeIIC_G0340", "Ha_G0336", "HaC_G0336")

AB = np.vstack([PN_x, PN_y])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(PN_x), 'y': np.array(PN_y) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(PN_x)[idx], np.array(PN_y)[idx], z[idx]
##############################################################################
#plots
##############################################################################

lgd_kws = {'frameon': True, 'fancybox': True}#, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
#sns.set_style("white")
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(xmin=-3.7,xmax=3.7)
ax1.set_ylim(bottom=-2.9,top=0.9)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$HeII - HeIIC$', fontsize= 25)
plt.ylabel(r'$H{\alpha} - H{\alpha}C$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax1.scatter(x, y, c=z, s=50, alpha=0.4, edgecolor='')
ax1.scatter(A1[4], B1[4],  c= sns.xkcd_rgb["bright red"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt with Raman OVI')
ax1.scatter(A1[1], B1[1], alpha=0.8, s=60, marker='o', facecolors='none', cmap=plt.cm.hot, edgecolor='red', linewidth=2.0, zorder=122.0, label='SySt without Raman OVI')
#ax1.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax1.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
ax1.scatter(A1[0], B1[0], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=50, marker='s', cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='CVs')
# pal = sns.dark_palette("palegreen", as_cmap=True)
# ax1 = sns.kdeplot(A1[0], B1[0], cmap=pal)
#ax1.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax1.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax1.scatter(A1[2], B1[2],  c= "lightsalmon", alpha=0.8, s=110, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='YSOs')
ax1.scatter(A1[3], B1[3],  c=sns.xkcd_rgb['azure'], alpha=0.8, s=80, cmap=plt.cm.hot, marker='^', edgecolor='black', zorder=111, label='B[e] stars')
ax1.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['cyan'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='RGB stars')
plt.axhline(y=-0.4, c="k", linestyle='-.', zorder=120)

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
redde_vector(-1.26260948181, -0.875335025787, -0.909874053228, -0.814979598636, 0.7, 1.2, -0.2, -0.1) #E=0.7

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
pltfile = 'Fig1-Ha_vs_HeII_V2.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
##################################################################################3
# Halpha -HalphaC vs OVi - OVICo##################################################
###################################################################################

n = 6
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("GMOS-S_6835A_B004", "GMOS-S_6780A_B006", "Ha_G0336", "HaC_G0337")

#PN model
for file_name2 in file_list2:
    with open(file_name2) as f2:
        data = json.load(f2)
        plot_mag_PN("GMOS-S_6835A_B004", "GMOS-S_6780A_B006", "Ha_G0336", "HaC_G0336")

AB = np.vstack([PN_x, PN_y])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(PN_x), 'y': np.array(PN_y) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(PN_x)[idx], np.array(PN_y)[idx], z[idx]
        
##############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True}#, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111)
ax2.set_ylim(bottom=-2.9,top=0.9)
#ax2.set_ylim(ymin=-0.5,ymax=0.2)
# ax2.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$OVI - OVIC$', fontsize= 25)
plt.ylabel(r'$H{\alpha} - H{\alpha}C$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax2.scatter(x, y, c=z, s=50, alpha=0.4, edgecolor='')
ax2.scatter(A1[4], B1[4],  c= sns.xkcd_rgb["bright red"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt with Raman OVI')
ax2.scatter(A1[1], B1[1], alpha=0.8, s=60, marker='o', facecolors='none', cmap=plt.cm.hot, edgecolor='red', linewidth=2.0, zorder=122.0, label='SySt without Raman OVI')
#ax1.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax1.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
ax2.scatter(A1[0], B1[0], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=50, marker='s', cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='CV')
# pal = sns.dark_palette("palegreen", as_cmap=True)
# ax1 = sns.kdeplot(A1[0], B1[0], cmap=pal)
#ax1.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax1.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax2.scatter(A1[2], B1[2],  c= "lightsalmon", alpha=0.8, s=110, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='YSO')
ax2.scatter(A1[3], B1[3],  c=sns.xkcd_rgb['azure'], alpha=0.8, s=80, cmap=plt.cm.hot, marker='^', edgecolor='black', zorder=111, label='B[e] stars')
ax2.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['cyan'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='Giant stars')
plt.axhline(y=-0.4, c="k", linestyle='-.', zorder=120)
plt.axvline(x=-0.2, c="k", linestyle=':', zorder=120)
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
redde_vector(-0.109506845474, -0.875335025787, 0.140656153361, -0.814979598636, 0.15, -0.99, 0.1, -0.1) #E=0.7

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
pltfile = 'Fig2-Ha_vs_OVI_v2.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()

###########################################################
# HeII - HeIIC vs OVI - OVIC ##############################
############################################################
n = 8
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("GMOS-S_6835A_B004", "GMOS-S_6780A_B006", "HeII_G0340", "HeIIC_G0341")

#PN model
for file_name2 in file_list2:
    with open(file_name2) as f2:
        data = json.load(f2)
        plot_mag_PN("GMOS-S_6835A_B004", "GMOS-S_6780A_B006", "HeII_G0340", "HeIIC_G0340")


for ii, iii in zip(file_list2, PN_x):
    if iii < -0.2:
        print(ii, iii)

AB = np.vstack([PN_x, PN_y])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(PN_x), 'y': np.array(PN_y) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(PN_x)[idx], np.array(PN_y)[idx], z[idx]

##############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True}#, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111)
# ax2.set_xlim(xmin=-3.7,xmax=3.7)
#ax3.set_ylim(ymin=-0.5,ymax=0.2)
#ax3.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$OVI - OVIC$', fontsize= 25)
plt.ylabel(r'$HeII - HeIIC$', fontsize= 25)
ax3.scatter(x, y, c=z, s=50, alpha=0.4, edgecolor='')
ax3.scatter(A1[4], B1[4],  c= sns.xkcd_rgb["bright red"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt with Raman OVI')
ax3.scatter(A1[1], B1[1], alpha=0.8, s=60, marker='o', facecolors='none', cmap=plt.cm.hot, edgecolor='red', linewidth=2.0, zorder=122.0, label='SySt without Raman OVI')
#ax1.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax1.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
ax3.scatter(A1[0], B1[0], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=50, marker='s', cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='CVs')
# pal = sns.dark_palette("palegreen", as_cmap=True)
# ax1 = sns.kdeplot(A1[0], B1[0], cmap=pal)
#ax1.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax1.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax3.scatter(A1[2], B1[2],  c= "lightsalmon", alpha=0.8, s=110, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='YSOs')
ax3.scatter(A1[3], B1[3],  c=sns.xkcd_rgb['azure'], alpha=0.8, s=80, cmap=plt.cm.hot, marker='^', edgecolor='black', zorder=111, label='B[e] stars')
ax3.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['cyan'], alpha=0.8, s=50, cmap=plt.cm.hot, marker='D',  edgecolor='black', zorder=110, label='RGB stars')
plt.axvline(x=-0.2, c="k", linestyle=':', zorder=120)
#ax3.scatter(A1[6], B1[6],  c=sns.xkcd_rgb['greyish'], alpha=0.8, s=70, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=90, label='MS and Giant stars')
#ax3 = sns.kdeplot(A1[6], B1[6],
                 #cmap="Blues", shade=True, shade_lowest=False)

#################################################################

# for label_, x, y in zip(label, B1[0], A1[0]):
#     ax3.annotate(label_, (x, y), alpha=5, size=8,
#                    xytext=(-5.0, -10.0), textcoords='offset points', ha='right', va='bottom',)
# ###################################################################
# bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.5, pad=0.1)
# ax3.annotate("H4-1", (np.array(B1[15]), np.array(A1[15])), alpha=15, size=10.0,
#                    xytext=(-7, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
# ax3.annotate("PNG 135.9+55.9", (np.array(B1[16]), np.array(A1[16])), alpha=15, size=10,
#                    xytext=(90, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150)
# ax3.annotate("DdDm-1", (np.array(B1[17]), np.array(A1[17])), alpha=10, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax3.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax3.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

#Region Halpha Emitters


# plt.text(0.01, 0.8, 'CLOUDY modelled hPNe',
#          transform=ax3.transAxes, fontsize=13.8)
# plt.text(0.7, 0.7, 'CV',
#          transform=ax3.transAxes, fontsize=13.8)

#reddening vector
redde_vector(-0.109506845474, -1.26260948181, 0.140656153361, -0.909874053228, 0.18, -0.99, 0.1, 0.0) #E=0.7
ax3.minorticks_on()
#ax3.grid(which='minor')#, lw=0.3)
#ax3.legend(scatterpoints=1, ncol=2, fontsize=11.0, loc='lower right', **lgd_kws)
#ax3.grid()
#lgd = ax3.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax3.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig3-HeII_vs_OVI_v2.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
