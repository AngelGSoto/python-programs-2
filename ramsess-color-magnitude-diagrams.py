# -*- coding: utf-8 -*-
'''
Make color-magnitude diagrams for RAMSSES proyect
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
file_list = glob.glob(pattern)

#reddenign vector
def redde_vector(x0, y0, x1, y1, a, b, c, d):
    plt.arrow(x0+a, y0+b, (x1+a)-(x0+a), (y1+b)-(y0+b),  fc="k", ec="k", width=0.01,
              head_width=0.07, head_length=0.15) #head_width=0.05, head_length=0.1)
    plt.text(x0+a+c, y0+b+d, 'A$_\mathrm{V}=2$', va='center', fontsize='x-large')


def filter_mag(e, s, f1, f2, f3):
    '''
    Calculate the colors using any of set of filters
    '''
    col, col0 = [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            diff = filter1 - filter2
            diff0 = filter3 
            col.append(diff)
            col0.append(diff0)
    
    return col, col0

def plot_mag(f1, f2, f3):
    x, y = filter_mag("HPNe", "", f1, f2, f3)
    x1, y1 = filter_mag("CV", "", f1, f2, f3)
    x2, y2 = filter_mag("tere_E00_300", "", f1, f2, f3)
    x3, y3 = filter_mag("tere_E01_300", "", f1, f2, f3)
    x4, y4 = filter_mag("tere_E02_300", "", f1, f2, f3)
    x5, y5 = filter_mag("tere_E00_600", "", f1, f2, f3)
    x6, y6 = filter_mag("tere_E01_600", "", f1, f2, f3)
    x7, y7 = filter_mag("tere_E02_600", "", f1, f2, f3)
    x8, y8 = filter_mag("sys", "", f1, f2, f3)
    x9, y9 = filter_mag("sys-IPHAS", "", f1, f2, f3) 
    x10, y10 = filter_mag("extr-SySt", '', f1, f2, f3)
    x11, y11 = filter_mag("ngc185", "", f1, f2, f3)
    x12, y12 = filter_mag("SySt-ic10", "", f1, f2, f3)
    x13, y13 = filter_mag("YSOs", "", f1, f2, f3)
    x14, y14 = filter_mag("tere_E00_100", "", f1, f2, f3)
    x15, y15 = filter_mag("tere_E01_100", "", f1, f2, f3)
    x16, y16 = filter_mag("tere_E02_100", "", f1, f2, f3)
    x17, y17 = filter_mag("Be", "", f1, f2, f3)
    x18, y18 = filter_mag("OVI", "", f1, f2, f3)
   
    for a, b in zip(x, y):
        A1[0].append(a)
        B1[0].append(b)
    for a, b in zip(x1, y1):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x2, y2):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x3, y3):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x4, y4):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x5, y5):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x6, y6):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x7, y7):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x8, y8):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x9, y9):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x10, y10):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x11, y11):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x12, y12):
        A1[3].append(a)
        B1[3].append(b)
    for a, b in zip(x13, y13):
        A1[4].append(a)
        B1[4].append(b)
    for a, b in zip(x14, y14):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x15, y15):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x16, y16):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x17, y17):
        A1[5].append(a)
        B1[5].append(b)
    for a, b in zip(x18, y18):
        A1[6].append(a)
        B1[6].append(b)
        
label = []

n = 7
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]


for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("1-HPNe"):
        #     label.append("")
        # elif data['id'].endswith("SLOAN-HPNe-"):
        #     label.append("H4-1")
        # elif data['id'].endswith("1359559-HPNe"):
        #     label.append("PNG 135.9+55.9")
        # if data['id'].startswith("ngc"):
        #     label.append("")
        # elif data['id'].startswith("mwc"):
        #     label.append("")
        plot_mag("Ha_G0336", "HaC_G0336", "Ha_G0336")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

##############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, facecolor="#eeeeee")
# ax1.set_xlim(xmin=-3.7,xmax=3.7)
ax1.set_ylim(ymin=2.5,ymax=24)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$Halpha - Cont(Halpha)$', fontsize= 25)
plt.ylabel(r'$Halpha(mag)$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
#ax1.scatter(x, y, c=z, s=50, alpha=0.5, edgecolor='')
ax1.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt no OVI')
ax1.scatter(A1[6], B1[6],  c= "blue", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt yes OVI')
#ax1.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax1.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
#ax1.scatter(A1[1], B1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=40, cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='SDSS CVs')
pal = sns.dark_palette("palegreen", as_cmap=True)
ax1 = sns.kdeplot(A1[1], B1[1], cmap=pal)
#ax1.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax1.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax1.scatter(A1[4], B1[4],  c= "lightsalmon", alpha=0.8, s=100, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='Obs. YSOs')
ax1.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=90, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')
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


# plt.text(0.01, 0.68, 'CLOUDY modelled hPNe',
#          transform=ax1.transAxes, fontsize=13.8)
plt.text(0.75, 0.78, 'CV',
         transform=ax1.transAxes, fontsize=13.8)

#reddening vector

ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, ncol=1, fontsize=11.0, loc='lower left', **lgd_kws)
#ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig4-Ha_update.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()

##################################################################################3
# Halpha -HalphaC vs OVi - OVICo##################################################
###################################################################################

n = 7
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("1-HPNe"):
        #     label.append("")
        # elif data['id'].endswith("SLOAN-HPNe-"):
        #     label.append("H4-1")
        # elif data['id'].endswith("1359559-HPNe"):
        #     label.append("PNG 135.9+55.9")
        # if data['id'].startswith("ngc"):
        #     label.append("")
        # elif data['id'].startswith("mwc"):
        #     label.append("")
        plot_mag("GMOS-S_6835A_B004", "GMOS-S_6780A_B006", "GMOS-S_6835A_B004")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

##############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(6, 6))
ax2 = fig.add_subplot(111, facecolor="#eeeeee")
# ax2.set_xlim(xmin=-1.58,xmax=0.25)
ax2.set_ylim(ymin=5,ymax=24.5)
# ax2.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$OVI - Cont(OVI)$', fontsize= 25)
plt.ylabel(r'$OVI(mag)$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
#ax2.scatter(x, y, c=z, s=50, alpha=0.5, edgecolor='')
ax2.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt no OVI')
ax2.scatter(A1[6], B1[6],  c= "blue", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt yes OVI')
#ax2.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax2.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
#ax2.scatter(A1[1], B1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=40, cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='SDSS CVs')
pal = sns.dark_palette("palegreen", as_cmap=True)
ax2 = sns.kdeplot(A1[1], B1[1], cmap=pal, zorder=1)
#ax2.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax2.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax2.scatter(A1[4], B1[4],  c= "lightsalmon", alpha=0.8, s=100, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. YSOs')
ax2.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=90, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')
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

ax2.minorticks_on()
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=1, fontsize=11.0, loc='lower left', **lgd_kws)
#ax2.grid()
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig5-OVI-update.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()

###########################################################
# HeII - HeIIC vs OVI - OVIC ##############################
############################################################
n = 7
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]


for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("1-HPNe"):
        #     label.append("")
        # elif data['id'].endswith("SLOAN-HPNe-"):
        #     label.append("H4-1")
        # elif data['id'].endswith("1359559-HPNe"):
        #     label.append("PNG 135.9+55.9")
        # if data['id'].startswith("ngc"):
        #     label.append("")
        # elif data['id'].startswith("mwc"):
        #     label.append("")
        plot_mag("HeII_G0340", "HeIIC_G0340", "HeII_G0340")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(A1[2]), 'y': np.array(B1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

##############################################################################
#plots
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(6, 6))
ax3 = fig.add_subplot(111, facecolor="#eeeeee")
# ax2.set_xlim(xmin=-3.7,xmax=3.7)
ax3.set_ylim(ymin=6,ymax=26.2)
#ax3.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$HeII - Cont(HeII)$', fontsize= 25)
plt.ylabel(r'$HeII(mag)$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
#ax3.scatter(x, y, c=z, s=50, alpha=0.5, edgecolor='')
ax3.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt no OVI')
ax3.scatter(A1[6], B1[6],  c= "blue", alpha=0.6, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='Obs. SySt yes OVI')
#ax3.scatter(A1[0], B1[0], color= sns.xkcd_rgb["aqua"], s=80, cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Obs. hPNe')
#ax3.scatter(B1[19], A1[19], c='y', alpha=0.8, s=40, label='Modeled halo PNe')
#ax3.scatter(A1[1], B1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=40, cmap=plt.cm.hot, edgecolor='black',  zorder=10.0, label='SDSS CVs')
pal = sns.dark_palette("palegreen", as_cmap=True)
ax3 = sns.kdeplot(A1[1], B1[1], cmap=pal)
#ax3.scatter(A1[3], B1[3],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=120.0, label='Obs. SySt')
#ax3.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax3.scatter(A1[4], B1[4],  c= "lightsalmon", alpha=0.8, s=100, cmap=plt.cm.hot, marker='*',  edgecolor='black', label='Obs. YSOs')
ax3.scatter(A1[5], B1[5],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=90, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')
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

ax3.minorticks_on()
#ax3.grid(which='minor')#, lw=0.3)
#ax3.legend(scatterpoints=1, ncol=2, fontsize=11.0, loc='lower right', **lgd_kws)
#ax3.grid()
#lgd = ax3.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax3.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
pltfile = 'Fig6-HeII_update.pdf'
save_path = 'Ramses/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
