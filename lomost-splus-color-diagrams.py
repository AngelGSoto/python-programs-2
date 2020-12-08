# -*- coding: utf-8 -*-
'''
Make color-color diagrams for SDSS/Lamost
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

pattern = "*-lamost-spec/*-Lamost-SPLUS18-magnitude.json"
#file_ = "jacoby_library_stellar/*-RAMSESGMOSS-magnitude.json"
file_list = glob.glob(pattern)
#file_list1 = glob.glob(file_)

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
    x0, y0 = filter_mag("PNe-Lamost", "", f1, f2, f3, f4)
    x1, y1 = filter_mag("SySt-Lamost", "", f1, f2, f3, f4)
    x2, y2 = filter_mag("ELO-Lamost", "", f1, f2, f3, f4)
   
    for a, b in zip(x0, y0):
        A1[0].append(a)
        B1[0].append(b)
    for a, b in zip(x1, y1):
        A1[1].append(a)
        B1[1].append(b)
    for a, b in zip(x2, y2):
        A1[2].append(a)
        B1[2].append(b)

n = 3
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("F625_R", "F766_I", "F625_R", "F660")

for ii, iii, iiii in zip(file_list, A1[2], B1[2]):
    if iiii >= 0.15*iii + 0.53:
        print(ii, iii, iiii)
        

################################################################################
#plots
################################################################################
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
sns.set_style('ticks')
#sns.set_style("white")
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-3.7,xmax=3.7)
ax1.set_xlim(-3.7,3.7)
ax1.set_ylim(-2.4,2.8)
#ax1.axis('equal')
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=25) 
plt.tick_params(axis='y', labelsize=25)
plt.xlabel(r'$r - i$', fontsize= 25)
plt.ylabel(r'$r - F660$', fontsize= 25)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax1.scatter(A1[0], B1[0], c = sns.xkcd_rgb["aqua"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='PNe')
ax1.scatter(A1[1], B1[1], c = sns.xkcd_rgb["red"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=122.0, label='SySt')
ax1.scatter(A1[2], B1[2], c = sns.xkcd_rgb["cerulean"], alpha=0.7, s=80, marker='o', cmap=plt.cm.hot, edgecolor='black', zorder=121.0, label='Emission objects')
#plt.axhline(y=-0.4, c="k", linestyle='-.', zorder=120)

# # Region where are located the PNe
# result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
# result_y = 8.0*result + 4.50

# x_new = np.linspace(-15.0, result, 200)
# x_new2 = np.linspace(-15.0, result, 200)
# y =  0.43*x_new + 0.65
# yy = -6.8*x_new2 - 1.3
# #Mask
# #mask = y >= result_y - 0.5
# #ax1.plot(x_new, y, color='k', zorder=300, linestyle='-.')
# #ax1.plot(x_new2, yy , color='k', zorder=300, linestyle='-.')

# # Region of the simbiotic stars
# result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
# x_new_s = np.linspace(-15.5, result1, 200)
# x_new2_s = np.linspace(result1, 15.5, 200)
# y_s = -220*x_new_s + 40.4
# yy_s = 0.39*x_new2_s + 0.73

# #ax1.plot(x_new_s, y_s, color='r', linestyle='--')
# #ax1.plot(x_new2_s, yy_s , color='r', linestyle='--')

#Region Halpha Emitters
x_new = np.linspace(-15.0, 1000, 200)
#y =  0.108*x_new + 0.5 #no tan criterioso
y = 0.25*x_new + 0.65

ax1.plot(x_new, y, color='k', zorder=100, linestyle='-.')

#Region Halpha Emitters My criterio
x_new_s = np.linspace(-15.0, 1000, 200)
#y =  0.108*x_new + 0.5 #no tan criterioso
ys = 0.15*x_new_s + 0.53

#Mask
#mask = y >= result_y - 0.5
ax1.plot(x_new_s, ys, color='k', zorder=300, linestyle=':')

textbb = {"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
textpars = {'ha': 'center', 'va': 'center', 'bbox': textbb, 'fontsize': 'small'}
plt.text(-2.6, 0.15, 'r - F660 = 0.15*(r - i) + 0.53', rotation=10, rotation_mode='anchor', zorder=301, **textpars)



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
pltfile = 'Fig1-rf660_vs_ri.pdf'
save_path = 'Plots-lamost/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()

###########################################################
#############################################################

n = 3
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("F911_Z", "F480_G", "F911_Z", "F660")

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
plt.ylabel('$z - F660$', size =35)
ax2.scatter(A1[0], B1[0], c=sns.xkcd_rgb['aqua'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax2.scatter(A1[1], B1[1], c=sns.xkcd_rgb['red'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax2.scatter(A1[2], B1[2], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=210.0)

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

# plt.text(0.58, 0.92, 'hPN zone',
#          transform=ax2.transAxes, fontsize=22)
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
pltfile = 'Fig2-zg-zf660.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()
###############################################################
###############################################################
'''
J0660 - r vs g - J0515
'''
n = 3
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("F660", "F625_R", "F480_G", "F515")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111)
ax3.set_xlim(left=-2.7,right=0.8)
ax3.set_ylim(bottom=-3.2,top=1.8)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$F660 - r$', size =35)
plt.ylabel('$g - F515$', size =35)
ax3.scatter(A1[0], B1[0], c=sns.xkcd_rgb['aqua'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax3.scatter(A1[1], B1[1], c=sns.xkcd_rgb['red'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax3.scatter(A1[2], B1[2], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=210.0)

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
ax3.plot(x_new, y, color='k', linestyle='-.')
ax3.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-0.19, -0.05, -2.66, -2.2, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(-15.0, result1, 200)
y_s = -0.19*x_new_s - 0.09
yy_s = -2.66*x_new2_s - 2.2

# plt.text(0.05, 0.07, 'hPN zone',
#          transform=ax3.transAxes, fontsize=22)
# plt.text(0.05, 0.92, 'SySt Zone',
#          transform=ax3.transAxes, color="red", fontsize=22):
# ax3.minorticks_on()

ax3.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax3.legend(scatterpoints=1, fontsize=15.0, loc="lower right", **lgd_kws)
#ax3.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'Fig3-gf515-f660r.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################################
####################################################################################
'''
g - i vs J0410 - J0660
'''
n = 3
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("F480_G", "F766_I", "F410", "F660")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax4 = fig.add_subplot(111)
ax4.set_xlim(left=-3.0,right=5.0)
ax4.set_ylim(bottom=-2.0,top=6.0)

plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xlabel('$g - i$', size=35)
plt.ylabel('$F410 - F660$', size =35)

ax4.scatter(A1[0], B1[0], c=sns.xkcd_rgb['aqua'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax4.scatter(A1[1], B1[1], c=sns.xkcd_rgb['red'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax4.scatter(A1[2], B1[2], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=210.0)

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
ax4.plot(x_new, y, color='k', linestyle='-.')
ax4.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(-5.2, +10.60, 2.13, -1.43, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -5.2*x_new_s + 10.60
yy_s = 2.13*x_new2_s - 1.43

ax4.plot(x_new_s, y_s, color='r', linestyle='--')
ax4.plot(x_new2_s, yy_s , color='r', linestyle='--')

# source_label(ax4, "", x4_np_MAG_APER_6_0_0, y5_np_MAG_APER_6_0_0, dy=-4.5)
# source_label(ax4, "LEDA 2790884", x4_np_MAG_APER_6_0_3, y5_np_MAG_APER_6_0_3, dx=10, dy=-4.5)
# source_label(ax4, "LEDA 101538", x4_np_MAG_APER_6_0_1, y5_np_MAG_APER_6_0_1, dx=-65, dy=-4.5)
# source_label(ax4, "PN Sp 4-1", x4_np_MAG_APER_6_0_2, y5_np_MAG_APER_6_0_2, dx= -50, dy=-4.5)
# source_label_hash(ax4, "TK 1", x4_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 6034, dy=-5)
# source_label_hash(ax4, "Kn J1857.7+3931", x4_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 3014)#, dx=-85)
# source_label_hash(ax4, "KnPa J1848.6+4151", x4_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 45492, dy=10)
# source_label_hash(ax4, "Jacoby 1", x4_np_hast_MAG_APER_6_0, y5_np_hast_MAG_APER_6_0, 5598, dx=-46, dy=-5)
# source_label_hash_s(ax4, "Fr 2-21", x4_np_hast_ISO_GAUSS_s, y5_np_hast_ISO_GAUSS_s, dx=-36, dy=8)

# plt.text(0.03, 0.90, 'hPN zone',
#          transform=ax4.transAxes, fontsize=22)

# plt.text(0.5, 0.93, 'SySt Zone',
#          transform=ax4.transAxes,color="red", fontsize=22)

ax4.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax4.legend(scatterpoints=1, fontsize=15.0, loc='lower right', **lgd_kws)
#ax4.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig4-f410f660-gi.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

###########################################
#F515-F660 vs F515 - F660 ##################################
###########################################
n = 3
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
PN_x, PN_y = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("F515", "F861", "F515", "F660")

current_palette = sns.color_palette()
sns.palplot(current_palette)
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax5 = fig.add_subplot(111)
ax5.set_xlim(left=-5.8,right=6.4)
ax5.set_ylim(bottom=-4.5,top=5.7)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.xlabel('$F515 - F861$', size =22)
plt.ylabel('$F515 - F660$', size =22) #edgecolor='black'
ax5.scatter(A1[0], B1[0], c=sns.xkcd_rgb['aqua'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax5.scatter(A1[1], B1[1], c=sns.xkcd_rgb['red'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=211.0)
ax5.scatter(A1[2], B1[2], c=sns.xkcd_rgb['cerulean'], alpha=0.7, marker ='o', s=100, edgecolor='white', zorder=210.0)
    
# Region where are located the PNe
result = findIntersection(2.7, 2.15, 0.0, 0.22058956, 0.0)
result_y = 2.5*result + 2.15

x_new = np.linspace(result, 15.5, 200)
x_new2 = np.linspace(-10.0, result, 200)
x_new3 = np.linspace(-10.0, 10.0, 200)
y = 2.7*x_new + 2.15
yy = 0.0*x_new2 + 0.22058956
#Mask
#mask = y >= result_y - 0.5
ax5.plot(x_new, y, color='k', linestyle='-.')
ax5.plot(x_new2, yy , color='k', linestyle='-.')

# Region of the simbiotic stars
result1 = findIntersection(5.5, -6.45, 0.98, -0.16, 0.0)
x_new_s = np.linspace(result1, 15.5, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = 5.5*x_new_s - 6.45
yy_s = 0.98*x_new2_s - 0.16

ax5.plot(x_new_s, y_s, color='r', linestyle='--')
ax5.plot(x_new2_s, yy_s , color='r', linestyle='--')
# ax5.plot(x_new_s, y_s, color='r', linestyle='--')
# ax5.plot(x_new2_s, yy_s , color='r', linestyle='--')
# plt.text(0.05, 0.92, 'hPN zone',
#          transform=ax5.transAx5es, fontsize=22)
# ax5.minorticks_on()

# plt.text(0.56, 0.92, 'SySt Zone',
#          transform=ax5.transAx5es, color="red", fontsize=22)
# ax5.minorticks_on()

#ax51.grid(which='minor')#, lw=0.3)
ax5.legend(scatterpoints=1, ncol=2, fontsize=12.3, loc="lower right", **lgd_kws)
#ax5.grid()
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'Fig5-f515f660-f515f861.pdf'
# save_path = '../../../../../Dropbox/paper-pne/Fig/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
