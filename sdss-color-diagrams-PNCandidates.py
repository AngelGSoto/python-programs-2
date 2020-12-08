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
import argparse
from astropy.io import fits
import scipy.stats as st
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#Find the point inteception between two lines     
def findIntersection(m, y, m1, y1, x0):
    x = np.linspace(-10.0, 15.5, 200)
    return fsolve(lambda x : (m*x + y) - (m1*x + y1), x0)


parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("filesource", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
fitsfile = cmd_args.filesource + ".fits"

hdulist= fits.open(fitsfile)


def sdss(filter_1, filter_2, filter_3, filter_4):
    band1 = hdulist[1].data[filter_1]
    band2 = hdulist[1].data[filter_2]
    band3 = hdulist[1].data[filter_3]
    band4 = hdulist[1].data[filter_4]
    return (band1 - band2), (band3 - band4)

gr, ri = sdss("modelMag_g", "modelMag_r", "modelMag_r", "modelMag_i")
ug, rii = sdss("modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i")

################################################################################
#plots #########################################################################
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
ax1.scatter(gr, ri, c = sns.xkcd_rgb['neon purple'], alpha=0.7, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110)

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
pltfile = 'Fig1-ri_vs_gr_cand.pdf'
save_path = 'plots-sdss/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
###################################################################################
# Halpha -HalphaC vs OVi - OVICo###################################################
###################################################################################

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
ax2.scatter(ug, rii, c = sns.xkcd_rgb['neon purple'], alpha=0.7, s=50, cmap=plt.cm.hot, marker='o',  edgecolor='black', zorder=110)

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
pltfile = 'Fig2-ri_vs_ug_cand.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
#plt.savefig('Fig1-JPLUS17-Viironen.pdf')
plt.clf()
