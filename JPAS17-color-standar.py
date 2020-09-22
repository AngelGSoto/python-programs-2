#-*- coding: utf-8 -*-
'''
Make color-color diagram for SPLUS
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
#import seaborn as sns
import sys
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.optimize import fsolve
import seaborn as sns

# fit curve models
def fit(ax, x,y, sort=True): #linel
    z = np.polyfit(x, y, 1)
    fit = np.poly1d(z)
    print(fit)
    if sort:
        x = np.sort(x)
    ax.plot(x, fit(x), label="fit func", color="k", alpha=1, lw=2.5)

def fit1(ax, x,y, sort=True): #curve
    z = np.polyfit(x, y, 2)
    fit1 = np.poly1d(z)
    print(fit1)
    if sort:
        x = np.sort(x)
    ax.plot(x, fit1(x), label="fit func", color="k", alpha=1, lw=2.5)

#Find the point inteception between two lines     
def findIntersection(m, y, m1, y1, x0):
    x = np.linspace(-10.0, 15.5, 200)
    return fsolve(lambda x : (m*x + y) - (m1*x + y1), x0)

pattern = "../*-spectros/*-JPAS17-magnitude.json"
file_list = glob.glob(pattern)

def filter_mag(e, s, f1, f2, f3):
    col, col0 = [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            diff = filter1 - filter2
            diff0 = filter3 - filter1
            col.append(diff)
            col0.append(diff0)
    
    return col, col0

def plot_mag(f1, f2, f3):
    x, y = filter_mag("HPNe", "", f1, f2, f3)
    x1, y1 = filter_mag("CV", "", f1, f2, f3)
    x2, y2 = filter_mag("E00", "", f1, f2, f3)
    x3, y3 = filter_mag("E01", "", f1, f2, f3)
    x4, y4 = filter_mag("E02", "", f1, f2, f3)
    x5, y5 = filter_mag("E00_900", "", f1, f2, f3)
    x6, y6 = filter_mag("E01_900", "", f1, f2, f3)
    x7, y7 = filter_mag("E02_900", "", f1, f2, f3)
    x8, y8= filter_mag("-DPNe", "", f1, f2, f3)
    x9, y9= filter_mag("QSOs-13", "", f1, f2, f3)
    x10, y10 = filter_mag("QSOs-24", "",  f1, f2, f3)
    x111, y111 = filter_mag("QSOs-30", "", f1, f2, f3)
    x11, y11 = filter_mag("QSOs-32", "", f1, f2, f3)
    x12, y12 = filter_mag("-SFGs", "", f1, f2, f3)
    x13, y13 = filter_mag("-sys", "", f1, f2, f3)
    x14, y14 = filter_mag("-sys-IPHAS", "", f1, f2, f3) 
    x15, y15 = filter_mag("-ExtHII", "", f1, f2, f3)
    x16, y16 = filter_mag("-sys-Ext", '', f1, f2, f3)
    x17, y17 = filter_mag("-survey", '', f1, f2, f3)
    x18, y18 = filter_mag("-SNR", '', f1, f2, f3)
    x19, y19 = filter_mag("extr-SySt-raman", '', f1, f2, f3)
    x20, y20 = filter_mag("-extr-SySt", '', f1, f2, f3)
    x21, y21 = filter_mag("-sys-raman", "", f1, f2, f3)
    x22, y22 = filter_mag("-sys-IPHAS-raman", "", f1, f2, f3)
    x23, y23 = filter_mag("ngc185-raman", "", f1, f2, f3)
    x24, y24 = filter_mag("SySt-ic10", "", f1, f2, f3)
    x25, y25 = filter_mag("LOAN-HPNe-", "", f1, f2, f3)
    x26, y26 = filter_mag("1359559-HPNe-", "", f1, f2, f3)
    x27, y27 = filter_mag("DdDm-1-HPNe-", "", f1, f2, f3)
    x28, y28 = filter_mag("E00_100", "", f1, f2, f3)
    x29, y29 = filter_mag("E01_100", "", f1, f2, f3)
    x30, y30 = filter_mag("E02_100", "", f1, f2, f3)
    x31, y31 = filter_mag("YSOs", "", f1, f2, f3)
    # x28, y28 = filter_mag("E00_300", "", f1, f2, f3)
    # x29, y29 = filter_mag("E00_600", "", f1, f2, f3)

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
        A1[4].append(a)
        B1[4].append(b)
    for a, b in zip(x10, y10):
        A1[5].append(a)
        B1[5].append(b)
    for a, b in zip(x111, y111):
        A1[6].append(a)
        B1[6].append(b)
    for a, b in zip(x11, y11):
        A1[6].append(a)
        B1[6].append(b)
    for a, b in zip(x12, y12):
        A1[7].append(a)
        B1[7].append(b)
    for a, b in zip(x13, y13):
        A1[8].append(a)
        B1[8].append(b)
    for a, b in zip(x14, y14):
        A1[9].append(a)
        B1[9].append(b)
    for a, b in zip(x15, y15):
        A1[10].append(a)
        B1[10].append(b)
    for a, b in zip(x16, y16):
        A1[11].append(a)
        B1[11].append(b)
    for a, b in zip(x17, y17):
        A1[12].append(a)
        B1[12].append(b)
    for a, b in zip(x18, y18):
        A1[13].append(a)
        B1[13].append(b)
    for a, b in zip(x19, y19):
        A1[14].append(a)
        B1[14].append(b)
    for a, b in zip(x20, y20):
        A1[15].append(a)
        B1[15].append(b)
    for a, b in zip(x21, y21):
        A1[16].append(a)
        B1[16].append(b)
    for a, b in zip(x22, y22):
        A1[17].append(a)
        B1[17].append(b)
    for a, b in zip(x23, y23):
        A1[18].append(a)
        B1[18].append(b)
    for a, b in zip(x24, y24):
        A1[19].append(a)
        B1[19].append(b)
    for a, b in zip(x25, y25):
        A1[20].append(a)
        B1[20].append(b)
    for a, b in zip(x26, y26):
        A1[21].append(a)
        B1[21].append(b)
    for a, b in zip(x27, y27):
        A1[22].append(a)
        B1[22].append(b)
    for a, b in zip(x28, y28):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x29, y29):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x30, y30):
        A1[2].append(a)
        B1[2].append(b)
    for a, b in zip(x31, y31):
        A1[23].append(a)
        B1[23].append(b)

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6900", "Jv0915_6600", "Jv0915_4701")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111, axisbg="#eeeeee")
# ax1.set_ylim(ymin=-1.7,ymax=2.5)
# ax1.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J4701 - J6900$', fontsize=24)
plt.ylabel(r'$J6900 - J6600$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax1.scatter(y, x, c=z, s=50, edgecolor='')
ax1.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax1.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax1.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax1.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax1.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax1.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax1.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax1.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax1.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax1.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax1.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax1.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax1.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax1.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax1.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax1.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax1.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax1.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax1.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax1.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax1.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax1.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax1.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax1.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(-4.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax1.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
ax1.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax1.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax1.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax1.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax1.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax1, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax1.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax1.plot(x_new, y, color='k', linestyle='-')
# ax1.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax1.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.37, 0.95, r'(Cont(H$\alpha$) - H$\alpha$) vs (He II - Cont(H$\alpha$))',
         transform=ax1.transAxes, fontsize="x-large")

# ax1.plot(x_new_s, y_s, color='r', linestyle='--')
# ax1.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig2-JPAS17-J6600-J4701-f3.jpg')
plt.clf()

###################################################################################################################################
###################################################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_5101", "Jv0915_5001", "Jv0915_3900")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111, axisbg="#eeeeee")
# ax2.set_ylim(ymin=-1.7,ymax=2.5)
# ax2.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J3900 - J5101$', fontsize=24)
plt.ylabel(r'$J5101 - J5001$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax2.scatter(y, x, c=z, s=50, edgecolor='')
ax2.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax2.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax2.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax2.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax2.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax2.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax2.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax2.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax2.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax2.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax2.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax2.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax2.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax2.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax2.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax2.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax2.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax2.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax2.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax2.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax2.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax2.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax2.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax2.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(25.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax2.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax2.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax2.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax2.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax2.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax2.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax2.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax2, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax2.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O III] - [O III]) vs ([Ne III] - Cont([O III])',
         transform=ax2.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax2.plot(x_new, y, color='k', linestyle='-')
# ax2.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax2.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax2.plot(x_new_s, y_s, color='r', linestyle='--')
# ax2.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax2.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax2.set_title(" ".join([cmd_args.source]))
#ax2.grid(True)
#ax2.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax2.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax2.minorticks_on()
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax2.grid()
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig3-JPAS17-J5001-J3900-f3.jpg')
plt.clf()

#############################################################################################################
############################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_5101", "Jv0915_5001", "Jv0915_9100")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111, axisbg="#eeeeee")
ax3.set_ylim(ymin=-2.2,ymax=5.5)
# ax3.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J9100 - J5101$', fontsize=24)
plt.ylabel(r'$J5101 - J5001$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax3.scatter(y, x, c=z, s=50, edgecolor='')
ax3.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax3.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax3.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax3.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax3.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax3.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax3.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax3.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax3.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax3.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax3.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax3.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax3.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax3.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax3.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax3.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax3.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax3.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax3.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax3.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax3.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax3.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax3.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax3.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax3.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax3.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax3.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax3.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax3.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax3.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax3.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax3, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax3.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O III]) - [O III]) vs (S III - Cont([O III]))',
         transform=ax3.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax3.plot(x_new, y, color='k', linestyle='-')
# ax3.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax3.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax3.plot(x_new_s, y_s, color='r', linestyle='--')
# ax3.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax3.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax3.set_title(" ".join([cmd_args.source]))
#ax3.grid(True)
#ax3.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax3.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax3.minorticks_on()
#ax3.grid(which='minor')#, lw=0.3)
#ax3.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax3.grid()
#lgd = ax3.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax3.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig4-JPAS17-J5001-J9100-f3.jpg')
plt.clf()

#######################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6098", "Jv0915_6300", "Jv0915_9100")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax4 = fig.add_subplot(111, axisbg="#eeeeee")
#ax4.set_ylim(ymin=-2.2,ymax=5.5)
# ax4.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J9100 - J6098$', fontsize=24)
plt.ylabel(r'$J6098 - J6300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax4.scatter(y, x, c=z, s=50, edgecolor='')
ax4.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax4.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax4.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax4.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax4.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax4.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax4.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax4.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax4.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax4.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax4.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax4.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax4.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax4.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax4.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax4.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax4.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax4.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax4.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax4.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax4.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax4.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax4.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax4.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax4.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax4.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax4.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax4.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax4.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax4.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax4.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax4, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax4.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O I]) - [O I]) vs (S III - Cont([O I]))',
         transform=ax4.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax4.plot(x_new, y, color='k', linestyle='-')
# ax4.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax4.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax4.plot(x_new_s, y_s, color='r', linestyle='--')
# ax4.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax4.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax4.set_title(" ".join([cmd_args.source]))
#ax4.grid(True)
#ax4.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax4.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax4.minorticks_on()
#ax4.grid(which='minor')#, lw=0.3)
#ax4.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax4.grid()
#lgd = ax4.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax4.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig5-JPAS17-J6300-J9100-f3.jpg')
plt.clf()

###############################################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6098", "Jv0915_6300", "Jv0915_4701")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax5 = fig.add_subplot(111, axisbg="#eeeeee")
#ax5.set_ylim(ymin=-2.2,ymax=5.5)
# ax5.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J4701 - J6098$', fontsize=24)
plt.ylabel(r'$J6098 - J6300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax5.scatter(y, x, c=z, s=50, edgecolor='')
ax5.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax5.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax5.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax5.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax5.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax5.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax5.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax5.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax5.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax5.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax5.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax5.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax5.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax5.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax5.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax5.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax5.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax5.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax5.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax5.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax5.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax5.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax5.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax5.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax5.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax5.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax5.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax5.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax5.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax5.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax5.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax5, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax5.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O I]) - [O I]) vs (He II - Cont([O I]))',
         transform=ax5.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax5.plot(x_new, y, color='k', linestyle='-')
# ax5.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax5.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax5.plot(x_new_s, y_s, color='r', linestyle='--')
# ax5.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax5.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax5.set_title(" ".join([cmd_args.source]))
#ax5.grid(True)
#ax5.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax5.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax5.minorticks_on()
#ax5.grid(which='minor')#, lw=0.3)
#ax5.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax5.grid()
#lgd = ax5.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax5.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig6-JPAS17-J6300-J4701-f3.jpg')
plt.clf()

#############################################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6098", "Jv0915_6300", "Jv0915_6600")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax6 = fig.add_subplot(111, axisbg="#eeeeee")
#ax6.set_ylim(ymin=-2.2,ymax=5.5)
# ax6.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J6600 - J6098$', fontsize=24)
plt.ylabel(r'$J6098 - J6300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax6.scatter(y, x, c=z, s=50, edgecolor='')
ax6.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax6.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax6.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax6.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax6.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax6.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax6.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax6.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax6.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax6.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax6.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax6.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax6.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax6.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax6.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax6.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax6.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax6.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax6.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax6.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax6.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax6.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax6.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax6.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax6.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax6.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax6.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax6.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax6.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax6.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax6.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax6, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax6.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O I]) - [O I]) vs (H$\alpha$ - Cont([O I]))',
         transform=ax6.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax6.plot(x_new, y, color='k', linestyle='-')
# ax6.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax6.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax6.plot(x_new_s, y_s, color='r', linestyle='--')
# ax6.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax6.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax6.set_title(" ".join([cmd_args.source]))
#ax6.grid(True)
#ax6.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax6.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax6.minorticks_on()
#ax6.grid(which='minor')#, lw=0.3)
#ax6.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax6.grid()
#lgd = ax6.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax6.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig7-JPAS17-J6300-J6600-f3.jpg')
plt.clf()

##########################################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_7500", "Jv0915_7300", "Jv0915_6600")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax7 = fig.add_subplot(111, axisbg="#eeeeee")
#ax7.set_ylim(ymin=-2.2,ymax=5.5)
# ax7.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J6600 - J7500$', fontsize=24)
plt.ylabel(r'$J7500 - J7300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax7.scatter(y, x, c=z, s=50, edgecolor='')
ax7.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax7.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax7.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax7.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax7.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax7.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax7.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax7.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax7.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax7.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax7.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax7.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax7.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax7.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax7.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax7.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax7.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax7.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax7.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax7.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax7.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax7.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax7.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax7.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax7.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax7.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax7.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax7.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax7.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax7.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax7.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax7, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax7.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O II]) - [O II]) vs (H$\alpha$ - Cont([O II]))',
         transform=ax7.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax7.plot(x_new, y, color='k', linestyle='-')
# ax7.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax7.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax7.plot(x_new_s, y_s, color='r', linestyle='--')
# ax7.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax7.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax7.set_title(" ".join([cmd_args.source]))
#ax7.grid(True)
#ax7.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax7.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax7.minorticks_on()
#ax7.grid(which='minor')#, lw=0.3)
#ax7.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax7.grid()
#lgd = ax7.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax7.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig8-JPAS17-J7300-J6600-f3.jpg')
plt.clf()

#########################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_7500", "Jv0915_7300", "Jv0915_9100")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax8 = fig.add_subplot(111, axisbg="#eeeeee")
#ax8.set_ylim(ymin=-2.2,ymax=5.5)
# ax8.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J9100 - J7500$', fontsize=24)
plt.ylabel(r'$J7500 - J7300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax8.scatter(y, x, c=z, s=50, edgecolor='')
ax8.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax8.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax8.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax8.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax8.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax8.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax8.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax8.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax8.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax8.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax8.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax8.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax8.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax8.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax8.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax8.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax8.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax8.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax8.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax8.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax8.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax8.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax8.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax8.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax8.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax8.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax8.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax8.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax8.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax8.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax8.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax8, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax8.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O II]) - [O II]) vs (S III - Cont([O II]))',
         transform=ax8.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax8.plot(x_new, y, color='k', linestyle='-')
# ax8.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax8.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax8.plot(x_new_s, y_s, color='r', linestyle='--')
# ax8.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax8.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax8.set_title(" ".join([cmd_args.source]))
#ax8.grid(True)
#ax8.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax8.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax8.minorticks_on()
#ax8.grid(which='minor')#, lw=0.3)
#ax8.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax8.grid()
#lgd = ax8.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax8.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig9-JPAS17-J7300-J9100-f3.jpg')
plt.clf()

#############################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_7500", "Jv0915_7300", "Jv0915_5001")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax9 = fig.add_subplot(111, axisbg="#eeeeee")
#ax9.set_ylim(ymin=-2.2,ymax=5.5)
# ax9.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J5001 - J7500$', fontsize=24)
plt.ylabel(r'$J7500 - J7300$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax9.scatter(y, x, c=z, s=50, edgecolor='')
ax9.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


ax9.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax9.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax9.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax9.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax9.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax9.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax9.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax9.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax9.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax9.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax9.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax9.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax9.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax9.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax9.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax9.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax9.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax9.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax9.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax9.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax9.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax9.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')


bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax9.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(20.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
###################################################################
# ax1.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax9.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(55, -15), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=100)
ax9.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax9.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax9.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax9.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax9.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax9.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax9, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax9.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

plt.text(0.32, 0.95, r'(Cont([O II]) - [O II]) vs ([O III] - Cont([O II]))',
         transform=ax9.transAxes, fontsize="large")
#Mask
#mask = y >= result_y - 0.5
# ax9.plot(x_new, y, color='k', linestyle='-')
# ax9.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax9.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

# ax9.plot(x_new_s, y_s, color='r', linestyle='--')
# ax9.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax9.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax9.set_title(" ".join([cmd_args.source]))
#ax9.grid(True)
#ax9.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax9.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax9.minorticks_on()
#ax9.grid(which='minor')#, lw=0.3)
#ax9.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax9.grid()
#lgd = ax9.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax9.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig10-JPAS17-J7300-J5001-f3.jpg')
plt.clf()

#####################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_5001", "Jv0915_6600", "Jv0915_5101")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax10 = fig.add_subplot(111, axisbg="#eeeeee")
# ax10.set_ylim(ymin=-1.7,ymax=2.5)
# ax10.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J5101 - J5001$', fontsize=24)
plt.ylabel(r'$J5001 - J6600$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax10.scatter(y, x, c=z, s=50, edgecolor='')
ax10.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax10.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax10.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax10.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax10.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax10.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax10.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax10.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax10.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax10.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax10.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax10.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax10.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax10.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax10.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax10.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax10.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax10.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax10.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax10.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax10.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax10.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax10.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax10.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(-4.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
###################################################################
# ax10.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax10.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
ax10.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax10.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax10.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax10.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax10.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax10.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax10, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax10.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax10.plot(x_new, y, color='k', linestyle='-')
# ax10.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax10.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.32, 0.95, r'([O III] - H$\alpha$) vs (Cont([O III]) - [O III])',
         transform=ax10.transAxes, fontsize="x-large")

# ax10.plot(x_new_s, y_s, color='r', linestyle='--')
# ax10.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax10.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax10.set_title(" ".join([cmd_args.source]))
#ax10.grid(True)
#ax10.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax10.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax10.minorticks_on()
#ax10.grid(which='minor')#, lw=0.3)
#ax10.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax10.grid()
#lgd = ax10.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax10.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig11-JPAS17-J5001-J6600-f3.jpg')
plt.clf()

#####################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6600", "Jv0915_5001", "Jv0915_6900")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax11 = fig.add_subplot(111, axisbg="#eeeeee")
ax11.set_ylim(ymin=-4.2,ymax=1.9)
# ax11.set_xlim(xmin=-,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J6900 - J6600$', fontsize=24)
plt.ylabel(r'$J6600 - J5001$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax11.scatter(y, x, c=z, s=50, edgecolor='')
ax11.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax11.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax11.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax11.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax11.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax11.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax11.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax11.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax11.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax11.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax11.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax11.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax11.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax11.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax11.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax11.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax11.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax11.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax11.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax11.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax11.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax11.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax11.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax11.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(-4.0, 2.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
###################################################################
# ax11.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax11.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
ax11.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax11.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax11.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax11.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax11.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax11.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax11, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax11.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax11.plot(x_new, y, color='k', linestyle='-')
# ax11.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax11.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.25, 0.95, r'(H$\alpha$ - [O III]) vs (Cont(H$\alpha$) - H$\alpha$)',
         transform=ax11.transAxes, fontsize="x-large")

# ax11.plot(x_new_s, y_s, color='r', linestyle='--')
# ax11.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax11.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax11.set_title(" ".join([cmd_args.source]))
#ax11.grid(True)
#ax11.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax11.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax11.minorticks_on()
#ax11.grid(which='minor')#, lw=0.3)
#ax11.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax11.grid()
#lgd = ax11.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax11.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig12-JPAS17-J6600-J5001-f3.jpg')
plt.clf()

############################################################################################################
###########################################################################################################
label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_5001", "Jv0915_4701", "Jv0915_5101")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax12 = fig.add_subplot(111, axisbg="#eeeeee")
ax12.set_ylim(ymin=-5.0,ymax=2.2)
# ax12.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J5101 - J5001$', fontsize=24)
plt.ylabel(r'$J5001 - J4701$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax12.scatter(y, x, c=z, s=50, edgecolor='')
ax12.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax12.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax12.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax12.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax12.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax12.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax12.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax12.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax12.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax12.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax12.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax12.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax12.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax12.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax12.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax12.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax12.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax12.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax12.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax12.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax12.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax12.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax12.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax12.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(-4.0, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
###################################################################
# ax12.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax12.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=501)
ax12.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax12.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax12.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax12.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax12.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax12.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax12, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax12.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax12.plot(x_new, y, color='k', linestyle='-')
# ax12.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax12.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.30, 0.95, r'([O III] - He II) vs (Cont([O III]) - [O III])',
         transform=ax12.transAxes, fontsize="x-large")

# ax12.plot(x_new_s, y_s, color='r', linestyle='--')
# ax12.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax12.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax12.set_title(" ".join([cmd_args.source]))
#ax12.grid(True)
#ax12.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax12.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax12.minorticks_on()
#ax12.grid(which='minor')#, lw=0.3)
#ax12.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax12.grid()
#lgd = ax12.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax12.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig13-JPAS17-J5001-J4701-f3.jpg')
plt.clf()

############################################################################################################
###########################################################################################################
label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_6600", "Jv0915_4701", "Jv0915_6900")
        
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax13 = fig.add_subplot(111, axisbg="#eeeeee")
ax13.set_ylim(ymin=-5.5,ymax=1.3)
# ax13.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J6900 - J6600$', fontsize=24)
plt.ylabel(r'$J6600 - J4701$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax13.scatter(y, x, c=z, s=50, edgecolor='')
ax13.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax13.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax13.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax13.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax13.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax13.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax13.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax13.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax13.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax13.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax13.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax13.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax13.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax13.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax13.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax13.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax13.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax13.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax13.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax13.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. HII region in NGC 55')
ax13.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax13.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax13.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax13.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(5, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
###################################################################
# ax13.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax13.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-1, 5), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=501)
ax13.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax13.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax13.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax13.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax13.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax13.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax13, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax13.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax13.plot(x_new, y, color='k', linestyle='-')
# ax13.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax13.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.30, 0.95, r'(H$\alpha$ - He II) vs (Cont(H$\alpha$) - H$\alpha$)',
         transform=ax13.transAxes, fontsize="x-large")

# ax13.plot(x_new_s, y_s, color='r', linestyle='--')
# ax13.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax13.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax13.set_title(" ".join([cmd_args.source]))
#ax13.grid(True)
#ax13.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax13.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax13.minorticks_on()
#ax13.grid(which='minor')#, lw=0.3)
#ax13.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax13.grid()
#lgd = ax13.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax13.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig14-JPAS17-J6600-J4701-f3.jpg')
plt.clf()

#########################################################################################################################################
#########################################################################################################################################

label = []
label1 = ["H4-1", "PNG 135.9+55.9"]
n = 24
A1, B1 = [[] for _ in range(n)], [[] for _ in range(n)]
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        # if data['id'].endswith("-1-HPNe-"):
        #     label.append("")
        if data['id'].endswith("41-HPNe"):
            label.append("H4-1")
        # elif data['id'].endswith("PNG_1359559-HPNe-"):
        #     label.append("")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        elif data['id'].startswith("BB1-HP"):
            label.append("BB1")
        plot_mag("Jv0915_4600", "Jv0915_6600", "Jv0915_4701")
      
AB = np.vstack([A1[2],B1[2]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': np.array(B1[2]), 'y': np.array(A1[2]) })

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = np.array(A1[2])[idx], np.array(B1[2])[idx], z[idx]


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
#sns.set(style="dark", context="talk")
#sns.set_style('ticks')
fig = plt.figure(figsize=(7, 6))
ax14 = fig.add_subplot(111, axisbg="#eeeeee")
ax14.set_ylim(ymin=-0.7,ymax=6.4)
# ax14.set_xlim(xmin=-2.9,xmax=3.5)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel(r'$J4701 - J4600$', fontsize=24)
plt.ylabel(r'$J4600 - J6600$', fontsize=24)
#plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
ax14.scatter(y, x, c=z, s=50, edgecolor='')
ax14.scatter(B1[0], A1[0], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100, label='Obs. halo PNe')


#ax14.scatter(B1[20], A1[20], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)#, label='Obs. halo PNe')

ax14.scatter(B1[21], A1[21], color= sns.xkcd_rgb["aqua"], s=60,  edgecolor='black', zorder=100)

ax14.scatter(B1[22], A1[22], color= sns.xkcd_rgb["aqua"],  s=60,  edgecolor='black', zorder=100)

ax14.scatter(B1[1], A1[1], c=sns.xkcd_rgb['pale yellow'], alpha=0.9, s=40, cmap=plt.cm.hot, edgecolor='black', zorder =10, label='SDSS CVs')
ax14.scatter(B1[4], A1[4],  c= "mediumaquamarine" , alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='SDSS QSOs')
ax14.scatter(B1[5], A1[5],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black',  cmap=plt.cm.hot)#, label='SDSS QSOs (2.4<z<2.6)')
ax14.scatter(B1[6], A1[6],  c= "mediumaquamarine", alpha=0.6, s=40, marker='D', edgecolor='black', cmap=plt.cm.hot)#, label='SDSS QSOs (3.2<z<3.4)')
ax14.scatter(B1[7], A1[7],  c= "goldenrod", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='SDSS SFGs ')
ax14.scatter(B1[8], A1[8],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts ')
ax14.scatter(B1[16], A1[16],  c= "red", alpha=0.6, s=40, marker='s', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts ')
#ax14.scatter(B1[72], A1[72],  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax14.scatter(B1[14], A1[14],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 205')
ax14.scatter(B1[15], A1[15],  c= "red", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black')  #label=' SySts in NGC 205')
ax14.scatter(B1[9], A1[9],  c= "red", alpha=0.6, s=40, marker='^', cmap=plt.cm.hot, edgecolor='black')#,  label='IPHAS SySts')
ax14.scatter(B1[17], A1[17],  c= "red", alpha=0.6, s=60, marker='^', cmap=plt.cm.hot, edgecolor='black', label='IPHAS SySts')
ax14.scatter(B1[19], A1[19],  c= "red", alpha=0.6, s=40, marker='o', cmap=plt.cm.hot, edgecolor='black')#, label='Obs. SySts in IC10 ')
ax14.scatter(B1[18], A1[18],  c= "red", alpha=0.6, s=40, marker='v', cmap=plt.cm.hot, edgecolor='black', label='Obs. SySts in NGC 185')
#ax14.scatter(B1[73], A1[73],  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax14.scatter(B1[10], A1[10],  c= "gray", alpha=0.6, s=40, marker='D', cmap=plt.cm.hot, edgecolor='black', zorder=501, label='Obs. HII region in NGC 55')
ax14.scatter(B1[23], A1[23],  c= "lightsalmon", alpha=0.8, s=80, marker='*', edgecolor='black', label='Obs. YSOs')
#ax14.scatter(colx, coly,  c= "red", alpha=0.8, s=300, marker='*', label='HASH PN')
#ax14.scatter(B1[74], A1[74],  c= "black", alpha=0.8, marker='.', label='SN Remanents')

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for label_, x, y in zip(label, B1[0], A1[0]):
    ax14.annotate(label_, (x, y), alpha=5, size=8,
                   xytext=(-4.0, 3.6), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=502,)
###################################################################
# ax14.annotate("H4-1", (np.array(B1[139]), np.array(A1[139])), alpha=5, size=8,
#                    xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom',)
ax14.annotate("PNG 135.9+55.9", (np.array(B1[21]), np.array(A1[21])), alpha=5, size=8,
                   xytext=(-5, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,zorder=502,)
ax14.annotate("DdDm-1", (np.array(B1[22]), np.array(A1[22])), alpha=10, size=8,
                   xytext=(35, -10), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props,)
##################################################################
#Obersevado porJPLUS
#for label_, x, y in zip(label1, d_768_jplus[0], d_644_jplus[0]):
# ax14.annotate("H4-1", (d_768_jplus[0], d_644_jplus[0]), alpha=8, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='blue',)

# ax14.annotate("PNG 135.9+55.9", (d_768_jplus[1], d_644_jplus[1]), alpha=8, size=8,
#                    xytext=(68, -10), textcoords='offset points', ha='right', va='bottom', color='green',)

# ax14.annotate("CI Cyg", (d_768_jplus[2], d_644_jplus[2]), alpha=20, size=8,
#                    xytext=(-5, 3), textcoords='offset points', ha='right', va='bottom', color='yellow',)

# ax14.annotate("TX CVn", (d_768_jplus[3], d_644_jplus[3]), alpha=20, size=8,
#                    xytext=(18, -13), textcoords='offset points', ha='right', va='bottom', color='m',)


#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax14.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
# plt.annotate(
#     '', xy=(B1[2][0],  A1[2][0]+0.35), xycoords='data',                    #
#     xytext=(B1[42][0]+0.4, A1[42][0]+0.4), textcoords='data', fontsize = 7,# vector extinction
#     arrowprops=dict(edgecolor='black',arrowstyle= '<-'))                   #
# plt.annotate(
#     '', xy=(B1[2][0]+0.35,  A1[2][0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

#fit1(ax14, B1[2], A1[2]) # curve fit: -0.7744 x**2 - 2.605 x + 0.2183 

# t = np.linspace(-3.0, 0.5, 200)
# m = -0.8*t**2 - 2.8*t + 0.35 >= 0.2206
# tt = -0.8*t**2 - 2.8*t + 0.35
# ax14.plot(t[m], tt[m], c="r")

# Region where are located the PNe
result = findIntersection(0.43, 0.65, -6.8, -1.3, 0.0)
result_y = 8.0*result + 4.50

x_new = np.linspace(-15.0, result, 200)
x_new2 = np.linspace(-15.0, result, 200)
#x_new3 = np.linspace(-10.0, 1.1, 200)
y =  0.43*x_new + 0.65
yy = -6.8*x_new2 - 1.3

#Mask
#mask = y >= result_y - 0.5
# ax14.plot(x_new, y, color='k', linestyle='-')
# ax14.plot(x_new2, yy , color='k', linestyle='-')

#Viironen
x_new4 = np.linspace(-10.0, 11.1, 200)
y_v =  0.25*x_new4 + 1.9
#ax14.plot(x_new4, y_v, color='k', linestyle='--')

# Region of the simbiotic stars
result1 = findIntersection(-220, +40.4, 0.39, 0.73, 0.0)
x_new_s = np.linspace(-15.5, result1, 200)
x_new2_s = np.linspace(result1, 15.5, 200)
y_s = -220*x_new_s + 40.4
yy_s = 0.39*x_new2_s + 0.73

plt.text(0.32, 0.95, r'(Cont(He II) - H$\alpha$) vs (He II - Cont(He II))',
         transform=ax14.transAxes, fontsize="x-large")

# ax14.plot(x_new_s, y_s, color='r', linestyle='--')
# ax14.plot(x_new2_s, yy_s , color='r', linestyle='--')


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax14.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax14.set_title(" ".join([cmd_args.source]))
#ax14.grid(True)
#ax14.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax14.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax14.minorticks_on()
#ax14.grid(which='minor')#, lw=0.3)
#ax14.legend(scatterpoints=1, ncol=2, fontsize=10.0, loc='lower center', **lgd_kws)
#ax14.grid()
#lgd = ax14.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax14.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.jpg')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig15-JPAS17-J4701-J6600-f3.jpg')
plt.clf()
