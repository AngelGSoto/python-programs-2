'''
Making plots spectra from Lamost
''' 
from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord 
from astroquery.sdss import SDSS
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_context("poster")
import glob
import argparse
import os
import lineid_plot
import matplotlib as mpl

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs""")

parser.add_argument("filesource", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix")

cmd_args = parser.parse_args()
file_ = cmd_args.filesource + ".fits"

namefile = file_.split(".f")[0]

hdu = fits.open(file_)
hdudata = hdu[0].data
wl = hdudata[2]
Flux = hdudata[0]

#Labeled emission lines
line_wave = [4686., 5007.0, 6560.28]
line_flux = np.interp(line_wave, wl, Flux)
line_label1 = ["He II", "[O III]", r"H$\alpha$"]
label1_sizes = np.array([5, 5, 12])
arrow_tips = [1.3, 1.3, 3.3]
box_loc = [3e2, 4e3, 4e3]

figfile = file_.replace(".fits", ".pdf")
fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(111)
ax.set_title(namefile)
ax.set_xlim(3600,9100)
#plt.ylim(ymin=0.0,ymax=500)
ax.set(xlabel='Wavelength $(\AA)$')
ax.set(ylabel='Flux')
ax.plot(wl, Flux, linewidth=0.6, zorder=5)

# ax.axvline(6560.28, color='k', linewidth=0.3, linestyle='--', label=r"H$\alpha$")
# ax.axvline(5000.7, color='k', linewidth=0.3, linestyle='--', label="[O III]")
# ax.axvline(4686, color='k', linewidth=0.3, linestyle='--', label="He II")
#ax.legend()
#plt.tight_layout()
save_path = '../Plots-lamost/'
file_save = os.path.join(save_path, figfile)
plt.savefig(file_save)
plt.clf()

#Zoom                       
figfileZoom = file_.replace(".fits", "-zoom.pdf")
ax1 = fig.add_subplot(1,1,1)
ax1.set_title(namefile)
ax1.set_xlim(3600,9100)
ax1.set_ylim(-300.0,1000)
ax1.set(xlabel='Wavelength $(\AA)$')
ax1.set(ylabel='Flux')
ax1.plot(wl, Flux, linewidth=0.6, zorder=5)
ax1.axvline(6560.28, color='k', linewidth=0.3, linestyle='--', label=r"H$\alpha$")
ax1.axvline(5000.7, color='k', linewidth=0.3, linestyle='-', label="[O III]")
ax1.axvline(4686, color='r', linewidth=0.3, linestyle='-', label="He II")
ax1.legend() 
plt.tight_layout()
file_saveZoom = os.path.join(save_path, figfileZoom)
plt.savefig(file_saveZoom)
plt.clf()
