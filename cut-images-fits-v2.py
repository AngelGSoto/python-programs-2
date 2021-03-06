'''
Cuting images fits
Based in pyFIST.py and extract-image.py from Henney program

'''
from __future__ import print_function
import numpy as np
import json
import os
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u 
import argparse
import sys


parser = argparse.ArgumentParser(
    description="""Cut images from fits files""")

parser.add_argument("source", type=str,
                    default="1000001-JPLUS-02363-v2_J0660",
                    help="Name of source (prefix for files) ")

parser.add_argument("--position", type=str,
                    default="HYDRA-0026-000010640-position",
                    help="Find the DS9 region")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + "_swp.fits"

path1 = "../"
try:
    hdu = fits.open(os.path.join(path1, regionfile))
except FileNotFoundError:
    hdu = fits.open(regionfile)
    
crop_coords_unit=u.degree

position = args.position + ".reg"
ra, dec = [], []

f = open(position, 'r')
header1 = f.readline()
header2 = f.readline()
header3 = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    coor = line.split("(")[-1].split("\"")[0]
    ra1, dec1, radiu = coor.split(",")[0:3]
    crop_c = coord.SkyCoord(ra1, dec1, unit=(u.degree, u.degree))
    #locc = sys.argv[1:]
    # ra = input('Enter RA: ')
    # dec = input('Enter DEC: ')
    # ra = args.ra
    # dec = args.dec
    print(crop_c)
    w = wcs.WCS(hdu[0].header)
    print(w)
     
    crop_radius = float(radiu)*0.2*4.0*u.arcsec
    #crop_radius = 100.0*0.2*u.arcsec         # PNe en SMC
    #crop_radius = 10.0*u.arcsec       # HII regions en SMC
    pix_scale = 0.0996*u.arcsec
    
    crop_c_pix = w.wcs_world2pix(crop_c.ra.degree, crop_c.dec.degree, 0)
    crop_radius_pixels = crop_radius.to(u.arcsec) / pix_scale.to(u.arcsec)
   
    x1 = int(np.clip(crop_c_pix[0]-crop_radius_pixels, 0, hdu[0].data.shape[0]-1))
    x2 = int(np.clip(crop_c_pix[0]+crop_radius_pixels, 0, hdu[0].data.shape[0]-1))
    y1 = int(np.clip(crop_c_pix[1]-crop_radius_pixels, 0, hdu[0].data.shape[1]-1))
    y2 = int(np.clip(crop_c_pix[1]+crop_radius_pixels, 0, hdu[0].data.shape[1]-1))
    

    hdu[0].data = hdu[0].data[y1:y2, x1:x2]
    
    hdu[0].header['CRPIX1'] -= x1
    hdu[0].header['CRPIX2'] -= y1
    # hdu[0].header['CRVAL1'] = crop_c.ra.degree
    # hdu[0].header['CRVAL2'] = crop_c.dec.degree
    w = WCS(hdu[0].header)
    
    #################### 
    #Save the new file##
    ####################
    outfile = regionfile.replace("_swp.fits", "_{}_swp-crop.fits".format(position.split("_p")[0]))
    new_hdu = fits.PrimaryHDU(hdu[0].data, header=hdu[0].header)
    new_hdu.writeto(outfile, output_verify="fix", overwrite=True)
