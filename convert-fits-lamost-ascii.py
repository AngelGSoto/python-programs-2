"""
Convert file.fits in file ascii (spectrum)
""" 
from __future__ import print_function
import glob
from astropy.io import fits
import os
import json
import numpy as np
import argparse
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

parser = argparse.ArgumentParser(
    description="""Convert file.fits in file ascii""")

parser.add_argument("source", type=str,
                    default="dddm1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
regionfile = cmd_args.source + ".fits"

hdu = fits.open(regionfile)

hdudata = hdu[0].data
wl = hdudata[2]
Flux = hdudata[0]

table = Table([wl, Flux], names=('wl', 'flux'), meta={'name': 'first table'})

asciifile = regionfile.replace(".fits", ".dat")
table.write(asciifile, format="ascii.commented_header", overwrite=True) 
