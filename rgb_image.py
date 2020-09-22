'''
Making RGB images from PLUS .fits
Autor: L. A. Gutiérrez Soto
28/06/19
'''

from __future__ import print_function
import aplpy
import numpy
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(
    description="""Plot side-by-side RGB images of sources""")

parser.add_argument("image_r", type=str,
                    default="1000001-JPLUS-01485-v2_iSDSS_swp-crop",
                    help="Name of original FITS image (section in database) in i")

parser.add_argument("image_g", type=str,
                    default="1000001-JPLUS-01485-v2_rSDSS_swp-crop",
                    help="Name of original FITS image (section in database) in r")

parser.add_argument("image_b", type=str,
                    default="1000001-JPLUS-01485-v2_gSDSS_swp-crop",
                    help="Name of original FITS image (section in database) in g")

parser.add_argument("--name", type=str,
                    default="PSP",
                    help="Name of the objet")

parser.add_argument("--vmin_r", type=float, default=None,
                    help="""Set minimum brightness directly - overrides minfactor - r""")
parser.add_argument("--vmax_r", type=float, default=None,
                    help="""Set maximum brightness directly - overrides maxfactor - r""")

parser.add_argument("--vmin_g", type=float, default=None,
                    help="""Set minimum brightness directly - overrides minfactor - g""")
parser.add_argument("--vmax_g", type=float, default=None,
                    help="""Set maximum brightness directly - overrides maxfactor - g""")

parser.add_argument("--vmin_b", type=float, default=None,
                    help="""Set minimum brightness directly - overrides minfactor - b""")
parser.add_argument("--vmax_b", type=float, default=None,
                    help="""Set maximum brightness directly - overrides maxfactor - b""")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info")


cmd_args = parser.parse_args()
image_r = cmd_args.image_r + ".fits"
image_g = cmd_args.image_g + ".fits"
image_b = cmd_args.image_b + ".fits"


#aplpy.make_rgb_cube(['1000001-JPLUS-01485-v2_iSDSS_swp-crop.fits', '1000001-JPLUS-01485-v2_rSDSS_swp-crop.fits',
                     #'1000001-JPLUS-01485-v2_gSDSS_swp-crop.fits'], 'JPLUS_cube.fits')

aplpy.make_rgb_cube([image_r, image_g, image_b], 'JPLUS_cube.fits')

aplpy.make_rgb_image('JPLUS_cube.fits', 'JPLUS_rgb.png',
                      vmin_r=cmd_args.vmin_r, vmax_r=cmd_args.vmax_r, vmin_g=cmd_args.vmin_g,
                                                      vmax_g=cmd_args.vmax_g, vmin_b=cmd_args.vmin_b, vmax_b=cmd_args.vmax_b)

#aplpy.make_rgb_image('JPLUS_cube.fits','JPLUS_linear.png')
#hdul = fits.open('JPLUS_cube_2d.fits')
# aplpy.make_rgb_image('JPLUS_cube.fits','JPLUS_rgb.png',
#                       stretch_r='arcsinh', stretch_g='arcsinh',
#                       stretch_b='arcsinh')


# With the mask regions, the file may not exist
position = "position.reg"
ra, dec = [], []

f = open(position, 'r')
header1 = f.readline()
header2 = f.readline()
header3 = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    coor = line.split("(")[-1].split("\"")[0]
    ra1, dec1 = coor.split(",")[0:2]
    c = SkyCoord(ra1, dec1, unit=(u.hourangle, u.deg))
    ra.append(c.ra.degree)
    dec.append(c.dec.degree)

# Launch APLpy figure of 2D cube
img = aplpy.FITSFigure('JPLUS_cube_2d.fits')
img.show_rgb('JPLUS_rgb.png')

# Maybe we would like the arcsinh stretched image more?
#img.show_rgb('ic348_color_arcsinh.png')

# Modify the tick labels for precision and format
# img.tick_labels.set_xformat('hh:mm:ss')
# img.tick_labels.set_yformat('dd:mm')
img.axis_labels.set_xtext('Right Ascension (J2000)')
img.axis_labels.hide_x()
img.axis_labels.set_ytext('Declination (J2000)')
img.axis_labels.set_font(size=25, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
img.axis_labels.hide()
img.axis_labels.hide_y()

img.tick_labels.set_font(size=25, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
#img.axis_labels.set_yposition('right')
img.tick_labels.set_yposition('right')
#img.tick_labels.hide()
img.tick_labels.hide_x()
img.tick_labels.hide_y()
# Let's add a scalebar to it
img.add_scalebar(20.0/3600.)
img.scalebar.set_label('20"')
img.scalebar.set(color='white', linewidth=4, alpha=0.9)
img.scalebar.set_font(size=70, weight='bold',
                      stretch='normal', family='sans-serif',
                      style='normal', variant='normal')

img.show_regions('position.reg')
#img.recenter(ra, dec, radius=20.0/3600.) #zoom
#img.show_markers(ra, dec, layer='marker', edgecolor='red', facecolor='none', marker='o', s=20, alpha=0.9, linewidths=100.)#, layer='marker_set_1', edgecolor='black', facecolor='none', s=30, alpha=0.5, linewidths=20)

# img.scalebar.set_font(size=23, weight='bold',
#                       stretch='normal', family='sans-serif',
#                       style='normal', variant='normal')

# We may want to lengthen the scalebar, move it to the top left,
# and apply a physical scale
#img.scalebar.set_corner('top left')
# img.scalebar.set_length(20/3600.)
# img.scalebar.set_label('20 arcsec') 
img.set_theme('publication')
img.save('{}-IRG.pdf'.format(cmd_args.name))
