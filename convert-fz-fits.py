'''
Based on the progam of Gabriel.
Original vrsion: covert.py
'''
from astropy.io import fits, ascii
import os
import argparse

def fz2fits(image):
    """
    It converts SPLUS images
    from .fz to .fits
    """
    datos = fits.open(image)[1].data
    heada = fits.open(image)[1].header
    imageout = image[:-2] + 'fits'
    print ('Creating file: ')
    print (imageout)
    fits.writeto(imageout, datos, heada, overwrite=True)

parser = argparse.ArgumentParser(
    description="""Convert file.fz to file.fits""")

parser.add_argument("fzfile", type=str,
                    default="MC0095_F378_swp",
                    help="Name of file, taken the prefix ")

cmd_args = parser.parse_args()
fzfile_ = cmd_args.fzfile + ".fz"
 
fz2fits(fzfile_)

#color=['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861', 
       #'U', 'G', 'R', 'I', 'Z']

#for i in color:
    #fz2fits('MC0070/MC0070_'+i+'_swp.fz')
    #fz2fits('MC0072/MC0072_'+i+'_swp.fz')
    #fz2fits('MC0092/images/MC0092_'+i+'_swp.fz')
    #fz2fits('MC0093/images/MC0093_'+i+'_swp.fz')
    #fz2fits('MC0094/images/MC0094_'+i+'_swp.fz')
    #fz2fits('MC0095/MC0095_'+i+'_swp.fz')
    
