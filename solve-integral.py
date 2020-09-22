'''
Solve integral to find the expected number of planetary nebula
'''
from __future__ import print_function
import scipy.integrate as integrate
import scipy.special as special
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def number_PN(m, a, b):
    return a_0*np.exp(0.307*m)*(1. - np.exp(3.0*(a_1 - m)))

a_0 = np.array(input('a(0):'), dtype=float)
# a_1 = input('a(1):')
#a_0 = 0.0117
a_1 = 10.55

I = quad(number_PN, 10.55, 20.45, args=(a_0, a_1))

print('Number:', I)

# Plot PNF

m1 = np.linspace(10.55, 21.7)
PN = a_0*np.exp(0.307*m1)*(1. - np.exp(3.0*(a_1 - m1)))

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlim(xmin=-0.5,xmax=2)
#ax1.set_ylim(ymin=15,ymax=-5)
#ax1.set_xlabel(r'$\lambda$')
plt.tick_params(axis='x', labelsize=19) 
plt.tick_params(axis='y', labelsize=19)
ax1.set_xlabel(r'Mag)', size = 19)
ax1.set_ylabel(r'N', size = 19)
ax1.plot(m1, PN)
#ax1.plot(Wavelengthh, Fluxx, 'k-')
#ax1.grid(True)
#plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
plt.savefig("PNFL-MW.pdf")

