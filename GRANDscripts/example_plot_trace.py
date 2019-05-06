'''
python3 example_plot_2D.py <path> <input format> <antenna ID> 

Parameters:
path: path to folder containing the trace files and antpos.dat
input format:   e: readin electric field traces, 
                f: read in filtered traces (f1, f2 in MHz hardcoded in the scripr, 
                v: read in voltage traces 
antenna ID: number of desired antenna

                
Output:
will plot all 3 field components for an antenna 

Note: This is just an example file for reading-in and plotting signals of a full array.
It is far from being perfect, just for beginners in the hand-on session
'''

### frequency, used if 'f' is chosen in MHz
f1 = 50
f2 = 200


import sys
from sys import argv
import os

import numpy as np
from numpy import *

import matplotlib.pyplot as plt
import pylab

# path to folder containing the inp-file, trace files and antpos.dat 
path = sys.argv[1]
# antenna ID
i = sys.argv[3]

# load trace
if sys.argv[2] == 'e': # readin electric field trace
    txt = np.loadtxt(path+ 'a'+str(i)+'.trace') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all electric field in muV/m
if sys.argv[2] == 'v': # readin voltage trace
    txt = np.loadtxt(path+ 'out_'+str(i)+'.txt') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all voltage in muV
if sys.argv[2] == 'f': # readin filtered electric field trace
    txt = np.loadtxt(path+ 'a'+str(i)+'_'+str(f1)+'-'+str(f2)+'MHz.dat') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all electric field in muV/m




#### Plotting section 

if sys.argv[2] == 'e':
    unit='muV/m'
if sys.argv[2] == 'f':
    unit='muV/m (' +str(f1)+'-'+str(f2)+'MHz)'
if sys.argv[2] == 'v':
    unit='muV'



fig=plt.figure(1,figsize=(8, 5), dpi=120, facecolor='w', edgecolor='k')

plt.title("antenna ID: "+str(i))
plt.plot(txt.T[0], txt.T[1], 'b-', label= "Ex=NS, PtP="+str(abs(max(txt.T[1]) - min( txt.T[1])))+unit )
plt.plot(txt.T[0], txt.T[2], 'r-', label= "Ey=EW, PtP="+str(abs(max(txt.T[2]) - min( txt.T[2])))+unit )
plt.plot(txt.T[0], txt.T[3], 'g-', label= "Ez=Up, PtP="+str(abs(max(txt.T[3]) - min( txt.T[3])))+unit )

plt.xlabel(r"time (ns)", fontsize=16)
plt.ylabel(r"Amplitude ("+unit+")", fontsize=16)
plt.legend(loc='best', fancybox=True)

plt.show()


## Filter
