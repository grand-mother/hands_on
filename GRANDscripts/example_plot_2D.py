'''
python3 example_plot_2D.py <path> <input format>

Parameters:
path: path to folder containing the trace files and antpos.dat
input format:   e: readin electric field traces, 
                f: read in filtered traces (f1, f2 in MHz hardcoded in the scripr, 
                v: read in voltage traces 
                
Output:
- will plot the total peak-to-peak distribution for an array in 3D
- will plot the peak-to-peak distribution for the single components as a 2D scatter plot

Note: This is just an example file for reading-in and plotting signals of a full array.
It is far from being perfect, just for beginners in the hand-on session
'''

### frequency, used if 'f' is chosen in MHz
f1 = 30
f2 = 80


import sys
from sys import argv
import os

import numpy as np
from numpy import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

from mpl_toolkits.mplot3d import Axes3D



# path to folder containing the inp-file, trace files and antpos.dat 
path = sys.argv[1]

# antpos.dat-file 
posfile = path +'antpos.dat' # 
positions=np.genfromtxt(posfile)# positions[:,0]:along North-South, positions[:,1]: along East-West ,positions[:,2]: Up, in m    
x_pos = positions.T[0]
y_pos = positions.T[1]
z_pos = positions.T[2]
#x_pos, y_pos, z_pos= np.loadtxt(posfile,delimiter=' ',usecols=(0,1,2),unpack=True)# positions[:,0]:along North-South, positions[:,1]: along East-West ,positions[:,2]: Up, in m 

number_ant = len(x_pos) # number of positions in the array
print('Number of antennas: ', number_ant)

# create an array
p2p_Ex = np.zeros(number_ant)
p2p_Ey = np.zeros(number_ant)
p2p_Ez = np.zeros(number_ant)
p2p_total = np.zeros(number_ant)
for i in range(0, number_ant): # loop over all antennas in folder
    try: 
        if sys.argv[2] == 'e': # readin electric field trace
            txt = np.loadtxt(path+ 'a'+str(i)+'.trace') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all electric field in muV/m
        if sys.argv[2] == 'v': # readin voltage trace
            txt = np.loadtxt(path+ 'out_'+str(i)+'.txt') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all voltage in muV
        if sys.argv[2] == 'f': # readin filtered electric field trace
            txt = np.loadtxt(path+ 'a'+str(i)+'_'+str(f1)+'-'+str(f2)+'MHz.dat') #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all electric field in muV/m
    
        # now it depends of which parameter we need the peak amplitude: here we go for the peak-to-peak amplitude
        p2p_Ex[i] = max(txt.T[1])-min(txt.T[1])
        p2p_Ey[i] = max(txt.T[2])-min(txt.T[2])
        p2p_Ez[i] = max(txt.T[3])-min(txt.T[3])
        amplitude = np.sqrt(txt.T[1]**2. + txt.T[2]**2. + txt.T[3]**2.) # combined components
        p2p_total[i] = max(amplitude)-min(amplitude)
    except IOError:
        p2p_Ex[i]=0.
        p2p_Ey[i]=0.
        p2p_Ez[i]=0.
        p2p_total[i]=0.




#################
#PLOTTING SECTION
#################
if sys.argv[2] == 'e':
    unit=r'muV/m'
if sys.argv[2] == 'f':
    unit=r'muV/m (' + str(f1)+'-'+str(f2)+'MHz)'
if sys.argv[2] == 'v':
    unit=r'muV'


##### Plot a 3d figure of the total peak amplitude to see the actual array and the the signal distribution
fig = plt.figure(1, figsize=(8,7), dpi=120, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
col=ax.scatter(x_pos, y_pos,z_pos , c=p2p_total,  vmin=min(p2p_total), vmax=max(p2p_total),  marker='o', cmap=cm.gnuplot2_r)
plt.colorbar(col)

ax.set_xlabel('positions along NS (m)')
ax.set_ylabel('positions along EW (m)')
ax.set_zlabel('positions Up (m)')
plt.title('peak-to-peak amplitude distribution in '+unit)



##### Plot a 2d figures of the NS, ES, UP component and total peak amplitude in positions along North-South and East-West 
fig2 = plt.figure(2,figsize=(12,7), dpi=120, facecolor='w', edgecolor='k')
    
ax1=fig2.add_subplot(221)
name = 'NS-component ('+unit+')'
plt.title(name)
ax1.set_xlabel('positions along NS (m)')
ax1.set_ylabel('positions along EW (m)')
col1=ax1.scatter(x_pos, y_pos, c=p2p_Ex,  vmin=min(p2p_Ex), vmax=max(p2p_Ex),  marker='o', cmap=cm.gnuplot2_r)
plt.colorbar(col1)
plt.tight_layout()
    
ax2=fig2.add_subplot(222)
name = 'EW-component ('+unit+')' 
plt.title(name)
ax2.set_xlabel('positions along NS (m)')
ax2.set_ylabel('positions along EW (m)')
col2=ax2.scatter(x_pos, y_pos, c=p2p_Ey,  vmin=min(p2p_Ey), vmax=max(p2p_Ey),  marker='o', cmap=cm.gnuplot2_r)
plt.colorbar(col2)
plt.tight_layout()
    
ax3=fig2.add_subplot(223)
name = 'Up-component ('+unit+')' 
plt.title(name)
ax3.set_xlabel('positions along NS (m)')
ax3.set_ylabel('positions along EW (m)')
col3=ax3.scatter(x_pos, y_pos, c=p2p_Ez,  vmin=min(p2p_Ez), vmax=max(p2p_Ez),  marker='o', cmap=cm.gnuplot2_r)
plt.colorbar(col3)
plt.tight_layout()

ax4=fig2.add_subplot(224)
name = 'total ('+unit+')'
plt.title(name)
ax4.set_xlabel('positions along NS (m)')
ax4.set_ylabel('positions along EW (m)')
col4=ax4.scatter(x_pos, y_pos, c=p2p_total,  vmin=min(p2p_total), vmax=max(p2p_total),  marker='o', cmap=cm.gnuplot2_r)
plt.colorbar(col4)
plt.tight_layout()


plt.show()
