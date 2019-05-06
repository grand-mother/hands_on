import numpy as np
import matplotlib.pyplot as plt

OUTPUTFILE='1ant10mavgnd.out'
Z0=120*np.pi
c0=3e8
REP='./data/'
OUTPUTFILE='butthalftripleX4p5m.out'

with open(REP+OUTPUTFILE,'r') as f:
    txt=f.read()
    txt=txt.split()
    txt=np.asarray(txt)

    ind=np.nonzero(txt=='(WATTS)')[0]
    realimp=np.zeros(len(ind))
    reactance=np.zeros(len(ind))

    for i in range(len(ind)):
        voltage=txt[ind[i]+3]
        if len(voltage)>11:
            current=txt[ind[i]+4]
            if len(current)>11:
                impedance=txt[ind[i]+5]
                if len(impedance)>11:
                    reactance[i]=impedance[11:]
                else:
                    reactance[i]=txt[ind[i]+6]
            else:
                impedance=txt[ind[i]+6]
                if len(impedance)>11:
                    reactance[i]=impedance[11:]
                else:
                    reactance[i]=txt[ind[i]+7]
        else:
            current=txt[ind[i]+5]
            if len(current)>11:                
                impedance=txt[ind[i]+6]
                if len(impedance)>11:
                    reactance[i]=impedance[11:]
                else:
                    reactance[i]=txt[ind[i]+7]
            else:
                impedance=txt[ind[i]+7]
                if len(impedance)>11:
                    reactance[i]=impedance[11:]
                else:
                    reactance[i]=txt[ind[i]+8]

        realimp[i]=float(impedance[0:11])

    ind=np.nonzero(txt=='FREQUENCY=')[0]
    freq=np.zeros(len(ind))

    for i in range(len(ind)):
        freq[i]=txt[ind[i]+1] #MHz

print(freq)

nangles=int(91*(90./5+1))
theta=np.zeros((len(realimp),nangles))
phi=np.zeros((len(realimp),nangles))
gain=np.zeros((len(realimp),nangles))
etheta=np.zeros((len(realimp),nangles))
ephi=np.zeros((len(realimp),nangles))
phasetheta=np.zeros((len(realimp),nangles))
phasephi=np.zeros((len(realimp),nangles))
lefftheta=np.zeros((len(realimp),nangles))
leffphi=np.zeros((len(realimp),nangles))
axialratio=np.zeros((len(realimp),nangles))
count=-1
countfreq=-1

with open(REP+OUTPUTFILE,'r') as f:
    for line in f:
        line=line.split()
        if len(line)>1:
            if line[0]=='DEGREES' and line[1]=='DEGREES':                
                countfreq=countfreq+1
                count=0
                continue
            if count>=0 and count<nangles:
                #print(theta,countfreq,count,line)
                theta[countfreq,count]=float(line[0])
                phi[countfreq,count]=float(line[1])
                gain[countfreq,count]=float(line[4])
                axialratio[countfreq,count]=float(line[5])
                if len(line)==12:
                    etheta[countfreq,count]=float(line[8])
                    ephi[countfreq,count]=float(line[10])
                    phasetheta[countfreq,count]=float(line[9])
                    phasephi[countfreq,count]=float(line[11])
                elif len(line)==11:
                    etheta[countfreq,count]=float(line[7])
                    ephi[countfreq,count]=float(line[9])
                    phasetheta[countfreq,count]=float(line[8])
                    phasephi[countfreq,count]=float(line[10])
                count=count+1


###unwrap of phase angle
print(phasetheta[:,0])

for i in range(1,len(freq)):
    for j in range(0,nangles):
        while phasetheta[i,j]-phasetheta[i-1,j]<-180:
            phasetheta[i,j]=phasetheta[i,j]+360
        while phasetheta[i,j]-phasetheta[i-1,j]>180:
            phasetheta[i,j]=phasetheta[i,j]-360
        while phasephi[i,j]-phasephi[i-1,j]<-180:
            phasephi[i,j]=phasephi[i,j]+360
        while phasephi[i,j]-phasephi[i-1,j]>180:
            phasephi[i,j]=phasephi[i,j]-360

print(phasetheta[:,0])

######save file with leff

for i in range(0,len(freq)):
    for j in range(0,nangles):
        leff=c0/(freq[i]*1e6)*np.sqrt(realimp[i]*(10**(gain[i,j]/10))/(Z0*np.pi))
        lefftheta[i,j]=leff*etheta[i,j]/np.sqrt(etheta[i,j]*etheta[i,j]+ephi[i,j]*ephi[i,j])
        leffphi[i,j]=leff*ephi[i,j]/np.sqrt(etheta[i,j]*etheta[i,j]+ephi[i,j]*ephi[i,j])

realimpbis=np.zeros((len(realimp),nangles))
freqbis=np.zeros((len(realimp),nangles))
reactancebis=np.zeros((len(realimp),nangles))
for j in range(0,nangles):
    freqbis[:,j]=freq
    realimpbis[:,j]=realimp
    reactancebis[:,j]=reactance

strtxt=REP+OUTPUTFILE[0:-4]+str('_leff')
np.save(strtxt, (freqbis,realimpbis,reactancebis,theta,phi,lefftheta,leffphi,phasetheta,phasephi))



indtheta=np.nonzero(theta[0,:]==30)[0]
indphi=np.nonzero(phi[0,:]==30)[0]
print(indtheta)
print(indphi)
indcom=np.intersect1d(indtheta,indphi)
print(indcom)
print(lefftheta[:,indcom],leffphi[:,indcom],phasetheta[:,indcom],phasephi[:,indcom])



######plots

plt.plot(freq,lefftheta[:,indcom])
plt.plot(freq,leffphi[:,indcom])
plt.plot(freq,phasetheta[:,indcom])
plt.plot(freq,phasephi[:,indcom])
#plt.show()
plt.close()
gain=10**(gain/10)
#plt.plot(gain[0,:])
#plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pylab as pl

gainkeep=np.zeros((len(freq),91))
leffthetakeep=np.zeros((len(freq),91))
leffphikeep=np.zeros((len(freq),91))
phasethetakeep=np.zeros((len(freq),91))
phasephikeep=np.zeros((len(freq),91))

for f in range(len(freq)):

    thetabis=np.zeros((19,91))
    phibis=np.zeros((19,91))
    gainbis=np.zeros((19,91))
    leffthetabis=np.zeros((19,91))
    phasethetabis=np.zeros((19,91))
    leffphibis=np.zeros((19,91))
    phasephibis=np.zeros((19,91))

    if freq[f]==sys.argv[1]:

      count=0
      for i in range(19):  # Fill in plot matrixes
        thetabis[i,:]=theta[f,0:91]
        phibis[i,:]=phi[f,0+count:91+count]
        gainbis[i,:]=gain[f,0+count:91+count]
        leffthetabis[i,:]=lefftheta[f,0+count:91+count]
        leffphibis[i,:]=leffphi[f,0+count:91+count]
        phasethetabis[i,:]=phasetheta[f,0+count:91+count]
        phasephibis[i,:]=phasephi[f,0+count:91+count]
        count=count+91

      print('Frequency=',freq[f])
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      
      '''X=gainbis*np.cos(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      Y=gainbis*np.sin(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      Z=gainbis*np.cos(thetabis*np.pi/180)'''
      
      X=np.sqrt(leffthetabis**2+leffphibis**2)*np.cos(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      Y=np.sqrt(leffthetabis**2+leffphibis**2)*np.sin(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      Z=np.sqrt(leffthetabis**2+leffphibis**2)*np.cos(thetabis*np.pi/180)
      
      # Create cubic bounding box to simulate equal aspect ratio
      max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
      Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
      Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
      Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
      # Comment or uncomment following both lines to test the fake bounding box:
      for xb, yb, zb in zip(Xb, Yb, Zb):
          ax.plot([xb], [yb], [zb], 'w')
      
      norm = colors.Normalize(vmin=0, vmax=1.3)
      surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, facecolors=cm.jet(norm(np.sqrt(leffthetabis**2+leffphibis**2))), linewidth=0, antialiased=False)
      m = cm.ScalarMappable(cmap=cm.jet,norm=norm)
      m.set_array(np.sqrt(leffthetabis**2+leffphibis**2))
      fig.colorbar(m)
      ax.set_title('l$_eq$ @'+str(freq[f]) +' MHz  [m]')
      
      #In EW mode, antenna is along the x axis, in NS mode, antenna is along the y axis
      ax.set_xlabel('EW')
      ax.set_ylabel('NS')
      plt.show()
        #plt.savefig('/home/sandra/designgrand/'+str(freq[f])+'.png')
        #plt.close()


    if 0:
      fig = plt.figure()
      ax = fig.gca(projection='3d')

      x=gainbis*np.cos(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      y=gainbis*np.sin(phibis*np.pi/180)*np.sin(thetabis*np.pi/180)
      z=gainbis*np.cos(thetabis*np.pi/180)
      surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, facecolors=cm.jet(gainbis), linewidth=0, antialiased=False)
      # Add a color bar which maps values to colors.
      #fig.colorbar(surf, shrink=0.5, aspect=5)

      ax.set_title('gain '+str(freq[f]))

      #In EW mode, antenna is along the x axis, in NS mode, antenna is along the y axis
      ax.set_xlabel('EW')
      ax.set_ylabel('NS')
      plt.show()
      #plt.savefig('/home/sandra/designgrand/'+str(freq[f])+'.png')
      plt.close()

    gainkeep[f,:]=gainbis[-1,:]
    leffthetakeep[f,:]=leffthetabis[-1,:]
    leffphikeep[f,:]=leffphibis[-1,:]
    phasethetakeep[f,:]=phasethetabis[-1,:]
    phasephikeep[f,:]=phasephibis[-1,:]



for f in range(len(freq)):
    #print(gainkeep[f,:])
    plt.plot(theta[f,0:91],gainkeep[f,:])
    plt.xlabel('Theta [deg]')
    plt.ylabel('Gain (phi=90)')
plt.show()

for f in range(len(freq)):
    #print(gainkeep[f,:])
    plt.plot(theta[f,0:91],phasephikeep[f,:])
    plt.xlabel('Theta [deg]')
    plt.ylabel('Phase phi [deg] (phi=90)')
plt.show()

print(realimp,reactance)

plt.plot(freq,realimp,'*-')
plt.plot(freq,reactance,'*-')
plt.xlabel('Freq [MHz]')
plt.ylabel('Resistance (blue) Reactance (green) [Ohm]')
plt.show()

'''
gain[gain<-39]=-39 #!!!!!!!!!!!!!!!!
for i in range(0,len(freq)):
    print(freq[i])
    ind=np.nonzero(axialratio[i,:]>=0.1)[0]
    indout=np.nonzero(axialratio[i,:]<0.1)[0]
    print(len(ind),len(indout))
    plt.hist(gain[i,indout],bins=80,range=[-40,40])
    plt.hist(gain[i,ind],bins=80,range=[-40,40])
    plt.xlabel('Gain [dBi]')
    plt.ylabel('Count')
    #plt.savefig(str(int(freq[i]))+'.png')
    plt.close()
plt.plot(gain[0,:],axialratio[0,:],'*')
plt.yscale('log')
plt.grid()
plt.show()'''



