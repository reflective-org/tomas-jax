#! /usr/bin/python

# getRF.py

# This script calculates the radiative forcing using the method of 
# Chylek and Wong 1995

# import some modules
import numpy as np
from bhmie import bhmie
import Nio
from scipy.integrate import quad
import sys

print sys.argv

wl=0.500 # wavelength [microns]
refind=complex(1.4,1E-8) # refractive index
Tatm = 0.85 # transmittance of atmosphere above layer
#alb = 0.35 # albedo
So = 1370. # solar constant W/m2
#name='AERAVG-5MTpart095-eq-20km.nc' # file name
name=sys.argv[1]+'.nc' # file name
#temp=210 # temp in kelvin, guess for now!!!
grav=9.8 # # m s^-2

# get temperature data
infile=Nio.open_file('AER-temperature.nc','r')
temp=np.array(infile.variables['TEMPERATURE'])
infile.close()
#temp=temp.mean(0) # average over months

# solar declination angle 
daymonth=np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
monSolDecAng=[] # monthly solar declination angle (radians)
tot=0.
for m in range(0,12):
   avgday=tot+daymonth[m]/2.
   monSolDecAng.append(0.409*np.cos(2.*np.pi*(avgday-173.)/365.))
   tot=tot+daymonth[m]
monSolDecAng=np.array(monSolDecAng)

# get aerosol data from AER model
infile=Nio.open_file(name,'r')
lat=np.array(infile.variables['latitude'])
lev=np.array(infile.variables['level'])
rad=np.array(infile.variables['radius'])
time=np.array(infile.variables['time'])
aerodist=np.array(infile.variables['aerosol'])
infile.close()

def HG(asym,costh): # Henyey-Greenstein Phase function
                    # Perry, EQN. 11.23
    HGpp=(1.-asym**2)/(1.+asym**2-2.*asym*costh)**(3./2.)
    return HGpp

# average albedos (http://www.tak2000.com/data/planets/earth.htm)
def getAlbedo(lat): # lat is in degrees
   if lat>80.:
      alb=0.67
   elif lat>70.:
      alb=.57
   elif lat>60.:
      alb=.46
   elif lat>50.:
      alb=.41
   elif lat>40.:
      alb=.36
   elif lat>30.:
      alb=.31
   elif lat>20.:
      alb=.26
   elif lat>10.:
      alb=.24
   elif lat>0.:
      alb=.25
   elif lat>-10.:
      alb=.23
   elif lat>-20.:
      alb=.23
   elif lat>-30.:
      alb=.24
   elif lat>-40.:
      alb=.29
   elif lat>-50.:
      alb=.35
   elif lat>-60.:
      alb=.42
   elif lat>-70.:
      alb=.51
   elif lat>-80.:
      alb=.64
   else:
      alb=.70
   return alb

# Solar power (at equinox)
def SP(time,lat,sda,So):  # lat is in radians
                   # time is in hours
#   sda=0.0
   ang=np.pi/2.-np.arcsin(np.sin(lat)*np.sin(sda) \
#       -np.cos(lat)*np.cos(sda)*np.cos(np.pi*time/12.-np.pi/2.))
       -np.cos(lat)*np.cos(sda)*np.cos(2.*np.pi*time/24.))
   powr=-So*np.cos(ang)
   if powr<0.:
      powr=0.
   return powr

# get 24-hour average SW radiative flux at each latitude
avgPow=np.zeros((len(time),len(lat)))
alb=[]
for la in range(0,len(lat)):
   alb.append(getAlbedo(lat[la])) # determine albedo
   for t in range(0,len(time)):
      latin=lat[la]/180.*np.pi
      # integrate over day
      powr=quad(SP,0.,24.,args=(latin,monSolDecAng[t],So))[0]
      # divide by time to get average
      avgPow[t,la]=powr/24.
alb=np.array(alb)

# the following integrals are from Wiscomb and Grams 1976 eqn 22.
def int1(thpr,theta,asym): #theta is solar zenith angle, thpr is angle int over
   val=np.arccos(1./np.tan(theta)*1./np.tan(thpr))*HG(asym,np.cos(thpr)) \
       *np.sin(thpr)
   return val
def int2(thpr,theta,asym):
   val=HG(asym,np.cos(thpr))*np.sin(thpr)
   return val

sizepar=2.*np.pi*rad/wl # unitless size parameter
# initialize some variables
Qscar=np.zeros(rad.shape)
gscar=np.zeros(rad.shape)
optall=np.zeros(aerodist.shape)
radf=np.zeros(aerodist.shape)
upscafrac=np.zeros((len(time),len(rad),len(lat)))

for r in range(0,len(rad)):
   # run Bohren and Huffman Mie code
   [S1,S2,Qext,Qsca,Qback,gsca]=bhmie(sizepar[r],refind,2)
   Qscar[r]=Qsca
   gscar[r]=gsca
   for la in range(0,len(lat)):
      for t in range(0,len(time)):
         # solar zenith angle
         sza=(lat[la]+monSolDecAng[t])/180.*np.pi
         if abs(sza)<0.001:
            sza=0.001 # to avoid singularity
         if abs(sza)<90.:
            # get upscatter fraction using
            # Wiscombe and Grams 1976 eqn 22.
            int1out = quad(int1,np.pi/2.-sza,np.pi/2.+sza,args=(sza,gscar[r]))[0]
            int2out = quad(int2,np.pi/2.+sza,np.pi,args=(sza,gscar[r]))[0]
            upscafrac[t,r,la]=1./(2.*np.pi)*int1out+1./2.*int2out
         else:
            upscafrac[t,r,la]=0.0

         for l in range(0,len(lev)):
            # convert aerosol size distribution to number per kg/air
            adist=aerodist[t,r,l,la]*(22400*1013./lev[l]*temp[t,l,la]/273.)/0.029 # num/kg air
            sca=(rad[r]*1E-6)**2.*np.pi*adist*Qscar[r] # total scattering area
                                                       # m2/kg air
            # get optical depth contribution of each size particle in each
            # gridbox
            optall[t,r,l,la]=sca*lev[l]/grav*100.* \
                   np.log(lev[1]/lev[2]) # levels are equally spaced
            # Get radiative forcing from average incoming radiation,
            # transmission above layer, surface albedo, upscatter fraction and
            # optical depth
            # Radiative forcing of gridbox and size bin
            # From Chylek and Wong 1995
            radf[t,r,l,la]=avgPow[t,la]*Tatm**2*(1-alb[la])**2*2.*upscafrac[t,r,la]* \
                           optall[t,r,l,la]

tot=0.
rad=[]
for la in range(0,len(lat)):
   rad.append(np.cos(lat[la]/180.*np.pi))
   tot=tot+np.cos(lat[la]/180.*np.pi)
# scaling factor for fractional area from each latitude band
fforce=np.array(rad)/tot

# get total optical depths and RF as function of latitude
OD = optall[:,:,10:,:].mean(0).sum(0).sum(0)
RF = radf[:,:,10:,:].mean(0).sum(0).sum(0)
# globally averaged RF
avgRF=(RF*fforce).sum()

# Save output
np.savez(sys.argv[1]+'.npz',a=lat,b=RF,c=OD,d=avgRF)
