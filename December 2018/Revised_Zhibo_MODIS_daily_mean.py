#!/usr/bin/env python
from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import os,datetime,sys,fnmatch
from jdcal import gcal2jd
#from plot_global_map import *
import math

def read_MODIS_level2_data(MOD06_file,MOD03_file):
    print(MOD06_file)
    print(MOD03_file)
    print('reading the cloud mask from MOD06_L2 product')
    MOD06 = Dataset(MOD06_file, 'r')
    CM1km = MOD06.variables['Cloud_Mask_1km']
    CM   = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
    print('level-2 cloud mask array shape',CM.shape)

    MOD03 = Dataset(MOD03_file,'r')
    print('reading the lat-lon from MOD03 product')
    lat  = MOD03.variables['Latitude']
    lon  = MOD03.variables['Longitude']
    print('level-2 lat-lon array shape',lat.shape)

    return lat,lon,CM

def value_locate(refx, x):
    refx = np.array(refx)
    x = np.array(x)
    loc = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else: loc[i] = ind[-1]

    return loc

def division(n, d):

    div = np.zeros(len(d))
    for i in range(len(d)):
        if d[i] >0:
          div[i]=n[i]/d[i]
        else: div[i]=None 

    return div


# beginning of the program
if __name__ == '__main__':
    import itertools
    MOD03_path = '/Users/huasong/Documents/Juypter/'
    MOD06_path = '/Users/huasong/Documents/Juypter/'
    satellite = 'Aqua'

    yr = [2008]
    mn = [1] #np.arange(1,13)  #[1]
    dy = [1] #np.arange(1,32) # [1] #np.arange(1,31)
    # latitude and longtitude boundaries of level-3 grid
    lat_bnd = np.arange(-90,91,1)
    lon_bnd = np.arange(-180,180,1)
    nlat = 180
    nlon = 360

    TOT_pix      = np.zeros(nlat*nlon)
    CLD_pix      = np.zeros(nlat*nlon)

    #for y,m,d in  itertools.product(yr,mn, dy):
        #-------------find the MODIS prodcts--------------#
    #    date = datetime.datetime(y,m,d)
    #    JD01, JD02 = gcal2jd(y,1,1)
    #    JD1, JD2 = gcal2jd(y,m,d)
    #    JD = np.int((JD2+JD1)-(JD01+JD02) + 1)
    #    granule_time = datetime.datetime(y,m,d,0,0)
    #    while granule_time <= datetime.datetime(y,m,d,23,55):  # 23,55
    #        print('granule time:',granule_time)
    MOD03_fp = 'MYD03.A*.hdf'
    MOD06_fp = 'MYD06_L2.A*.hdf'
    MOD03_fn, MOD06_fn =[],[]
    for MOD06_flist in  os.listdir(MOD06_path):
        if fnmatch.fnmatch(MOD06_flist, MOD06_fp):
           MOD06_fn = MOD06_flist
    for MOD03_flist in  os.listdir(MOD03_path):
        if fnmatch.fnmatch(MOD03_flist, MOD03_fp):
           MOD03_fn = MOD03_flist
    if MOD03_fn and MOD06_fn: # if both MOD06 and MOD03 products are in the directory
                print('reading level 2 geolocation and cloud data')
                print(MOD06_fn)
                Lat,Lon,CM = read_MODIS_level2_data(MOD06_path+MOD06_fn,MOD03_path+MOD03_fn)
                Lat=np.ravel(Lat)
                Lon=np.ravel(Lon)
                CM=np.ravel(CM)
                print('Total Number of pixels in this granule (cloud mask CM>=0)',np.sum(CM>=0))
                print('Total Number of cloudy pixels (cloud mask CM<=1)',np.sum(CM<=1))
                print('cloud fraction of this granule',np.sum(CM<=1)/np.sum(CM>=0))
                print('projecting granule on level3 lat lon grids')
                lat_index = value_locate(lat_bnd,Lat)
                lon_index = value_locate(lon_bnd,Lon)
                latlon_index = lat_index*nlon + lon_index
                print('computing simple level3 statistics')
                latlon_index_unique = np.unique(latlon_index)
                print('this granule occupies',latlon_index_unique.size,'1x1 degree box')
                for i in np.arange(latlon_index_unique.size):
                    j=latlon_index_unique[i]
                    TOT_pix[j] = TOT_pix[j]+np.sum(CM[np.where(latlon_index == j)]>=0)
                    CLD_pix[j] = CLD_pix[j]+np.sum(CM[np.where(latlon_index == j)]<=1) 
                
    #        granule_time += datetime.timedelta(minutes=5)

    print('derive the averaged Level-3 cloud fraction')
    total_cloud_fraction  =  division(CLD_pix,TOT_pix).reshape([nlat,nlon])
    print(np.nansum(total_cloud_fraction))

