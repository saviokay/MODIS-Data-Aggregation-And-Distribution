#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May  16 17:16:55 2017

@author: zhibo
"""
from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from pyhdf.SD import SD, SDC
import os,datetime,sys,fnmatch
from jdcal import gcal2jd
from plot_global_map import *
import math
from operator import add
from pyspark.sql import SparkSession

def read_MODIS_level2_data(MOD06_file,MOD03_file):
    print(MOD06_file)
    print(MOD03_file)
    print('reading the cloud mask from MOD06_L2 product')
    MOD06 = SD(MOD06_file, SDC.READ)
    CM1km = MOD06.select('Cloud_Mask_1km').get()
    CM   = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
#    print('level-2 cloud mask array shape',CM.shape)
    MOD06.end()

    MOD03 = SD(MOD03_file, SDC.READ)
#    print('reading the lat-lon from MOD03 product')
    lat  = MOD03.select('Latitude').get()
    lon  = MOD03.select('Longitude').get()
#    print('level-2 lat-lon array shape',lat.shape)
    MOD03.end()
    return lat,lon,CM

def value_locate(refx, x):
    """
    VALUE_LOCATE locates the positions of given values within a
    reference array.  The reference array need not be regularly
    spaced.  This is useful for various searching, sorting and
    interpolation algorithms.
    The reference array should be a monotonically increasing or
    decreasing list of values which partition the real numbers.  A
    reference array of NBINS numbers partitions the real number line
    into NBINS+1 regions, like so:
        REF:           X[0]         X[1]   X[2] X[3]     X[NBINS-1]
        <----------|-------------|------|---|----...---|--------------->
        INDICES:  -1           0          1    2       3        NBINS-1
        VALUE_LOCATE returns which partition each of the VALUES falls
        into, according to the figure above.  For example, a value between
        X[1] and X[2] would return a value of 1.  Values below X[0] return
        -1, and above X[NBINS-1] return NBINS-1.  Thus, besides the value
        of -1, the returned INDICES refer to the nearest reference value
        to the left of the requested value.

        Example:
            >>> refx = [2, 4, 6, 8, 10]
            >>> x = [-1, 1, 2, 3, 5, 5, 5, 8, 12, 30]
            >>> print value_locate(refx, x)
            array([-1, -1,  0,  0,  1,  1,  1,  3,  4,  4])

            This implementation is likely no the most efficient one, as there
            is
            a loop over all x, which will in practice be long. As long as x is
            shorter than 1e6 or so elements, it should still be fast (~sec).

    """

    refx = np.array(refx)
    x = np.array(x)
    loc = np.zeros(len(x), dtype='int')

    for i in xrange(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else: loc[i] = ind[-1]

    return loc

def division(n, d):

    div = np.zeros(len(d))
    for i in xrange(len(d)):
        if d[i] >0:
          div[i]=n[i]/d[i]
        else: div[i]=None

    return div


def aggregate(filenames):    ### Written by Hua Song

    MOD03_path = '/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD03/'
    MOD06_path = '/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD06_L2/'

    MOD06_fn=filenames.split(" ")[0]
    MOD03_fn=filenames.split(" ")[1]

    lat_bnd = np.arange(-90,91,1)
    lon_bnd = np.arange(-180,180,1)
    nlat = 180
    nlon = 360
    TOT_pix      = np.zeros(nlat*nlon)
    CLD_pix      = np.zeros(nlat*nlon)

    Lat,Lon,CM = read_MODIS_level2_data(MOD06_path+MOD06_fn,MOD03_path+MOD03_fn)
    Lat=Lat.ravel()
    Lon=Lon.ravel()
    CM=CM.ravel()
    lat_index = value_locate(lat_bnd,Lat)
    lon_index = value_locate(lon_bnd,Lon)
    latlon_index = lat_index*nlon + lon_index
    latlon_index_unique = np.unique(latlon_index)
    for i in np.arange(latlon_index_unique.size):
     #-----loop through all the grid boxes ccupied by this granule------#
        j=latlon_index_unique[i]
        TOT_pix[j] = np.sum(CM[np.where(latlon_index == j)]>=0)
        CLD_pix[j] = np.sum(CM[np.where(latlon_index == j)]<=1)
    return (TOT_pix,CLD_pix)



# Problem 2 for Homework 9 for daily mean cloud fraction
# Team 4

# beginning of the program
if __name__ == '__main__':
    import itertools
    MOD03_path = '/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD03/'
    MOD06_path = '/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD06_L2/'    
    satellite = 'Aqua'

    yr = [2008]
    mn = [1]  #np.arange(1,13) 
    dy = [1] 

    # latitude and longtitude boundaries of level-3 grid
    lat_bnd = np.arange(-90,91,1)
    lon_bnd = np.arange(-180,180,1)
    nlat = 180
    nlon = 360

    TOT_pix      = np.zeros(nlat*nlon)
    CLD_pix      = np.zeros(nlat*nlon)
 
    ### To use Spark in Python
    spark = SparkSession\
        .builder\
        .appName("Aggregation")\
        .getOrCreate()
    filenames0=['']*500
    i=0
    for y,m,d in  itertools.product(yr,mn,dy):
        #-------------find the MODIS prodcts--------------#
        date = datetime.datetime(y,m,d)
        JD01, JD02 = gcal2jd(y,1,1)
        JD1, JD2 = gcal2jd(y,m,d)
        JD = np.int((JD2+JD1)-(JD01+JD02) + 1)
        granule_time = datetime.datetime(y,m,d,0,0)
        while granule_time <= datetime.datetime(y,m,d,23,55):  # 23,55
            print('granule time:',granule_time)
            MOD03_fp = 'MYD03.A{:04d}{:03d}.{:02d}{:02d}.006.?????????????.hdf'.format(y,JD,granule_time.hour,granule_time.minute)
            MOD06_fp = 'MYD06_L2.A{:04d}{:03d}.{:02d}{:02d}.006.?????????????.hdf'.format(y,JD,granule_time.hour,granule_time.minute)
            MOD03_fn, MOD06_fn =[],[]
            for MOD06_flist in  os.listdir(MOD06_path):
                if fnmatch.fnmatch(MOD06_flist, MOD06_fp):
                    MOD06_fn = MOD06_flist
            for MOD03_flist in  os.listdir(MOD03_path):
                if fnmatch.fnmatch(MOD03_flist, MOD03_fp):
                    MOD03_fn = MOD03_flist
            if MOD03_fn and MOD06_fn: # if both MOD06 and MOD03 products are in the directory           
                filenames0[i]=MOD06_fn+' '+MOD03_fn
                i=i+1
            granule_time += datetime.timedelta(minutes=5)            
    
    ##### To split all the input files to different partitions for parallel computation    
    filenames=filter(lambda x: len(x)>0, filenames0)
    result = spark.sparkContext.parallelize(filenames, 288).map(lambda x: aggregate(x)).reduce(add)
    TOT_pix = np.sum(result[::2],axis=0)
    CLD_pix = np.sum(result[1::2],axis=0)
    ##### Written by Hua Song, using functions aggregate above
 
    print('derive the averaged Level-3 cloud fraction')
    total_cloud_fraction  =   division(CLD_pix,TOT_pix).reshape([nlat,nlon])

    print('plot global map')
    plot_global_map(lat_bnd,lon_bnd,total_cloud_fraction, cmap= plt.get_cmap('jet'), \
            vmin=0.0,vmax=1.0,title='cloud fraction', figure_name='MODIS_total_cloud_fraction_daily_mean_Spark')

    spark.stop()

