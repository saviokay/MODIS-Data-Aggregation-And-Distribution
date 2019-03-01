#!/bin/bash

#SBATCH --job-name=MODIS_Driver
#SBATCH --partition=batch
#SBATCH --qos=medium+
#SBATCH --mem=16000


python MODIS_Driver_XArray_Dask_GroupBy.py HDFFiles/ HDFFiles/ 
