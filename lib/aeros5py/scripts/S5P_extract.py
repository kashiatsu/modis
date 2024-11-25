#!/home/farouk/anaconda3/bin/python
import glob
import read_s5p_v1_0
import read_s5p_mask_v3 #suports batch mode
import read_s5p_mask_v2 #no batch mode
from sys import exit

#input_paths=['/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/20191222/11346']
date = '20191224'
#date = '20200907'
input_paths= glob.glob(f'/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/{date}/*')
#input_paths= glob.glob(f'/DATA/SPECAT_2/farouk/S5P_l1b/CALIFORNIA/{date}/*')

#lat_min=-35.00
#lat_max=-34.80
#lon_min=144.00
#lon_max=145.00

#Sun glint check
#lat_min=-15.70
#lat_max=-13.5 #-11.46
#lon_min=137.22
#lon_max=138.5 #142.00

#reproduce error with 11390 25/12
#lat_min=-20.0
#lat_max=-19.0
#lon_min=120.0
#lon_max=121.0

#lat_min=-39.0 ; lat_max=-10.0 ; lon_min=110.0 ; lon_max=155.0 #big aus 
#lat_min=32.5 ; lat_max=41.5 ; lon_min=-124.5 ; lon_max=-100.0 #california fire 20200907
#lat_min=38.65 ; lat_max=41.63 ; lon_min=-124.90 ; lon_max=-121.40 #california fire 20200907 zoom on plume

lat_min=-39.0 ; lat_max=-10.0 ; lon_min=110.0 ; lon_max=155.0 #big aus HR 
#lat_min=-39.0 ; lat_max=-10.0 ; lon_min=130.0 ; lon_max=135.0 #big aus HR 
#lat_min=-39.0 ; lat_max=-37.0 ; lon_min=130.0 ; lon_max=132.0 #big aus HR 
#lat_min=-39.0 ; lat_max=-10.0 ; lon_min=140.0 ; lon_max=150.0 #transect 22 HR 
#lat_min=-39.0 ; lat_max=-10.0 ; lon_min=145.0 ; lon_max=152.0 #transect 24 HR 

output_path='/DATA/OTHERS_4/SPECAT/farouk/tmp/' #'./output/'
#output_path='./output/'
#output_path='/DATA/SPECAT_2/farouk/AUSTRALIA_transects/20191222/INPUTS'
#output_path='./output/'
cloud_fraction_max='1.0'
isrf_file='/DATA/SPECAT_2/farouk/S5P_l1b/S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc'

for input_path in input_paths : #each orbit
  os.mkdir(output_path + input_path.split('/')[-1])
  os.chdir(output_path + input_path.split('/')[-1])
  read_s5p_mask_v3.main(input_path, output_path, lat_min, lat_max, lon_min, lon_max, cloud_fraction_max, isrf_file, 1)

