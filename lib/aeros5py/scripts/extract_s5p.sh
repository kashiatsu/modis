#!/bin/bash
pwd=$(pwd)
if [ $1 == 1 ]; then
etiquet='bigaus_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
output_path='/DATA/SPECAT_2/farouk/AUSTRALIA_BIG/INPUTS/20191222/gome2_scratch/'
lat_min=-30.0 #-30.0
lat_max=-20.0 #-29.5
lon_min=125.0 #130.0
lon_max=135.0 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=0
elif [ $1 == 2 ] ; then
etiquet='smallaus_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
output_path='/DATA/SPECAT_2/farouk/AUSTRALIA_BIG/INPUTS/20191222/gome2_scratch/'
lat_min=-24.5 #-30.0
lat_max=-24.0 #-29.5
lon_min=128.0 #130.0
lon_max=128.5 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=0
elif [ $1 == 3 ] ; then
etiquet='test_isrf_aus_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
output_path='/DATA/SPECAT/farouk/AUSTRALIA_S5P_testing/INPUTS/20191222/gome2_scratch/'
lat_min=-24.5 #-30.0
lat_max=-24.0 #-29.5
lon_min=128.0 #130.0
lon_max=128.5 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=1
elif [ $1 == 4 ] ; then
etiquet='test_relaz_aus_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
output_path='/DATA/SPECAT/farouk/AUSTRALIA_S5P_testing/INPUTS/20191222/gome2_scratch1/'
lat_min=-24.5 #-30.0
lat_max=-24.0 #-29.5
lon_min=128.0 #130.0
lon_max=128.5 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=0
elif [ $1 == 5 ]; then
etiquet='bigaus2_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
output_path='/DATA/SPECAT_2/farouk/AUSTRALIA_BIG2/INPUTS/20191222/gome2_scratch/'
lat_min=-30.0 #-30.0
lat_max=-20.0 #-29.5
lon_min=135.0 #130.0
lon_max=145.0 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=1
elif [ $1 == 6 ]; then
etiquet='BIG3_s5p'
input_path='/DATA/SPECAT_2/farouk/S5P_l1b/AUSTRALIA/'
#output_path='./output/' #/DATA/SPECAT_2/farouk/AUSTRALIA_BIG2/INPUTS/20191222/'
output_path='/DATA/SPECAT_2/farouk/AUSTRALIA_BIG3/INPUTS/20191222/gome2_scratch/'
lat_min=-30.0 #-30.0
lat_max=-29.0 #-29.5
lon_min=140.0 #130.0
lon_max=141.0 #130.5 
cloud_fraction_max='1'
isrf_path='/DATA/SPECAT/farouk/S5P_L1B/ISRF/binned_uvn_spectral_unsampled/'
tar=1
fi

#if [[ ! -d $output_path ]] ; then mkdir $output_path; fi
#ln -s $output_path . 2> /dev/null

#nohup python read_s5p_v1_0.py $input_path $output_path $lat_min $lat_max $lon_min $lon_max $cloud_fraction_max $isrf_path > ./logs/$etiquet.log

#nohup python S5P_extract.py > ./logs/$etiquet.log

#cp ./logs/$etiquet.log $output_path

echo "....FINISH...."

batch_mode=`tail -n 1 $output_path/*.log`
#if [ $batch_mode == 'Finished : batch mode']; then
#  cd $output_path/
#  if [ $? != 0 ] ; then exit 1 ; fi
#  find ./isrf/  -maxdepth 1 -type f -name '*' -printf '%f\n' > isrf.files
#  #tar -cf $folder.tar --remove-files -C $folder --files-from $folder.files  && rm -r $folder
#  tar -cf isrf.tar -C isrf --files-from isrf.files
#exit

  nbatch=`ls $output_path | wc -w`
  nbatch=$((nbatch-3)) # for logfile and isrf

  echo nbatch = $nbatch
exit
  if [ $tar = 1 ]; then 
  for b in $(seq 0 $nbatch);
    do
      cd $output_path/$b/
        if [ $? != 0 ] ; then exit 1 ; fi
        for folder in $(ls .); 
          do
            find ./$folder/ -maxdepth 1 -type f -name '*' -printf '%f\n' > $folder.files
            #tar -cf $folder.tar --remove-files -C $folder --files-from $folder.files  && rm -r $folder
            tar -cf $folder.tar -C $folder --files-from $folder.files
          done
    done
  fi
exit
else

  if [ $tar = 1 ]; then
        cd $output_path/
          if [ $? != 0 ] ; then exit 1 ; fi
          for folder in $(ls .);
            do
              find ./$folder/ -maxdepth 1 -type f -name '*' -printf '%f\n' > $folder.files
              #tar -cf $folder.tar --remove-files -C $folder --files-from $folder.files  && rm -r $folder
              tar -cf $folder.tar -C $folder --files-from $folder.files
            done
  fi

fi

