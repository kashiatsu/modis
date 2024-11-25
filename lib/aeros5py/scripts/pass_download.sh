#!/bin/bash
#this script downloads a single pass
#set -e
# -c lon1,lat1:lon2,lat2
#dhusget.sh -i TROPOMI -t 3 -T L1B_RA_BD1 -c 1.40,48.0:3.63,49.3
#dhusget.sh -i TROPOMI -T L2__CLOUD_ -S 2019-06-01T06:00:00.000Z -E 2019-06-01T23:00:00.000Z -c 1.40,48.0:3.63,49.3
#dhusget.sh -i TROPOMI -T L1B_IR_UVN -S 2019-06-01T01:00:00.000Z -E 2019-06-01T23:00:00.000Z -o product

# Australian fire
#dhusget.sh -i TROPOMI -T L1B_RA_BD1 -S 2019-12-20T00:00:00.000Z -E 2019-12-20T23:59:59.000Z -c 110.0,-25.0:180.0,-20.0
#dhusget.sh -i TROPOMI -T L1B_RA_BD* -S 2020-01-03T00:00:00.000Z -E 2020-01-03T23:59:59.000Z -c 132.78,-25.0:132.70,-20.1 


if [[ -f product_list ]] ; then rm product_list; fi
if [[ -f products-list.csv ]] ; then rm products-list.csv; fi

output_path='/DATA/SPECAT_2/farouk/S5P_l1b/'
#output_path='/scratch/farouk/S5P_l1b/'
mkdir -p $output_path

adate=$1
zone=$2

if [ $zone = oceania ] ;then echo $zone; zone=' 132.78,-25.0:132.70,-20.1';
echo $zone

elif [ $zone = idf ] ; then echo $zone; zone=' 1.40,48.0:3.63,49.3';
else echo $zone; echo 'Wrong argument!' ; echo 'pass_download.sh yyyy-mm-yy zone<oceania/idf>'; exit 1; fi



dhusget.sh -l 10 -i TROPOMI -T L1B_RA* -S ${adate}T00:00:00.000Z -E ${adate}T23:59:59.000Z -c $zone
#read -p "proceed downloading RA (y/n) ? " 

#if [[ $REPLY = y ]] ; then 
  n=0
  echo "Downloading RA "
  while IFS='' read -r line
  do
    n=$((n+1))
    filename="$(echo $line | cut -d ',' -f1)"
    orbit=`echo $filename | cut -d '_' -f8`
    mkdir -p "$output_path/tmp_nc/$adate/$orbit" && cd $_
    query="$(echo $line | cut -d ',' -f2)"
    bd=`echo $filename | cut -d '_' -f5`
    if [[ $bd == BD7 || $bd == BD8 ]]; then continue; 
    else 
      echo Now downloading $bd
      wget --user s5pguest --password s5pguest $query/\$value -O $output_path/tmp_nc/$adate/$filename.nc & 
    fi
    if [ $n == 2 ] ; then wait; fi
  done < products-list.csv
#fi
exit

dhusget.sh -l 10 -i TROPOMI -T L1B_IR_UVN -S ${adate}T00:00:00.000Z -E ${adate}T23:59:59.000Z
read -p "proceed downloading IR (y/n) ? " 

if [[ $REPLY = y ]] ; then
  if [[ ! -d "$output_path/tmp_nc/$adate" ]] ; then  mkdir $output_path/tmp_nc/$(adate) && cd $_ ; fi
  echo "Downloading IR"

  while IFS='' read -r line
  do
    filename="$(echo $line | cut -d ',' -f1)"
    query="$(echo $line | cut -d ',' -f2)"
    wget --user s5pguest --password s5pguest $query/\$value -O $output_path/tmp_nc/$adate/$filename.nc & 
  done < products-list.csv
fi



dhusget.sh -l 10 -i TROPOMI -T L2__CLOUD_ -S ${adate}T00:00:00.000Z -E ${adate}T23:00:00.000Z -c $zone
read -p "proceed downloading CLOUDS (y/n) ? "

if [[ $REPLY = y ]] ; then
  echo "Downloading CLOUDS"

  while IFS='' read -r line
  do
    filename="$(echo $line | cut -d ',' -f1)"
    orbit=`echo $filename | cut -d '_' -f9`
    mkdir -p "$output_path/tmp_nc/$adate/$orbit" && cd $_
    query="$(echo $line | cut -d ',' -f2)"
    wget --user s5pguest --password s5pguest $query/\$value -O $output_path/tmp_nc/$adate/$filename.nc & 
  done < products-list.csv
fi





echo 
echo Orbit $orbit from day $adate done
echo 
