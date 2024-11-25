#!/bin/bash

#for d in $(cat dates.txt) ; do python  extract_viirs.py $d -39.0 -10.0 110.0 155.0 2 "default" 'Integer_Cloud_Mask' ; done
for d in $(cat dates.txt) ; do python  extract_viirs.py $d 4.0 17.0 -12.0 7.0 2 "default" 'Integer_Cloud_Mask' ; done
