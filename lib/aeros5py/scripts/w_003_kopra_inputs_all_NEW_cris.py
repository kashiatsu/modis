import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import glob, os, sys
#hidden 
import pkg_resources.py2_warn, cftime

#overpass = sys.argv[8]
#a = sys.argv[9]
#nprosjob = sys.argv[10]




def main(lat1, lat2, lon1, lon2, year1, month1, day1) :
  lat1 = float(sys.argv[1])
  lat2 = float(sys.argv[2])
  lon1 = float(sys.argv[3])
  lon2 = float(sys.argv[4])
  year1 = int(sys.argv[5])
  month1 = int(sys.argv[6])
  day1 = int(sys.argv[7])
  #date = datapath.split('/')[-1][15:23]
  date = f'{year1}{month1:02d}{day1:02d}'

  #datapath = '/home/farouk/cluster/DATA/SPECAT_2/farouk/CRIS_l1b/SNDR.SNPP.CRIS.20191225T0354.m06.g040.L1B.std.v02_22.G.191225140716.nc'
  #datapath = '/DATA/SPECAT_2/farouk/CRIS_l1b/SNDR.SNPP.CRIS.20191225T0354.m06.g040.L1B.std.v02_22.G.191225140716.nc'

  datapath = '/DATA/SPECAT_2/farouk/CRIS_l1b/AUSTRALIA/'

  filenames = glob.glob( datapath + date + '/*' ) 

  for filename in filenames :
    d = nc.Dataset(filename)


    #Interpolation
    wnum_lw = np.arange(d['wnum_lw'][:].data[0], d['wnum_lw'][:].data[-1], 0.25)

    #Writing files
    tab = '      '
    pixnum = len(glob.glob('./*.zen') ) + 10000
    lshape = d['lon'].shape
    with open( 'list.cloudfree', 'a') as cf :
      for a in range(lshape[0]):
        for x in range(lshape[1]):
          if (any(d['lon'][:][a,x] > lon1) and any(d['lon'][:][a,x] < lon2)) and (any(d['lat'][:][a,x] > lat1) and any(d['lat'][:][a,x] < lat2)) :
            for fov in range(lshape[2]):
              pixnum=pixnum+1
              cf.write( str(d['lat'][:][a,x,fov] ) + tab)       
              cf.write( str(d['lon'][:][a,x,fov] ) + tab)       
              cf.write( '0' + tab)       
              cf.write( date + '/CRIS_L1/' + date + '.' +  ''.join([str( d['obs_time_utc'][:][a,x,t]) for t in range(3,6)])+ '\n')       

              with open( str(pixnum) + '.latlong', 'w') as f :
                f.write( str( d['lat'][:][a,x,fov] ) + tab )
                f.write( str( d['lon'][:][a,x,fov] ))       

              with open( str(pixnum) + '.zen', 'w') as f :
                f.write( str( d['sat_zen'][:][a,x,fov] ))
       
              with open( str(pixnum) + '.UTtime', 'w') as f :
                f.write( tab.join( [ str( d['obs_time_utc'][:][a,x,t]) for t in range(0,6)]))


              rad_lw = np.interp(wnum_lw, d['wnum_lw'][:].data, d['rad_lw'][:][a,x,fov,:])*1e2 #convert mW/(m2 sr cm-1) to nW/(cm2 sr cm-1)

              # 6 Tsurf mws for OZONE ONLY
              m1 = ( wnum_lw > 830.50) * (wnum_lw < 834.75) 
              m2 = ( wnum_lw > 934.00) * (wnum_lw < 934.25)
              m3 = ( wnum_lw > 935.50) * (wnum_lw < 936.00)
              m4 = ( wnum_lw > 939.25) * (wnum_lw < 939.75)
              m5 = ( wnum_lw > 943.00) * (wnum_lw < 943.50)
              m6 = ( wnum_lw > 950.25) * (wnum_lw < 950.50)
              ts_mask = m1 + m2 + m3 + m4 + m5 + m6

              with open( str(pixnum) + '.mesTs', 'w') as f :
                np.savetxt(f, rad_lw[ts_mask] )

              #!!! Tprofile mws
              tp_mask = ( wnum_lw > 670) * (wnum_lw < 700)

              with open( str(pixnum) + '.mesTp', 'w') as f :
                np.savetxt(f, rad_lw[tp_mask] )

              # 7 O3 mws ONLY
              m1 = ( wnum_lw > 980.00 ) *( wnum_lw < 998.00 )
              m2 = ( wnum_lw > 1002.00) *( wnum_lw < 1009.00)
              m3 = ( wnum_lw > 1020.00) *( wnum_lw < 1027.00)
              m4 = ( wnum_lw > 1030.50) *( wnum_lw < 1049.75)
              m5 = ( wnum_lw > 1052.00) *( wnum_lw < 1055.00)
              m6 = ( wnum_lw > 1056.00) *( wnum_lw < 1065.00)
              m7 = ( wnum_lw > 1067.50) *( wnum_lw < 1073.25)
              o3_mask = m1 + m2 + m3 + m4 + m5 + m6 + m7

              with open( str(pixnum) + '.mesO3', 'w') as f :
                np.savetxt(f, rad_lw[o3_mask] )


  #os.system(f'tar -cf {date}.zen.spectra.tar *.zen --remove-files')
  #os.system(f'tar -cf {date}.latlong.spectra.tar *.latlong --remove-files')
  #os.system(f'tar -cf {date}.UTtime.spectra.tar *.UTtime --remove-files')
  #os.system(f'tar -cf {date}.mesTs.spectra.tar *.mesTs --remove-files')
  #os.system(f'tar -cf {date}.mesTp.spectra.tar *.mesTp --remove-files')
  #os.system(f'tar -cf {date}.mesO3.spectra.tar *.mesO3 --remove-files')

if __name__ == '__main__' :
  main(*sys.argv[1:-3])

#Read iasi example
#diasi = np.loadtxt('/home/farouk/cluster/DATA/SPECAT_2/farouk/CRIS_l1b/iasi/obr_metopb_20191201.txt', skiprows=1)
#print(diasi.shape) 

#print(diasi[19:27])


#iasi_grid = np.linspace()
 

#plt.plot( d['wnum_lw'][o3_mask].data, d['rad_lw'][:][6,6,6, o3_mask], label='O3' ) 
#plt.plot( d['wnum_lw'][ts_mask].data, d['rad_lw'][:][6,6,6, ts_mask], label='Ts' )
#plt.plot( d['wnum_lw'][tp_mask].data, d['rad_lw'][:][6,6,6, tp_mask], label='Tp' )
#plt.legend()
#plt.grid()
#plt.show()
#not found : SatID, Tb, Bearing, OrbitNumber, ScanLineNumber, DayVersion, CloudFraction, GqisQualFlag123 

#variables_list = ['obs_time_utc', 'lat', 'lon', 'sat_zen', 'sol_zen', 'sol_azi', 'sat_alt', 'wnum_lw', 'wnum_mw', 'wnum_sw', 'rad_lw', 'rad_mw', 'rad_sw', 'land_frac', 'fov_num']
#
#a=np.arange(1,8461)
#
#header = 'SatelliteIdentifier Tb  Year  Month Day Hour  Minute  Milliseconds  Latitude  Longitude SatelliteZenithAngle  Bearing SolarZenithAngle  SolarAzimuth  FieldOfViewNumber OrbitNumber ScanLineNumber  HeightOfStation DayVersion  StartChannel1 EndChannel1 GqisQualFlag1 StartChannel2 EndChannel2 GqisQualFlag2 StartChannel3 EndChannel3 GqisQualFlag3 CloudFraction SurfaceType' + " ".join(str(x) for x in a)
#
#fig = plt.figure(figsize=[10,10])
#ax = fig.add_subplot(211)
#plt.plot( d['wnum_lw'][:].data, d['rad_lw'][:][10,10,5,:].data)
#plt.plot( d['wnum_mw'][:].data, d['rad_mw'][:][10,10,5,:].data)
#plt.plot( d['wnum_sw'][:].data, d['rad_sw'][:][10,10,5,:].data)
#plt.title('CRIS')
#plt.grid()
#
#ax2 = fig.add_subplot(212)
#plt.plot(diasi[30:])
#plt.title('IASI')
#plt.grid()
#
#plt.show()



##Writing
#with open('obr_cris_' + date + '.txt', 'w') as f :
#  f.write(header)
#  for i in range(0,45):
#    for j in range(0,30):
#      for fov in range(0,9):
#        f.write(
#                f"""\n0\t0\t{d['obs_time_utc'][:][i,j,0]}\t\
#                    {d['obs_time_utc'][:][i,j,1]}\t{d['obs_time_utc'][:][i,j,2]}\t{d['obs_time_utc'][:][i,j,3]}\t\
#                    {d['obs_time_utc'][:][i,j,4]}\t{d['obs_time_utc'][:][i,j,5]}\t\
#                    {d['lat'][:][i,j,fov]}\t{d['lon'][:][i,j,fov]}\t\
#                    {d['sat_zen'][:][i,j,fov]}\t{d['sol_zen'][:][i,j,fov]}\t{d['sol_azi'][:][i,j,fov]}\t{d['fov_num'][:][fov]}\t0\t0\t{d['sat_alt'][:][i]}\t0\t\
#                    {d['wnum_lw'][:][0]}\t{d['wnum_lw'][:][-1]}\t0\t{d['wnum_mw'][:][0]}\t{d['wnum_mw'][:][-1]}\t0\t{d['wnum_sw'][:][0]}\t{d['wnum_sw'][:][-1]}\t0\t\
#                    0\t{d['land_frac'][:][i,j,fov]}\t""")
#
#        np.savetxt( f, d['rad_lw'][:][i,j,fov,:].data, newline=" ")
#        np.savetxt( f, d['rad_mw'][:][i,j,fov,:].data, newline=" ")
#        np.savetxt( f, d['rad_sw'][:][i,j,fov,:].data, newline=" ")
#




