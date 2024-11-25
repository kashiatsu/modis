
def read_l1b(directory) :
  import numpy as np
  nbands = int(1/4 * (sum(1 for line in open(directory)) -3))

  data = dict()

  filename = directory.split('/')[-1]
  for n in range(nbands):
      data[f'BAND{n+1}'] = dict()

  with open(directory, 'r') as l1b :
      #for _ in range(3): l1b.readline()
      l1b.readline()
      a = l1b.readline().split()
      l1b.readline()
      for bd in data.keys():
          data[bd]['lambdas_rad'] = np.array(l1b.readline().split(), dtype=float)
          data[bd]['rad'] = np.array(l1b.readline().split(), dtype=float)
          data[bd]['lambdas_irrad'] = np.array(l1b.readline().split(),dtype=float)
          data[bd]['irrad'] = np.array(l1b.readline().split(),dtype=float)
          data[bd]['lat'] = float(a[0])
          data[bd]['lon'] = float(a[1])

  return data


def read_measurements(directory) :
  import numpy as np
  nbands = int(1/4 * (sum(1 for line in open(directory)) -3))

  data = dict()

  filename = directory.split('/')[-1]
  for n in range(nbands):
      data[f'BAND{n+1}'] = dict()

  with open(directory, 'r') as l1b :
      #for _ in range(3): l1b.readline()
      l1b.readline()
      a = l1b.readline().split()
      l1b.readline()
      for bd in data.keys():
          data[bd]['lambdas_rad'] = np.array(l1b.readline().split(), dtype=float)
          data[bd]['rad'] = np.array(l1b.readline().split(), dtype=float)
          data[bd]['lambdas_irrad'] = np.array(l1b.readline().split(),dtype=float)
          data[bd]['irrad'] = np.array(l1b.readline().split(),dtype=float)
          data[bd]['lat'] = float(a[0])
          data[bd]['lon'] = float(a[1])

  directory_err = directory.replace('L1B','ERR')
  with open(directory_err, "r") as err :
      data['BAND1']['rad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND2']['rad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND3']['rad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND4']['rad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND1']['irrad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND2']['irrad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND3']['irrad_err'] = np.array(err.readline().split(), dtype=float)
      data['BAND4']['irrad_err'] = np.array(err.readline().split(), dtype=float)
  return data


def read_target(directory): 
  import numpy as np
  data = dict()
  filename = directory.split('/')[-1]
  with open(directory, 'r') as l1b :
    for _ in range(7): l1b.readline()
    a = l1b.readline().split()
    return np.asarray(a[1:], dtype=float)
      
  
