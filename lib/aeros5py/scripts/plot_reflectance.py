#!/home/farouk/anaconda3/bin/python3.7

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources.py2_warn, cftime

compare2sim = False

fig = plt.figure(figsize=(17,10))
def main(directory, savefig, xlim) : 
    filename = directory.split('/')[-1]
    reflectance_sim = np.loadtxt(fname=directory, skiprows=5)

    if compare2sim :
        with open(path + 'Reflectance_vlidort_gome2_20191222_27736.asc_inv','r') as f1 :
            with open(path + 'tmp1','w') as f2 :
                for line in f1 :
                    f2.write(line.replace('%','#'))

        reflectance_sim1 = np.loadtxt(fname=path+'tmp1')
        plt.plot(reflectance_sim[l1:l2,0], reflectance_sim[l1:l2,1]-reflectance_sim[l1:l2,2], label='resid')
        plt.plot(reflectance_sim[l1:l2,0], reflectance_sim1[l1:l2,1]-reflectance_sim1[l1:l2,2], label='resid1')
        plt.grid(); plt.legend()
        plt.title('compare2sim')
        plt.show(); sys.exit()

    else :
        plt.plot(reflectance_sim[:,0], reflectance_sim[:,2], label='reflectance meas/soft')
        plt.plot(reflectance_sim[:,0], reflectance_sim[:,1], label='reflectance vlidort')
        plt.scatter(reflectance_sim[:,0], reflectance_sim[:,1]-reflectance_sim[:,2], label='meas-sim')
        plt.grid()
        plt.title(filename)
        plt.legend()
        if xlim[0] != None and xlim[1]!= None :
            plt.xlim(xlim[0], xlim[1])

        plt.suptitle(filename)
        if savefig :
            return plt.savefig(filename+'.png')
        else:
            return plt.show()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=str, help='Absolute or relative path to input file.')
    parser.add_argument('-s','--savefig', action='store_true', help='Save the figure under the input file name.')
    parser.add_argument('--xlim', type=float, nargs=2, default=[None, None], help='xlim lambda in nanometer')
    args = parser.parse_args()
    main(**vars(args))
