#!/bin/python


def main(directory, xcol, ycol, savefig, xlim):
    import numpy as np
    import matplotlib.pyplot as plt
    import pkg_resources.py2_warn
    filename = directory.split('/')[-1]
    data = np.loadtxt(directory)
    plt.plot(data[:,xcol],data[:,ycol])
    plt.grid()
    plt.title(filename)
    if savefig :
        return plt.savefig(filename+'.png')
    else:
        return plt.show()


if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('directory', type=str, help='Absolute or relative path to input file.')
    parser.add_argument('xcol', type=int, help='X axis column')
    parser.add_argument('ycol', type=int, help='y axis column')
    parser.add_argument('-s','--savefig', action='store_true', help='Save the figure under the input file name.')
    parser.add_argument('--xlim', type=float, nargs=2, default=[None, None], help='xlim lambda in nanometer')
    args = parser.parse_args()
    main(**vars(args))
