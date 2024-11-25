import numpy as np
import matplotlib.pyplot as plt
import sys
filename = sys.argv[1].split('/')[-1]
solar = np.loadtxt(sys.argv[1])
plt.plot(solar[:,0], solar[:,1])
plt.savefig(filename+'.png')
