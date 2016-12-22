import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

class gaussianKernel():
    
    def __init__(self, sampleInterval=1, sigma=1):
        self.sampleInterval = sampleInterval
        self.sigma = sigma

        nSigmas = 3
        nX = round(2*nSigmas*sigma/sampleInterval) + 1
        self.x = np.linspace(-nSigmas*sigma,nSigmas*sigma,nX)
        self.kernel = 1/(sigma*math.sqrt(2*math.pi))*np.exp(-self.x*self.x/(2*sigma*sigma))
        return 

    def plotKernel(self, axis):
    
        yScale = .25   
        xlims = axis.get_xlim()
        ylims = axis.get_ylim()

        # Plot the kernel on the plot
        plt.plot(self.x - self.x.min() + xlims[0], \
                 self.kernel*yScale*(ylims[1]-ylims[0])/self.kernel.max() + ylims[0],\
                 label='Kernel')
        return