import pandas as pd
import numpy as np
import math
import scipy.spatial.distance as dist
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

# Convolution ensures results same length as longArg
def safeSameConvolve(longArg, shortArg):
    
        longLen = len(longArg)
        shortLen = len(shortArg)
        if longLen > shortLen:
            return np.convolve(longArg, shortArg, mode='same')
        else:
            # fullConv has length longLen + shortLen - 1
            fullConv = np.convolve(longArg, shortArg, mode='full')
            stIX = int(np.round(shortLen/2 - 1))
            enIX = stIX + longLen
            return fullConv[stIX:enIX]
            
# Utility function for doubly-centering a matrix X
def doublyCenter(X):
    Xcols = np.outer(np.ones([X.shape[0],1]),X.mean(axis=0))
    Xrows = np.outer(X.mean(axis=1),np.ones([1,X.shape[1]]))
    Xdc = X - Xcols - Xrows + X.mean()
    return Xdc

# Calculate the distance covariance of two matrices
# After: Székely, Gábor J., and Maria L. Rizzo. 2009. 
#        “Brownian Distance Covariance.” The Annals of Applied Statistics 3 (4).
#        Institute of Mathematical Statistics: 1236–65.
def distCov(X, Y):
    # Calculate pair-wise distance matrices
    Xd = dist.squareform(dist.pdist(X, metric='euclidean', p=2))
    Yd = dist.squareform(dist.pdist(Y, metric='euclidean', p=2))
    # Doubly center
    Xdc = doublyCenter(Xd)
    Ydc = doublyCenter(Yd)
    # Element-wise multiply
    dCov = (Xdc*Ydc).mean()
    return dCov

# Calculate the distance correlation of two matrices.
def distCorr(X, Y):
    # Calculate pair-wise distance matrices
    Xd = dist.squareform(dist.pdist(X, metric='euclidean', p=2))
    Yd = dist.squareform(dist.pdist(Y, metric='euclidean', p=2))
    # Doubly center
    Xdc = doublyCenter(Xd)
    Ydc = doublyCenter(Yd)
    # Element-wise multiply
    dCov = (Xdc*Ydc).mean()
    
    dVarX = (Xdc*Xdc).mean()
    dVarY = (Ydc*Ydc).mean()
    
    dCorr = dCov/math.sqrt(dVarX*dVarY)
    return dCorr            
            