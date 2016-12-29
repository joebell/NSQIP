#
#
#
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

class wrappedKDE(BaseEstimator):
 
    def __init__(self, bandwidth=1, kernel='gaussian', Pnull=0.5):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.Pnull = Pnull
        self._estimator_type = "classifier"
        self.modelAll = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol=.0001) 
        self.model1 = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol=.0001) 
        self.bootedCI = False
        return self
    
    def fit(self, X, y):
        
        self.modelAll.fit(X)
        self.model1.fit(X[y==1])
        self.Pnull = y.mean()
        return self
        
    def score(self, X, y):
        predRes = self.predict_proba(X)
        rocAUC = sklm.roc_auc_score(y,predRes)
        return rocAUC
    
    def predict_proba(self, X):
        
        logProbAll = self.modelAll.score_samples(X)
        logProb1 = self.model1.score_samples(X)
        proba = self.Pnull*np.exp(logProb1)/np.exp(logProbAll)
        return proba
    
    
    def bootModelCI(self, X, y, nBoots, alpha):

        lowerEdge, upperEdge = self.makeBinEdges(X)        
        binEdges = np.linspace(lowerEdge,upperEdge, self.nBins + 1)
        nSamp = len(y)
        
        print('\t Bootstrapping confidence intervals...')
        bootCurves = np.empty((nBoots,len(self.modelX)))
        for bootN in np.arange(nBoots):
            # Draw a collection of observations from the data with replacement
            sampIdx = [random.randint(0,nSamp-1) for n in np.arange(nSamp)]
            bootX = X[sampIdx]
            bootY = y[sampIdx]
          
            bootTotalCounts, _ = np.histogram(bootX,bins=binEdges)
            bootTargetCounts, _ = np.histogram(bootX[bootY==1],bins=binEdges)
            
            bSmTotal = safeSameConvolve(bootTotalCounts,self.G.kernel)
            bSmTarget = safeSameConvolve(bootTargetCounts,self.G.kernel)
            
            bootCurves[bootN,:] = bSmTarget/bSmTotal
    
        lciIdx = round(alpha/2*(nBoots-1))
        uciIdx = round((1-alpha/2)*(nBoots-1))
        bootCurves.sort(axis=0)
        self.lowCI = bootCurves[lciIdx,:]
        self.upCI  = bootCurves[uciIdx,:]
        self.bootedCI = True
        return 
    
    def plotModel(self,axis, x, **kwargs):
        proba = self.predict_proba(x)
        plt.plot(x,proba, axes=axis, **kwargs)
        # if self.bootedCI:
        #    plt.fill_between(self.modelX, self.upCI, self.lowCI, color='gray', alpha=0.5)
        # self.G.plotKernel(axis)
        plt.ylabel('P(y | X)')
        plt.xlabel('X')
        return