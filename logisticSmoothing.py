#
#
#
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
from sklearn.base import BaseEstimator
from joeStats import gaussianKernel

class smoothedLookupEstimator(BaseEstimator):
 
    def __init__(self, sigma=1, nBins=512, modelX=[], modelY=[]):
        self.sigma = sigma
        self.nBins = nBins
        self.modelX = modelX
        self.modelY = modelY
        self._estimator_type = "classifier"
        self.G = []
        self.bootedCI = False
        self.lowCI = []
        self.upCI = []
        return

    def fit(self, X, y):
        
        binEdges = np.linspace(X.min(),X.max()+1, self.nBins + 1)
        binWidth = binEdges[1] - binEdges[0]
        
        totalCounts, _ = np.histogram(X,bins=binEdges)
        targetCounts, _ = np.histogram(X[y==1],bins=binEdges)
        
        self.G = gaussianKernel( binWidth, self.sigma)
        smTotal = np.convolve(totalCounts,self.G.kernel,mode='same')
        smTarget = np.convolve(targetCounts,self.G.kernel,mode='same')
        
        self.modelX = binEdges[0:-1] + binWidth/2
        self.modelY = smTarget/smTotal
        return
        
    def score(self, X, y):
        predRes = self.predict_proba(X)
        rocAUC = sklm.roc_auc_score(y,predRes)
        return rocAUC
    
    def predict_proba(self, X):
        
        sizeX = len(X)
        sizeModel = len(self.modelX)
        bigX = X.reshape(1,-1).repeat(sizeModel,axis=0)
        bigModel = self.modelX.reshape(-1,1).repeat(sizeX,axis=1)
        
        idx = (np.abs(bigX - bigModel)).argmin(axis=0)
        return self.modelY[idx]
    
    def bootModelCI(self, X, y, nBoots, alpha):

        binEdges = np.linspace(X.min(),X.max()+1, self.nBins + 1)
        nSamp = len(y)
        
        print('Bootstrapping confidence intervals...')
        bootCurves = np.empty((nBoots,len(self.modelX)))
        for bootN in np.arange(nBoots):
            # Draw a collection of observations from the data with replacement
            sampIdx = [random.randint(0,nSamp-1) for n in np.arange(nSamp)]
            bootX = X[sampIdx]
            bootY = y[sampIdx]
          
            bootTotalCounts, _ = np.histogram(bootX,bins=binEdges)
            bootTargetCounts, _ = np.histogram(bootX[bootY==1],bins=binEdges)
            
            bSmTotal = np.convolve(bootTotalCounts,self.G.kernel,mode='same')
            bSmTarget = np.convolve(bootTargetCounts,self.G.kernel,mode='same')
            
            bootCurves[bootN,:] = bSmTarget/bSmTotal
    
        lciIdx = round(alpha/2*(nBoots-1))
        uciIdx = round((1-alpha/2)*(nBoots-1))
        bootCurves.sort(axis=0)
        self.lowCI = bootCurves[lciIdx,:]
        self.upCI  = bootCurves[uciIdx,:]
        self.bootedCI = True
        return 
    
    def plotModel(self,axis):
        plt.plot(self.modelX,self.modelY)
        if self.bootedCI:
            plt.fill_between(self.modelX, self.upCI, self.lowCI, color='gray', alpha=0.5)
        self.G.plotKernel(axis)
        plt.ylabel('P(y | X)')
        plt.xlabel('X')
        return