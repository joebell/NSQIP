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
from joeStats import safeSameConvolve

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
    
    def makeBinEdges(self, X):
        # lowerEdge = X.min()
        # upperEdge = X.max()
        
        # Create binnings for the model that omit the edges
        edgePercentile = .025
        lowIX = np.round(edgePercentile*len(X))
        highIX = np.round((1-edgePercentile)*len(X))
        sortX = np.sort(X.ravel())

        lowerEdge = sortX[int(lowIX)]
        upperEdge = sortX[int(highIX)]
        
        # Protect against degenerate binning
        if lowerEdge == upperEdge:
            print('*** Bin Edge Calculations in logisticSmoothing.py Failed ***')
            upperEdge = upperEdge + 1
        
        return lowerEdge, upperEdge

    def fit(self, X, y):
        
        lowerEdge, upperEdge = self.makeBinEdges(X)
        
        binEdges = np.linspace(lowerEdge,upperEdge, self.nBins + 1)
        binWidth = binEdges[1] - binEdges[0]
        
        totalCounts, _ = np.histogram(X,bins=binEdges)
        targetCounts, _ = np.histogram(X[y==1],bins=binEdges)
        
        self.G = gaussianKernel( binWidth, self.sigma)
        smTotal = safeSameConvolve(totalCounts,self.G.kernel)
        smTarget = safeSameConvolve(targetCounts,self.G.kernel)
        
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

        lowerEdge, upperEdge = self.makeBinEdges(X)        
        binEdges = np.linspace(lowerEdge,upperEdge, self.nBins + 1)
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
    
    def plotModel(self,axis):
        plt.plot(self.modelX,self.modelY)
        if self.bootedCI:
            plt.fill_between(self.modelX, self.upCI, self.lowCI, color='gray', alpha=0.5)
        self.G.plotKernel(axis)
        plt.ylabel('P(y | X)')
        plt.xlabel('X')
        return