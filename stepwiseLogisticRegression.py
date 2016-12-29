import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import sklearn.metrics as sklm
import sklearn.linear_model as lm

class stepwiseLogisticRegression(BaseEstimator):
    
    def __init__(self):
        self.model = []    
        self.useCols = []
    
    def score(self, X, y):
        predRes = self.predict_proba(X[self.useCols])
        rocAUC = sklm.roc_auc_score(y,predRes)
        return rocAUC
    
    def predict_proba(self, X):
        predRes = self.model.predict_proba(X[self.useCols])[:,1]
        return predRes
    
    def modelLogLikelihood(self, X, y):
        
        pY = self.predict_proba(X)
        LLcomponents = np.log(pY*y + (1 - pY)*(1 - y))
        
        return LLcomponents.sum()
    
    def modelAIC(self, X, y):
        
        k = len(self.useCols)
        AIC = -2*self.modelLogLikelihood(X,y) + 2*k
        return AIC

    def fit(self, X, y, **kwargs):
    
        # Make a list of columns to include in our fit
        inclCol = pd.Series([False for colName in X.columns], dtype=bool, index=X.columns)
        inclCol['intercept'] = True
    
        # Loop while we're sitll adding or subtracting columns
        anyChanges = True
        while (anyChanges):
            anyChanges, inclCol, smresult = self.tryCols( X, y, inclCol, **kwargs)
        # print(smresult.summary())
        
        self.model = smresult
        self.useCols = inclCol[inclCol==True].index.values.tolist()
        return self

    def tryCols(self, X, y, inclCol, **kwargs):

        anyChanges = False

        baseCols = inclCol[inclCol==True].index.values.tolist()
        tryCols =  inclCol[inclCol==False].index.values.tolist()
        
        # Calculate the base model fit and AIC
        try:
            lr = lm.LogisticRegression(fit_intercept=True, penalty='l2', C=1000)
            baseModel = lr.fit(X[baseCols], y)
            self.model = baseModel
            self.useCols = baseCols
            baseAIC = self.modelAIC(X[baseCols], y)    
        except:
            baseAIC = np.inf
        print('baseAIC = %.3f' % baseAIC)

        # Try to add all the variables we should try out in the model, recording AIC
        fitResultList = []
        for colName in tryCols:
            modelCols = baseCols + [colName]

            try: 
                lr = lm.LogisticRegression(fit_intercept=True, penalty='l2', C=1000)
                lrresult = lr.fit(X[modelCols], y)
                self.model = lrresult
                self.useCols = modelCols
                modelAIC = self.modelAIC(X[modelCols], y) 
                fitResultList = fitResultList + [(modelAIC, colName, 'add', lrresult)]
            except:
                0
                # print('*** ' + colName + ': Exception ***')

        # Try to remove all the variables currently in the model, recording AIC
        for colName in baseCols:
            modelCols = [aColName for aColName in baseCols if aColName != colName]
            
            try: 
                lr = lm.LogisticRegression(fit_intercept=True, penalty='l2', C=1000)
                lrresult = lr.fit(X[modelCols], y)
                self.model = lrresult
                self.useCols = modelCols
                modelAIC = self.modelAIC(X[modelCols], y) 
                fitResultList = fitResultList + [(modelAIC, colName, 'remove', lrresult)]
            except:
                0
                # print('*** ' + colName + ': Exception ***')
                
        # If the most significant variable is below current AIC, add it to the list
        fitResultList.sort()
        bestFit = fitResultList[0]
        
        if bestFit[0] < baseAIC:
            if bestFit[2] == 'add':
                inclCol[bestFit[1]] = True 
                print('\t Adding: AIC = %.2f \t %s' % (bestFit[0], bestFit[1]))
            elif bestFit[2] == 'remove':
                inclCol[bestFit[1]] = False
                print('\t Removing: AIC = %.2f \t %s' % (bestFit[0], bestFit[1]))
            smresult = bestFit[3]
            anyChanges = True
        else:
            print('\t Not changing terms. AIC = %.2f' % baseAIC)
            smresult = baseModel
            anyChanges = False

        return anyChanges, inclCol, smresult