# Nb: This produces a binary variable. Cases with multiple complications are only count once.
import numpy as np
import pandas as pd

def getOutcomeVariable(cdf, whichVar):
    
    # True if mortality within 30 days
    if whichVar == 'mortality':
        y = (cdf['DOPERTOD'] != -100)
        y[cdf['DOPERTOD'].isnull()] = 0
        return y
    
    # True if there's a major complication
    elif whichVar == 'major':
      
        morbList = ['WNDINFD-Deep Incisional SSI',\
                    'ORGSPCSSI-Organ/Space SSI',\
                    'DEHIS-Wound Disruption',\
                    'REINTUB-Unplanned Intubation',\
                    'PULEMBOL-Pulmonary Embolism',\
                    'FAILWEAN-On Ventilator greater than 48 Hours',\
                    'CNSCVA-Stroke/CVA',\
                    'CNSCOMA-Coma greater than 24 Hours',\
                    'CDARREST-Cardiac Arrest Requiring CPR',\
                    'CDMI-Myocardial Infarction',\
                    'OTHSYSEP-Sepsis',\
                    'OTHSESHOCK-Septic Shock',\
                    'RETURNOR-Yes',\
                    'REOPERATION-Yes']
        y = cdf[morbList].any(1)
        y[cdf[morbList].isnull().all(1)] = 0
        return y
    
    elif whichVar == 'minor':
        
        morbList = ['SUPINFEC-Superficial Incisional SSI',\
                    'OUPNEUMO-Pneumonia',\
                    'OPRENAFL-Acute Renal Failure',\
                    'NEURODEF-Peripheral Nerve Injury',\
                    'OTHDVT-DVT Requiring Therapy',\
                    'OTHDVT-DVT Requiring Therap']
        y = cdf[morbList].any(1)
        y[cdf[morbList].isnull().all(1)] = 0
        return y
                    
    elif whichVar == 'bleeding':
        # Old transfusion variable
        cdf.loc[:,'RBC-nonzero'] = (cdf.loc[:,'RBC'] > 0).astype(int)
        morbList = ['RBC-nonzero', 'OTHBLEED-Transfusions/Intraop/Postop']
        y = cdf[morbList].any(1)
        y[cdf[morbList].isnull().all(1)] = 0
        return y
                  
    elif whichVar == 'readmit':
        morbList = ['UNPLANREADMISSION-Yes']
        y = cdf[morbList].any(1)
        y[cdf[morbList].isnull().all(1)] = 0
        return y
    
 