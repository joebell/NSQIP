import numpy as np
import pandas as pd

def getOutcomeVariable(cdf, whichVar):
    
    if whichVar == 'readmission':
        y = cdf[['READMISSION1-Yes','READMISSION-Yes']].any(1)
        y[cdf[['READMISSION1-Yes','READMISSION-Yes']].isnull().all(1)] = np.nan
        return y
    elif whichVar == 'infection':
        columnList = ['SUPINFEC-Superficial Incisional SSI',\
                      'WNDINFD-Deep Incisional SSI']
        y = cdf[columnList].any(1)
        y[cdf[columnList].isnull().all(1)] = np.nan
        return y
    # True if mortality within 30 days
    elif whichVar == 'mortality':
        y = cdf['DOPERTOD'] != -100
        y[cdf['DOPERTOD'].isnull()] = np.nan
        return y
    elif whichVar == 'morbidity':

        # Superficial infections, deep infections, dehissence, pneumonia
        # reintubation, pulmonary embolus, renal failure, UTI
        # CVA, coma, neuro deficit, cardiac arrest, MI,
        # early transfusion, graft or prosthesis failure, sepsis, shock
        morbList = ['SUPINFEC','WNDINFD','DEHIS','OUPNEUMO',\
                    'REINTUB','PULEMBOL','OPRENAFL','URNINFEC',\
                    'CNSCVA','CNSCOMA','NEURODEF','CDARREST','CDMI',\
                    'OTHBLEED','OTHGRAFL','OTHDVT','OTHSYSEP','OTHSESHOCK']

        # Fix the DVT column
        cdf['OTHDVT-DVT Requiring Therapy'] = cdf[['OTHDVT-DVT Requiring Therapy',\
                                                   'OTHDVT-DVT Requiring Therap']].any(1)
        # Actually, only take the significant columns here
        morbList = [\
            'CDARREST-Cardiac Arrest Requiring CPR',
            'CDMI-Myocardial Infarction',
            'CNSCOMA-Coma greater than 24 Hours',
            'CNSCVA-Stroke/CVA',
            'DEHIS-Wound Disruption',
            'NEURODEF-Peripheral Nerve Injury',
            'OPRENAFL-Acute Renal Failure',
            'OTHBLEED-Transfusions/Intraop/Postop',
            'OTHDVT-DVT Requiring Therapy',
            'OTHGRAFL-Graft/Prosthesis/Flap Failure',
            'OTHSESHOCK-Septic Shock',
            'OTHSYSEP-Sepsis',
            'OUPNEUMO-Pneumonia',
            'PULEMBOL-Pulmonary Embolism',
            'REINTUB-Unplanned Intubation',
            'SUPINFEC-Superficial Incisional SSI',
            'URNINFEC-Urinary Tract Infection',
            'WNDINFD-Deep Incisional SSI']
        y = cdf[morbList].any(1)
        y[cdf[morbList].isnull().all(1)] = np.nan
        return y
    elif whichVar == 'reoperation':
        # REOPERATION, REOPERATION1
        y = cdf[['REOPERATION1-Yes','REOPERATION-Yes']].any(1)
        y[cdf[['REOPERATION1-Yes','REOPERATION-Yes']].isnull().all(1)] = np.nan
        return y