__author__ = 'kek1'

import lmfit
import numpy as np


def initializeT1IR (nroi=None,TI=None, data=None, roi = None, useROIs = False):
    """initialize parameters for T1IR absolute value model"""
    nT1IRparams =3      #max number of parameters, some may be fixed
    if nroi == None:    #if no parameters are passed return the number of fitting parameters for this model
      return nT1IRparams
    T1params = lmfit.Parameters()   #define parameter dictionary
    paramlist = []    # list of parameters used for this model
    if useROIs: #if true use ROI values else use best guess
      T1guess = roi.T1
    else:
      # may not apply.
      T1guess=TI[np.argmin(data)]/np.log(2) #minimum signal should occur at ln(2)T1
    T1params.add('T1', value= T1guess, min=0, vary = True)
    paramlist.append('T1')
    T1params.add('Si', value= np.amax(data), vary = True)
    paramlist.append('Si')
    T1params.add('B',  value= 2,  min=1.5, max=2.5, vary = True)
    paramlist.append('B')
    return [T1params,paramlist]

# define objective function: returns the array to be minimized
def T1IR(params, TI, data):
    """ T1-IR model abs(exponential); TI inversion time array, T1 recovery time"""
    B = params['B'].value
    Si = params['Si'].value
    T1 = params['T1'].value

    model = Si*(1-B * np.exp(-TI/T1))
    return (model - data)

def fitT1IR(params, TI, data):
    """fits signal vs TI data to T1IRabs model"""
    result = lmfit.minimize(T1IRabs, params, args=(TI, data))
    final = data + result.residual
    return final
