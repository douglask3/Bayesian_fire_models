import numpy as np
import iris

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
from pdb import set_trace

import sys
sys.path.append('libs/')
from select_key_or_default import *


class ConFire(object):
    def __init__(self, params, inference = False):
        """
        Initalise parameters and calculates the key variables needed to calculate burnt area.
        """
        self.inference = inference
        if self.inference:
            self.numPCK =  __import__('pytensor').tensor
        else:
            self.numPCK =  __import__('numpy')
        
        self.params = params

        def select_param_or_default(*args, **kw):
            return select_key_or_default(self.params, numPCK = self.numPCK, *args, **kw) 

        self.controlID = self.params['controlID']
        self.control_Direction = self.params['control_Direction']
        self.x0 = select_param_or_default('x0', [0])
        self.betas = select_param_or_default('betas', [[0]], stack = False)
        self.powers = select_param_or_default('powers', None, stack = False)
        self.driver_Direction = self.params['driver_Direction']
        self.Fmax = select_param_or_default('Fmax', 1.0, stack = False)

    def burnt_area(self, X, return_controls = False, return_limitations = False):
        ## finds controls        
        def cal_control(cid = 0):
            ids = self.controlID[cid]
            betas =  self.betas[cid] * self.driver_Direction[cid]
        
            X_i = X[:,ids]
            if self.powers is not None:
                X_i = self.numPCK.power(X_i, self.powers[cid])
            return self.numPCK.sum(X[:,ids] * betas[None, ...], axis=-1)
            
        
        controls = [cal_control(i) for i in range(len(self.controlID))]
        if return_controls: return controls

        def sigmoid(y, k):
            if k == 0: return None
            return 1.0/(1.0 + self.numPCK.exp(-y * k))
        
        
        limitations = [sigmoid(y, k) for y, k in zip(controls, self.control_Direction)]
        
        if return_limitations:
            return limitations

        limitations = [lim for lim in limitations if lim is not None]
        
        BA = sigmoid(self.Fmax, 1.0) * self.numPCK.prod(limitations, axis = 0)
        return BA
    
    
    def emc_weighted(self, emc, precip, wd_pg):
        
        try:
            wet_days = 1.0 - self.numPCK.exp(-wd_pg * precip)
            emcw = (1.0 - wet_days) * emc + wet_days
        except:
            emcw = emc.copy()
            emcw.data  = 1.0 - self.numPCK.exp(-wd_pg * precip.data)
            emcw.data = emcw.data + (1.0 - emcw.data) * emc.data
        return(emcw)


      

