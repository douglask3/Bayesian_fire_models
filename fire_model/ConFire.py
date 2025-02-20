import numpy as np
import iris
import pandas as pd

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

    def list_model_params(self, params, varnames = None):
        controlID = params[0]['controlID']
        def list_one_line_of_parmas(param):
            def select_param_or_default(*args, **kw):
                return select_key_or_default(param, numPCK = __import__('numpy'), *args, **kw)
            
            control_Direction = param['control_Direction']
            x0s = select_param_or_default('x0', [0])
            betas = select_param_or_default('betas', [[0]], stack = False)
            powers = select_param_or_default('powers', None, stack = False)
            driver_Direction = param['driver_Direction']
            Fmax = select_param_or_default('Fmax', 1.0, stack = False)


            directions = [np.array(driver)*control for control, driver in \
                          zip(control_Direction, driver_Direction)]
            betas = [direction * beta for direction, beta in zip(directions , betas)]
            

            def mish_mash(beta, power, x0):
                return np.append(np.column_stack((beta, power)).ravel(), x0)
            varps = [mish_mash(beta, power, x0) for beta, power, x0 in zip(betas, powers, x0s)]
            
            return np.append(Fmax, np.concatenate(varps))
            
        params_sorted = np.array([list_one_line_of_parmas(param) for param in params])
    
        param = ['Fmax']
        controlN = ['']
        variable_name = ['']
        
        for IDs, IDn in zip(controlID, range(len(controlID))):
            for ID in IDs:
                param.append('beta')
                controlN.append(str(IDn))
                variable_name.append(varnames[ID])
                param.append('power')
                controlN.append(str(IDn))
                variable_name.append(varnames[ID])
            param.append('beta')
            controlN.append(str(IDn))
            variable_name.append('beta0')

        header_df = pd.DataFrame([param, controlN, variable_name])   
        index_labels = ["Parameter", "Control", "Variable"] + \
                        list(map(str, range(1, params_sorted.shape[0] + 1)))

        data_df = pd.DataFrame(params_sorted)
        full_df = pd.concat([header_df, data_df], ignore_index=True) 

        full_df.index = index_labels

        return full_df
