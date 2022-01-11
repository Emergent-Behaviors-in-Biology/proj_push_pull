from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import numpy as np
import scipy as sp

from numba import njit

@njit
def predict_nonpplatable(ST):
        
    return np.zeros_like(len(ST), np.float64)


@njit
def predict_substrate_only(ST, vbgp):

    SpT = ST*vbgp / (vbgp + 1)
        
    return SpT

@njit
def predict_push(WT, ST, vbgp, vWSp, alphaWS):
  
    Ncells = len(WT)
        
    poly_coeffs = np.zeros((Ncells, 3), np.float64)
    poly_coeffs[:, 0] = 1.0 # x^2
    poly_coeffs[:, 1] = 1.0 + (WT-ST)/alphaWS # x^1
    poly_coeffs[:, 2] = -ST/alphaWS # x^0
    
    
    Spf = np.zeros(Ncells, np.float64)
    for i in range(Ncells):
        
        roots = np.roots(poly_coeffs[i].astype(np.complex128))
                
        Spf[i] = np.max(np.real(roots)) * alphaWS
        
    Wf = WT/(1+Spf/alphaWS)
        
    pWSu = Wf/alphaWS/(1+Wf/alphaWS)

    SpT = ST*(vWSp*pWSu + vbgp)/ (vWSp*pWSu + vbgp + 1)
        
        
    return SpT


@njit
def predict_pushpull(WT, ET, ST, vbgp, vWSp, alphaWS, vESu, alphaES):
      
    Ncells = len(WT)
        
    poly_coeffs = np.zeros((Ncells, 4), np.float64)
    poly_coeffs[:, 0] = 1.0 # x^3
    poly_coeffs[:, 1] = (alphaWS+alphaES + WT + ET - ST)/np.sqrt(alphaWS*alphaES) # x^2
    poly_coeffs[:, 2] = 1.0 + (WT-ST)/alphaWS + (ET-ST)/alphaES # x^1
    poly_coeffs[:, 3] = -ST/np.sqrt(alphaWS*alphaES) # x^0
        
    Spf = np.zeros(Ncells, np.float64)
    for i in range(Ncells):
        
        roots = np.roots(poly_coeffs[i].astype(np.complex128))
                                
        Spf[i] = np.max(np.real(roots)) * np.sqrt(alphaWS*alphaES)
        
    Wf = WT/(1+Spf/alphaWS)
    Ef = ET/(1+Spf/alphaES)
        
    pWSu = Wf/alphaWS/(1+Wf/alphaWS+Ef/alphaES)
    pESp = Ef/alphaES/(1+Wf/alphaWS+Ef/alphaES)

    SpT = ST*(vWSp*pWSu + vbgp)/ (vWSp*pWSu + vESu*pESp + vbgp + 1)
        
        
    return SpT
        