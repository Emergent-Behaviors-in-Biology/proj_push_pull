from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import numpy as np
import scipy as sp
import numpy.linalg as la

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
    
    
def predict_twolayerpushpull(WT, ET, S1T, S2T, vpS1bg, vuS1bg, vpS2bg, vpWS1, alphaWS1, vuES1, alphaES1, vpS1S2, alphaS1S2):
    
    Ncells = len(WT)
    
    S1pT_array = np.zeros_like(WT)
    S2pT_array = np.zeros_like(WT)
    
    for i in range(Ncells):
        

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
            
        def func(x):
                        
            (logfS1p, logfS1u) = x.tolist()
            
            S1p = 10**logfS1p * S1T[i]
            S1u = 10**logfS1u * S1T[i]
            
            
            W = WT[i] / (1+(S1p+S1u)/alphaWS1)
            E = ET[i] / (1+(S1p+S1u)/alphaES1)
            S2f = S2T[i] / (1 + S1p/alphaS1S2)
            
            pWS1u = W/alphaWS1/(1 + W/alphaWS1 + E/alphaES1)
            pES1p = E/alphaES1/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
            
            S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vuES1*pES1p + vpS1bg + vuS1bg)
            S1uT = S1T[i] - S1pT
            
            f = np.zeros_like(x)
            
            f[0] = S1p/S1T[i] - S1pT/S1T[i]/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
            f[1] = S1u/S1T[i] - S1uT/S1T[i]/(1 + W/alphaWS1 + E/alphaES1)
            
#             print(la.norm(f))
            
#             print(x)
            
            return f
        
        x0 = np.zeros(2, np.float64)
        x0[0] = 0.5 #S1p/S1T
        x0[1] = 0.5 #S1u/S1T
        
        x0 = np.log10(x0)
            
        res = sp.optimize.root(func, x0=x0, method='hybr', options={'eps': 1e-6, 'xtol': 1e-6})
          
        if not res.success:
            if (res.x > -2).any():
                print(i, res.message)
                print(10**res.x)
           
        
        
        (logfS1p, logfS1u) = res.x.tolist()
            
        S1p = 10**logfS1p * S1T[i]
        S1u = 10**logfS1u * S1T[i]
          
        W = WT[i] / (1+(S1p+S1u)/alphaWS1)
        E = ET[i] / (1+(S1p+S1u)/alphaES1)
        S2f = S2T[i] / (1 + S1p/alphaS1S2)
           
        pWS1u = W/alphaWS1/(1 + W/alphaWS1 + E/alphaES1)
        pES1p = E/alphaES1/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
        pS1pS2u = S1p/alphaS1S2/(1 + S1p/alphaS1S2)

        S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vuES1*pES1p + vpS1bg + vuS1bg)
        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        S1pT_array[i] = S1pT
        S2pT_array[i] = S2pT
        
        
    return S1pT_array, S2pT_array



        
        