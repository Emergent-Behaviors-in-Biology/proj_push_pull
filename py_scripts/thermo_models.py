from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import numpy as np
import scipy as sp
from scipy import optimize
import numpy.linalg as la
import joblib

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

def predict_twolayer_nowriter(ET, S1T, S2T, vpS1bg, vpS2bg, vuES1, alphaES1, vpS1S2, alphaS1S2):
    
    Ncells = len(S1T)
    
#     S1pT_array = np.zeros_like(S1T)
#     S2pT_array = np.zeros_like(S2T)
    
    
    
    def loop(i):
        

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
#         @njit    
        def func(x):
            
            (fS1p, fS1u) = x.tolist()
            
            S1p = fS1p * S1T[i]
            S1u = fS1u * S1T[i]
            
#             try:
#                 S1p = 10**logfS1p * S1T[i]
#                 S1u = 10**logfS1u * S1T[i]
#             except OverflowError as err:
#                 print(err.args)
#                 print("Overflowed:", i, logfS1p, logfS1u, S1T[i])
#                 S1p = 10**np.clip(logfS1p, -8, 8) * S1T[i]
#                 S1u = 10**np.clip(logfS1u, -8, 8) * S1T[i]
            
            
            E = ET[i] / (1+(S1p+S1u)/alphaES1)
            S2f = S2T[i] / (1 + S1p/alphaS1S2)
            
            pES1p = E/alphaES1/(1 + E/alphaES1 + S2f/alphaS1S2)
            
            S1pT = S1T[i] * vpS1bg / (vuES1*pES1p + vpS1bg + 1)
            S1uT = S1T[i] - S1pT
            
            f = np.zeros_like(x)
            
            f[0] = S1p/S1T[i] - S1pT/S1T[i]/(1 + E/alphaES1 + S2f/alphaS1S2)
            f[1] = S1u/S1T[i] - S1uT/S1T[i]/(1 + E/alphaES1)
            
#             f = f**2
#             print(la.norm(f))
            
#             print(x)
            
            return f
        
        x0 = np.zeros(2, np.float64)
        x0[0] = 1.0 # S1p/S1T
        x0[1] = 1.0 # S1u/S1T
                                      
        bounds = (0, 1)
        res = optimize.least_squares(func, x0=x0, bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                        method='trf', max_nfev=1000)
    
    
#         print(i, res.cost, res.fun, res.x, res.nfev, res.njev, res.message)
    
        if not res.success:
            print(res)
#             if (res.x > -2).any():
#                 print(i, res.message)
#                 print(10**res.x)
           
#         if res.cost > 1e-4:
#             print("cost too large")
#             print(res)
        
        (fS1p, fS1u) = res.x.tolist()
            
        S1p = fS1p * S1T[i]
        S1u = fS1u * S1T[i]
          
        E = ET[i] / (1+(S1p+S1u)/alphaES1)
        S2f = S2T[i] / (1 + S1p/alphaS1S2)
           
        pES1p = E/alphaES1/(1 + E/alphaES1 + S2f/alphaS1S2)
        pS1pS2u = S1p/alphaS1S2/(1 + S1p/alphaS1S2)

        S1pT = S1T[i] * vpS1bg / (vuES1*pES1p + vpS1bg + 1)
        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        return S1pT, S2pT
        
    res = joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
        
    return S1pT_array, S2pT_array

def predict_twolayer_noeraser(WT, S1T, S2T, vpS1bg, vpS2bg, vpWS1, alphaWS1, vpS1S2, alphaS1S2):
    
    Ncells = len(WT)
    
#     S1pT_array = np.zeros_like(WT)
#     S2pT_array = np.zeros_like(WT)
    
    def loop(i):

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
#         @njit    
        def func(x):
            
            (fS1p, fS1u) = x.tolist()
            
            S1p = fS1p * S1T[i]
            S1u = fS1u * S1T[i]
            
#             try:
#                 S1p = 10**logfS1p * S1T[i]
#                 S1u = 10**logfS1u * S1T[i]
#             except OverflowError as err:
#                 print(err.args)
#                 print("Overflowed:", i, logfS1p, logfS1u, S1T[i])
#                 S1p = 10**np.clip(logfS1p, -8, 8) * S1T[i]
#                 S1u = 10**np.clip(logfS1u, -8, 8) * S1T[i]
            
            
            W = WT[i] / (1+(S1p+S1u)/alphaWS1)
            S2f = S2T[i] / (1 + S1p/alphaS1S2)
            
            pWS1u = W/alphaWS1/(1 + W/alphaWS1)
            
            S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vpS1bg + 1)
            S1uT = S1T[i] - S1pT
            
            f = np.zeros_like(x)
            
            f[0] = S1p/S1T[i] - S1pT/S1T[i]/(1 + W/alphaWS1 + S2f/alphaS1S2)
            f[1] = S1u/S1T[i] - S1uT/S1T[i]/(1 + W/alphaWS1)
            
#             f = f**2
#             print(la.norm(f))
            
#             print(x)
            
            return f
        
        x0 = np.zeros(2, np.float64)
        x0[0] = 1.0 # S1p/S1T
        x0[1] = 1.0 # S1u/S1T
        
                              
        bounds = (0, 1)
        res = optimize.least_squares(func, x0=x0, bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                        method='trf', max_nfev=1000)
    
    
#         print(i, res.cost, res.fun, res.x, res.nfev, res.njev, res.message)
    
        if not res.success:
            print(res)
#             if (res.x > -2).any():
#                 print(i, res.message)
#                 print(10**res.x)
           
#         if res.cost > 1e-4:
#             print("cost too large")
#             print(res.message, res.cost)
        
        
        (fS1p, fS1u) = res.x.tolist()
            
        S1p = fS1p * S1T[i]
        S1u = fS1u * S1T[i]
          
        W = WT[i] / (1+(S1p+S1u)/alphaWS1)
        S2f = S2T[i] / (1 + S1p/alphaS1S2)
           
        pWS1u = W/alphaWS1/(1 + W/alphaWS1)
        pS1pS2u = S1p/alphaS1S2/(1 + S1p/alphaS1S2)

        S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vpS1bg + 1)
        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        return S1pT, S2pT
        
    res = joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
    return S1pT_array, S2pT_array
    
def predict_twolayer(WT, ET, S1T, S2T, vpS1bg, vpS2bg, vpWS1, alphaWS1, vuES1, alphaES1, vpS1S2, alphaS1S2):
    
    Ncells = len(WT)
    
#     S1pT_array = np.zeros_like(WT)
#     S2pT_array = np.zeros_like(WT)
    
    
    
#     for i in range(Ncells):
        
    def loop(i):

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
#         @njit    
        def func(x):
            
#             (logfS1p, logfS1u) = x.tolist()

#         S1p = 10**logfS1p * S1T[i]
#         S1u = 10**logfS1u * S1T[i]

            (fS1p, fS1u) = x.tolist()
            
            S1p = fS1p * S1T[i]
            S1u = fS1u * S1T[i]
            
            W = WT[i] / (1+(S1p+S1u)/alphaWS1)
            E = ET[i] / (1+(S1p+S1u)/alphaES1)
            S2f = S2T[i] / (1 + S1p/alphaS1S2)
            
            pWS1u = W/alphaWS1/(1 + W/alphaWS1 + E/alphaES1)
            pES1p = E/alphaES1/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
            
            S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vuES1*pES1p + vpS1bg + 1)
            S1uT = S1T[i] - S1pT
            
            f = np.zeros_like(x)
            
            f[0] = S1p/S1T[i] - S1pT/S1T[i]/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
            f[1] = S1u/S1T[i] - S1uT/S1T[i]/(1 + W/alphaWS1 + E/alphaES1)
            
            return f
        
        x0 = np.zeros(2, np.float64)
        x0[0] = 1.0 # S1p/S1T
        x0[1] = 1.0 # S1u/S1T
        
#         x0 = np.log10(x0)
                              
#         bounds = (-6, 0)
        bounds = (0, 1)
        res = optimize.least_squares(func, x0=x0, bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                        method='dogbox', max_nfev=1000)
        
    
#         print(res.message)
    
#         print(i, res.x, res.cost, res.fun, res.grad, res.message)
    
        if not res.success:
            print(res)
#             if (res.x > -2).any():
#                 print(i, res.message)
#                 print(10**res.x)
           
#         if res.cost > 1e-4:
#             print("cost too large")
#             print(res.message, res.cost)
        
        
#         (logfS1p, logfS1u) = res.x.tolist()
            
#         S1p = 10**logfS1p * S1T[i]
#         S1u = 10**logfS1u * S1T[i]

        (fS1p, fS1u) = res.x.tolist()
            
        S1p = fS1p * S1T[i]
        S1u = fS1u * S1T[i]
   
        W = WT[i] / (1+(S1p+S1u)/alphaWS1)
        E = ET[i] / (1+(S1p+S1u)/alphaES1)
        S2f = S2T[i] / (1 + S1p/alphaS1S2)
           
        pWS1u = W/alphaWS1/(1 + W/alphaWS1 + E/alphaES1)
        pES1p = E/alphaES1/(1 + W/alphaWS1 + E/alphaES1 + S2f/alphaS1S2)
        pS1pS2u = S1p/alphaS1S2/(1 + S1p/alphaS1S2)

        S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vuES1*pES1p + vpS1bg + 1)
        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
#         S1pT_array[i] = S1pT
#         S2pT_array[i] = S2pT

        return S1pT, S2pT
        
    res = joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
    return S1pT_array, S2pT_array
    