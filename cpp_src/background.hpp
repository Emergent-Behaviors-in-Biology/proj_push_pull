#ifndef BG_HPP
#define BG_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cmath>
#include <Eigen/Dense>

#include "eigen_macros.hpp"

class Background {
    
    public:
    
    // Total substrate concentration
    XVec ST;
    // Total posphorylated substrate concentration
    XVec SpT;    
    
    
    Background() {}
    
    void set_data(RXVec ST, RXVec SpT) {
        this->ST = ST;
        this->SpT = SpT;
        
    }
    
    
    XVec predict(double ST, RXVec params) {
        
        double vbgp = params(0);

        double SpT = ST*vbgp / (1.0 + vbgp);

        double SuT = ST - SpT;

        XVec result = XVec::Zero(2);
        result(0) = SpT;
        result(1) = SuT;
        
        return result;
        
    }
  
    XVec predict_all(RXVec ST, RXVec params) {
        
        int N = ST.size();
        
        XVec SpT_predict = XVec::Zero(N);
        
        for(int i = 0; i < N; i++) {
            
            auto res = predict(ST(i), params);
            
            SpT_predict(i) = res(0);
            
        }
        
        return SpT_predict;
        
    }
    
    
    XVec predict_all(RXVec params) {
        
        return predict_all(ST, params);
        
    }
    
    double loss(RXVec params, RXVec noise_params) {
        
        int N = ST.size();
        
        auto SpT_predict = predict_all(params);
        
        double Sigma2 = noise_params(0);
        double A = noise_params(1);
        double B = noise_params(2);
         
                                
        return (SpT.array().log10() - A*SpT_predict.array().log10() - B).matrix().squaredNorm() / (2*Sigma2*N) + log(Sigma2)/2.0;
    }
    
    
    
};




#endif // BG_HPP