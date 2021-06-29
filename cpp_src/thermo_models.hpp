#ifndef THERMO_HPP
#define THERMO_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "eigen_macros.hpp"

class ThermoModel {
    
    public:
    
    // predict phosphorylated substrate for one data point
    virtual XVec predict(RXVec data, RXVec params) = 0;
    
    // predict for multiple data points
    XMat predict_all(RXMat data, RXVec params) {
        
        XMat prediction = XMat::Zero(data.rows(), 7);
        for(int i = 0; i < data.rows(); i++) {
            XVec d = data.row(i);
            prediction.row(i) = predict(d, params);
        }
        
        return prediction;
    }
    
    // Calculate loss
    double loss(RXVec target, RXMat data, RXVec params, bool use_log=false) {
        
        XVec prediction = predict_all(data, params);
        
        if(use_log) {
            return 0.5 * (target.array().log() - prediction.array().log()).matrix().squaredNorm();
        } else {
            return 0.5 * (target - prediction).squaredNorm();
        }
                   
    };
    
    
    
    double loss_mixture(RXVec target, RXMat data, RXVec model_params, RXVec noise_params) {
        
        XVec prediction = predict_all(data, model_params).array().log().matrix();
        
        XMat data_empty = data;
        data_empty.col(0) = XVec::Zero(data.rows());
        
        XVec prediction_empty = predict_all(data_empty, model_params).array().log().matrix();
        
//         py::print((prediction-prediction_empty).norm());
        
        double rho = noise_params(0);
        
        XVec signal = (target.array().log() - prediction.array()).pow(2.0).matrix();
        signal = -0.5*signal;
        
        
        XVec empty = (target.array().log() - prediction_empty.array()).pow(2.0).matrix();
        empty = -0.5*empty;
 
        
        return -((1-rho)*signal.array().exp() + rho*empty.array().exp() + 1e-16).log().sum();
            

    };
};


class PushAmp: public ThermoModel {
    
    
    public:
    
    
    XVec predict(RXVec data, RXVec params) {
       
        double WT = data(0);
        double ST = data(1);
        
        double vSp = params(0);
        double vWSp = params(1);
        double alphaWS = params(2);
        
                                
        DVec(3) poly_coeffs = DVec(3)::Zero();
        poly_coeffs(0) = -ST/alphaWS; // x^0
        poly_coeffs(1) = 1.0 + (WT-ST)/alphaWS; // x^1
        poly_coeffs(2) = 1.0; // x^2
      
        DMat(2) companion_mat = DMat(2)::Zero();
        companion_mat(0, 1) = -poly_coeffs(0) / poly_coeffs(2);
        companion_mat(1, 1) = -poly_coeffs(1) / poly_coeffs(2);
        
        for(int i = 1; i < 2; i++) {
            companion_mat(i, i-1) = 1.0;
        }
        
//         py::print(companion_mat);
//         py::print(companion_mat.eigenvalues());
        
        auto evals = companion_mat.eigenvalues();
                
        if((evals.imag().array().cwiseAbs()/evals.real().array().cwiseAbs() > 1e-15).count() > 0) {
            py::print("Imaginary Roots:", evals); 
            py::print("Params:", params);
        } else if ((evals.real().array() > 0.0).count() > 1) {
            py::print("Multiple Positive Roots:", evals);
            py::print("Params:", params);
        }
        
        double Sf = evals.real().maxCoeff() * alphaWS;
                
        double W = WT/(1+Sf/alphaWS);

        double pWSu = W/alphaWS/(1+W/alphaWS);

        double SpT = ST*(vWSp*pWSu + vSp)/ (vWSp*pWSu + vSp + 1);

//         return SpT;
        
        double SuT = ST - SpT;

        double Sp = SpT/(1+W/alphaWS);
        double Su = SuT/(1+W/alphaWS);

        double WSu = pWSu*SuT;
        double WSp = WT - W - WSu;
                
        XVec result = XVec::Zero(7);
        result(0) = SpT;
        result(1) = SuT;
        result(2) = W;
        result(3) = Sp;
        result(4) = Su;
        result(5) = WSp;
        result(6) = WSu;
        
        return result;
         
    }
   
    
};


class Background: public ThermoModel {
    
    
    public:
    
    
    XVec predict(RXVec data, RXVec params) {
       
        
        double ST = data(0);
        
        double vbgp = params(0);

        double SpT = ST*vbgp / (1.0 + vbgp);

//         return SpT;
        
        double SuT = ST - SpT;
        
        XVec result = XVec::Zero(2);
        result(0) = SpT;
        result(1) = SuT;
        
        return result;
        
        
         
    }
   
    
};



#endif // THERMO_HPP