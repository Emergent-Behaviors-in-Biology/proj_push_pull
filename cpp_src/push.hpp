#ifndef PUSH_HPP
#define PUSH_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cmath>
#include <Eigen/Dense>

#include "eigen_macros.hpp"

class Push {
    
    public:
    
    // Total writer concentration
    XVec WT;
    // Total substrate concentration
    XVec ST;
    // Total posphorylated substrate concentration
    XVec SpT;    
    
    
    Push() {}
    
    void set_data(RXVec WT, RXVec ST, RXVec SpT) {
        this->WT = WT;
        this->ST = ST;
        this->SpT = SpT;
        
    }
    
    
    XVec predict(double WT, double ST, RXVec params) {
        
        double alphaWS = params(0);
        double vWSp = params(1);
        double vSp = params(2);
                                
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
  
    XVec predict_all(RXVec WT, RXVec ST, RXVec params) {
        
        int N = WT.size();
        
        XVec SpT_predict = XVec::Zero(N);
        
        for(int i = 0; i < N; i++) {
            
            auto res = predict(WT(i), ST(i), params);
            
            SpT_predict(i) = res(0);
            
        }
        
        return SpT_predict;
        
    }
    
    
    XVec predict_all(RXVec params) {
        
        return predict_all(WT, ST, params);
        
    }
    
    double loss(RXVec params, RXVec noise_params) {
        
        int N = WT.size();
        
        auto SpT_predict = predict_all(params);
        
        double Sigma2 = noise_params(0);
        double A = noise_params(1);
        double B = noise_params(2);
         
                                
        return (SpT.array().log10() - A*SpT_predict.array().log10() - B).matrix().squaredNorm() / (2*Sigma2*N) + log(Sigma2)/2.0;
    }
    
    
      
    std::tuple<XVec, XMat> predict_grad(double WT, double ST, RXVec params) {
        
        double vWS = params(0);
        double vWSp = params(1);
        double vSp = params(2);
        
        XVec cons = predict(WT, ST, params);
        double Sf = cons(3) + cons(4);
        double W = cons(2);
        
        double pWSu = vWS*W/(1+vWS*W);
        
        // Only gradient wrt to SpT, but could return more in principle
        XMat grad = XMat::Zero(1, 3);
        
        double dSf_dvWS = -Sf*(Sf + WT - ST)/(2*vWS*Sf + 1 + vWS*(WT-ST));
        double dW_dvWS = -WT/pow(1+vWS*Sf, 2.0)*(Sf + vWS*dSf_dvWS);
        double dpWSu_dvWS = 1/pow(1+vWS*W, 2.0) * (W + vWS*dW_dvWS);
        
        double dSpT_dvWS = ST*vWSp/pow(vWSp*pWSu + vSp + 1, 2.0) * dpWSu_dvWS;
        double dSpT_dvWSp = ST*pWSu/pow(vWSp*pWSu + vSp + 1, 2.0);
        double dSpT_dvSp = ST/pow(vWSp*pWSu + vSp + 1, 2.0);
       
        grad(0,0) = dSpT_dvWS;
        grad(0,1) = dSpT_dvWSp;
        grad(0,2) = dSpT_dvSp;
        
        return std::make_tuple(cons, grad);
        
    }
    
    
    std::tuple<XVec, XMat> predict_grad_all(RXVec WT, RXVec ST, RXVec SpT, RXVec params) {
        
        int N = WT.size();
        
        XVec SpT_predict = XVec::Zero(N);
        XMat SpT_predict_grad = XMat::Zero(N, 3);
        
        for(int i = 0; i < N; i++) {
            
            XVec cons;
            XMat grad;
            
            auto res = predict_grad(WT(i), ST(i), params);
            
            std::tie(cons, grad) = res;
            
            SpT_predict(i) = cons(0);
            SpT_predict_grad.row(i) = grad.row(0);
            
        }
        
        return std::make_tuple(SpT_predict, SpT_predict_grad);
        
    }
    
    
    std::tuple<XVec, XMat> predict_grad_all(RXVec params) {
        
        return predict_grad_all(WT, ST, SpT, params);
        
    }
    
    std::tuple<double, XVec> loss_grad(RXVec params, RXVec noise_params) {
        
        
        int N = WT.size();
        
        
        XVec SpT_predict;
        XMat SpT_predict_grad;
            
        auto result = predict_grad_all(params);
        std::tie(SpT_predict, SpT_predict_grad) = result;
                
        double Sigma2 = noise_params(0);
        double A = noise_params(1);
        double B = noise_params(2);
        
        XVec residual = (SpT.array().log10() - A*SpT_predict.array().log10() - B).matrix();
         
        double loss = residual.squaredNorm() / (2*Sigma2*N) + log(Sigma2)/2.0;
        
        XVec grad = XVec::Zero(6);
                        
        auto temp = SpT_predict_grad.array().colwise()*(1/SpT_predict.array() * residual.array());
        
        grad.segment(0, 3) = -A/(Sigma2*N*log(10))*temp.colwise().sum();
        
//         py::print("large?", grad(2), SpT_predict.minCoeff(), residual.cwiseAbs().maxCoeff(), SpT_predict_grad.col(2).cwiseAbs().maxCoeff());
        
        grad(3) = -residual.squaredNorm() / (2*pow(Sigma2, 2.0)*N) + 1/(2.0*Sigma2);
        grad(4) = -SpT_predict.array().log10().matrix().dot(residual) / (Sigma2*N);
        grad(5) = -residual.sum() / (Sigma2*N);
            
            
                                
        return std::make_tuple(loss, grad);
        
    }
    
    
    
    
};




#endif // PUSH_HPP