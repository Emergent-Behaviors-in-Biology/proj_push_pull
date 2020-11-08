#ifndef PUSHPULL_HPP
#define PUSHPULL_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <Eigen/Dense>

#include "eigen_macros.hpp"

class PushPull {
    
    public:
    
    // Total writer concentration
    XVec WT;
    // Total eraser concentration
    XVec ET;
    // Total substrate concentration
    XVec ST;
    // Total posphorylated substrate centration
    XVec SpT;
    
    
    PushPull() {}
    
    PushPull(RXVec WT, RXVec ET, RXVec ST, RXVec SpT) : 
        WT(WT), ET(ET), ST(ST), SpT(SpT) {}
    
    
    
    
    XVec predict(double WT, double ET, double ST, RXVec params) {
        
        double vWS = params(0);
        double vES = params(1);
        double kWSp = params(2);
        double kESu = params(3);
        double kSp = params(4);
        double kSu = params(5);
                                
        DVec(4) poly_coeffs = DVec(4)::Zero();
        poly_coeffs(0) = -ST; // x^0
        poly_coeffs(1) = 1 + vWS*WT + vES*ET - (vWS+vES)*ST; // x^1
        poly_coeffs(2) = vWS + vES + vWS*vES*(WT+ET-ST); // x^2
        poly_coeffs(3) = vWS*vES; // x^3
      
        DMat(3) companion_mat = DMat(3)::Zero();
        companion_mat(0, 2) = -poly_coeffs(0) / poly_coeffs(3);
        companion_mat(1, 2) = -poly_coeffs(1) / poly_coeffs(3);
        companion_mat(2, 2) = -poly_coeffs(2) / poly_coeffs(3);
        
        for(int i = 1; i < 3; i++) {
            companion_mat(i, i-1) = 1.0;
        }
        
//         py::print(companion_mat);
//         py::print(companion_mat.eigenvalues());
        
        auto evals = companion_mat.eigenvalues();
                
        if((evals.imag().array().cwiseAbs() > 1e-8).count() > 0) {
            py::print("Imaginary Roots:", evals);           
        } else if ((evals.real().array() > 0.0).count() > 1) {
            py::print("Multiple Positive Roots:", evals);
        }
        
        double Sf = evals.real().maxCoeff();
                
        double W = WT/(1+vWS*Sf);
        double E =  ET/(1+vES*Sf);  

        double pWSu = vWS*W/(1+vWS*W+vES*E);
        double pESp = vES*E/(1+vWS*W+vES*E);

        double SpT = ST*(kWSp*pWSu + kSp)/ (kWSp*pWSu + kESu*pESp + kSp + kSu);

        double SuT = ST - SpT;

        double Sp = SpT/(1+vWS*W+vES*E);
        double Su = SuT/(1+vWS*W+vES*E);

        double WSu = pWSu*SuT;
        double WSp = WT - W - WSu;

        double ESp = pESp*SpT;
        double ESu = ET -E - ESp;
                
        XVec result = XVec::Zero(10);
        result(0) = SpT;
        result(1) = SuT;
        result(2) = W;
        result(3) = E;
        result(4) = Sp;
        result(5) = Su;
        result(6) = WSp;
        result(7) = WSu;
        result(8) = ESp;
        result(9) = ESu;
        
        return result;
        
    }
    
    double loss(RXVec params) {
        
        int N = WT.size();
        
        double loss = 0.0;
        for(int i = 0; i < N; i++) {
            
            auto res = predict(WT(i), ET(i), ST(i), params);
            
            double SpT_predict = res(0);
            
            loss += pow(SpT(i)/ST(i) - SpT_predict/ST(i), 2.0);
        }
        
        loss /= N;
        
        return loss;
    }
    
};



#endif // PUSHPULL_HPP