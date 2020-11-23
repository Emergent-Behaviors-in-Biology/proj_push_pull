#ifndef PUSHPULL_HPP
#define PUSHPULL_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cmath>
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
    
//     double sigma2;
    
    
    PushPull() {}
    
    void set_data(RXVec WT, RXVec ET, RXVec ST, RXVec SpT) {
        this->WT = WT;
        this->ET = ET;
        this->ST = ST;
        this->SpT = SpT;
        
    }
    
    void set_hyperparams(RXVec hyper) {
//         sigma2 = hyper(0);
    }
    
    
    XVec predict(double WT, double ET, double ST, RXVec params) {
        
        double vWS = params(0);
        double vES = params(1);
        double vWSp = params(2);
        double vSp = params(3);
        double vSu = params(4);
                                
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
                
        if((evals.imag().array().cwiseAbs()/evals.real().array().cwiseAbs() > 1e-15).count() > 0) {
            py::print("Imaginary Roots:", evals); 
            py::print("Params:", params);
        } else if ((evals.real().array() > 0.0).count() > 1) {
            py::print("Multiple Positive Roots:", evals);
            py::print("Params:", params);
        }
        
        double Sf = evals.real().maxCoeff();
                
        double W = WT/(1+vWS*Sf);
        double E =  ET/(1+vES*Sf);  

        double pWSu = vWS*W/(1+vWS*W+vES*E);
        double pESp = vES*E/(1+vWS*W+vES*E);

        double SpT = ST*(vWSp*pWSu + vSp)/ (vWSp*pWSu + pESp + vSp + vSu);
//         double SpT = ST*(vWSp*pWSu )/ (vWSp*pWSu + pESp);

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
    
    XVec predict_all(RXVec WT, RXVec ET, RXVec ST, RXVec SpT, RXVec params) {
        
        int N = WT.size();
        
        XVec SpT_predict = XVec::Zero(N);
        
        for(int i = 0; i < N; i++) {
            
            auto res = predict(WT(i), ET(i), ST(i), params);
            
            SpT_predict(i) = res(0);
            
        }
        
        return SpT_predict;
        
    }
    
    XVec predict_all(RXVec params) {
        
        return predict_all(WT, ET, ST, SpT, params);
        
    }
    
    double loss(RXVec params) {
        
        int N = WT.size();
        
        auto SpT_predict = predict_all(params);
                                
        return (SpT.array().log10() - SpT_predict.array().log10()).matrix().squaredNorm() / N;
    }
    
};


// class NoiseModel {
    
//     public:
    
//     int nbins_anti;
//     int nbins_gfp;

//     double min_anti;
//     double max_anti;
    
//     double min_gfp;
//     double max_gfp;

//     XMat prob_anti_given_gfp;
//     XVec d_gfp;
        
    
//     NoiseModel(RXVec anti, RXVec gfp, int nbins_anti, int nbins_gfp) : nbins_anti(nbins_anti), nbins_gfp(nbins_gfp) {
            
//         min_anti = anti.minCoeff();
//         max_anti = anti.maxCoeff();
        
//         min_gfp = gfp.minCoeff();
//         max_gfp = gfp.maxCoeff();
                
//         // Calculate differential for each bin along gfp axis
//         d_gfp = XVec::Zero(nbins_gfp);
//         for(int i = 0; i < nbins_gfp; i++) {
            
//             double low = log(min_gfp) + (log(max_gfp) - log(min_gfp)) * i/nbins_gfp;
//             double high = log(min_gfp) + (log(max_gfp) - log(min_gfp)) * (i+1)/nbins_gfp;
            
//             d_gfp(i) = exp(high) - exp(low);
            
//         }
          
//         // Calculate conditional probability of antibody given gfp
//         prob_anti_given_gfp = XMat::Zero(nbins_gfp, nbins_anti);
        
//         // First create histogram of counts
//         for(int i = 0; i < anti.size(); i++) {
//             int bin_anti = get_bin_anti(anti(i));
//             int bin_gfp = get_bin_gfp(gfp(i));
                        
//             prob_anti_given_gfp(bin_gfp, bin_anti) += 1.0;
            
//         }
         
//         // Normalize each column so that they integrate to one (prob_anti_given_gfp.tranpose() * d_gfp = [1, 1, 1, 1, ...] )
//         for(int i = 0;  i < nbins_anti; i++) {
//             double sum = prob_anti_given_gfp.col(i).sum();
//             if (sum > 0.0) {
//                 prob_anti_given_gfp.col(i) = prob_anti_given_gfp.col(i) / sum;
//                 prob_anti_given_gfp.col(i) = prob_anti_given_gfp.array().col(i) / d_gfp.array();
//             }
            
//         }
                
//     }
    
//     int get_bin_anti(double a) {
        
//         // need to check lowe and upper bounds first
        
//         return int(fmin((log(a)-log(min_anti)) / (log(max_anti)-log(min_anti)) * nbins_anti, nbins_anti-1.0)); 
//     }
    
//     int get_bin_gfp(double g) {
//         return int(fmin((log(g)-log(min_gfp)) / (log(max_gfp)-log(min_gfp)) * nbins_gfp, nbins_gfp-1.0)); 
//     }
    
//     double get_val_anti(int i) {
//         // Geometric mean of bin edges
//         return exp((log(max_anti)-log(min_anti)) * (i+0.5) / nbins_anti);
//     }
    
//     double get_val_gfp(int i) {
//         // Geometric mean of bin edges
//         return exp((log(max_gfp)-log(min_gfp)) * (i+0.5) / nbins_gfp);
//     }
    
    
    
    
// };

// class NoisyPushPull: public PushPull {
    
//     public:
    
// //     // Total writer concentration
// //     XVec WT;
// //     // Total eraser concentration
// //     XVec ET;
// //     // Total substrate concentration
// //     XVec ST;
// //     // Total posphorylated substrate centration
// //     XVec SpT;
    
//     NoiseModel nm_WT;
//     NoiseModel nm_ET;
//     NoiseModel nm_ST;
          
//     NoisyPushPull(RXVec WT, RXVec ET, RXVec ST, RXVec SpT, NoiseModel nm_WT, NoiseModel nm_ET, NoiseModel nm_ST) :
//         PushPull(WT, ET, ST, SpT), nm_WT(nm_WT), nm_ET(nm_ET), nm_ST(nm_ST) {}
    
    
//     double loss(RXVec params) {
        
//         int N = WT.size();
                
//         double loss = 0.0;
//         for(int i = 0; i < N; i++) {
            
// //             bool over = false;
            
//             int bin_anti_WT = nm_WT.get_bin_anti(WT(i));
//             int bin_anti_ET = nm_ET.get_bin_anti(ET(i));
//             int bin_anti_ST = nm_ST.get_bin_anti(ST(i));
            
//             double total_prob = 0.0;
            
//             for(int i_WT = 0; i_WT < nm_WT.nbins_gfp; i_WT++) {
//                 double weight_WT = nm_WT.prob_anti_given_gfp(i_WT, bin_anti_WT);
//                 if(weight_WT == 0.0) {
//                     continue;
//                 }
                
//                 double gfp_WT = nm_WT.get_val_gfp(i_WT);
                
//                 for(int i_ET = 0; i_ET < nm_ET.nbins_gfp; i_ET++) {
//                     double weight_ET = nm_ET.prob_anti_given_gfp(i_ET, bin_anti_ET);
//                     if(weight_ET == 0.0) {
//                         continue;
//                     }
                    
//                     double gfp_ET = nm_ET.get_val_gfp(i_ET);

//                     for(int i_ST = 0; i_ST < nm_ST.nbins_gfp; i_ST++) {
//                         double weight_ST = nm_ST.prob_anti_given_gfp(i_ST, bin_anti_ST);
//                         if(weight_ST == 0.0) {
//                             continue;
//                         }
                        
//                         double gfp_ST = nm_ST.get_val_gfp(i_ST);
                        
                        
//                         auto res = predict(gfp_WT, gfp_ET, gfp_ST, params);
                        
//                         double SpT_predict = res(0);
                        
// //                         if(i == 778) {
// //                             py::print(weight_WT, weight_ET, weight_ST, exp(-pow(SpT(i)/ST(i) - SpT_predict/ST(i), 2.0)), SpT(i), ST(i), SpT_predict);
// //                         }
            
//                         // Gaussian noise model for phosphorylated substrate
                        
// //                         if(SpT(i)/1e3/gfp_ST > 1.0) {
// // //                             py::print(SpT(i)/gfp_ST);
// //                             over = true;
                            
// //                         }
                        
//                         double a = 0.3/0.4;
//         double b = 2.5 - 0.3/0.4 * 4.0;
//         double sigma2 = (0.3*0.4-0.3*0.3)/0.4;
//                         total_prob += weight_WT*weight_ET*weight_ST*exp(-pow(log10(SpT(i)) -a*log10(SpT_predict) - b, 2.0) / (2.0*sigma2))/SpT_predict;
//                     }
//                 }
         
//             }
            
// //             if(over) {
                                
// //                 py::print(i, total_prob, SpT(i), 10*ST(i));
// //             }
                        
//             loss += -log(total_prob);
            
//              if(total_prob == 0.0) {
                
//                 py::print(i, total_prob, SpT(i), py::arg("flush")=true);
//             } 
                        
            
//         }
        
//         loss /= N;
        
//         return loss;
//     }
    
// };



#endif // PUSHPULL_HPP