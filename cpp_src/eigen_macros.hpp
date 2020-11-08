#ifndef EIGEN_MACROS_HPP
#define EIGEN_MACROS_HPP
  
#include <Eigen/Dense>

// Variable length vectors and and matrices
typedef Eigen::VectorXd XVec;
typedef Eigen::MatrixXd XMat;
typedef Eigen::VectorXcd XcVec;
typedef Eigen::MatrixXcd XcMat;

// Reference type variable length vectors and and matrices
// Useful for pybind interfacing
typedef Eigen::Ref<XVec > RXVec;
typedef Eigen::Ref<XMat > RXMat;

// Map to interface raw array buffers with eigen vectors
typedef Eigen::Map<XVec > XVecMap;
typedef Eigen::Map<XMat > XMatMap;

# define DVec(DIM) Eigen::Matrix<double, DIM, 1>
# define DMat(DIM) Eigen::Matrix<double, DIM, DIM>

#endif // EIGEN_MACROS_HPP