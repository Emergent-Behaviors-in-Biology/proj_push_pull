#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "eigen_macros.hpp"

#include "background.hpp"
#include "push.hpp"
#include "push_pull.hpp"


// int add(int i, int j) {
//     return i + j;
// }


PYBIND11_MODULE(push_pull_amp, m) {
    
//     m.def("add", &add);
    
    py::class_<Background>(m, "Background")
        .def_readwrite("ST", &Background::ST)
        .def_readwrite("SpT", &Background::SpT)
        .def_readwrite("N_data", &Background::N_data)
        .def(py::init())
        .def("set_data", &Background::set_data)
        .def("predict", &Background::predict)
        .def("predict_all", (XVec (Background::*)(RXVec)) &Background::predict_all)
        .def("predict_all", (XVec (Background::*)(RXVec, RXVec)) &Background::predict_all)
        .def("loss", &Background::loss);
    
    py::class_<Push>(m, "Push")
        .def_readwrite("WT", &Push::WT)
        .def_readwrite("ST", &Push::ST)
        .def_readwrite("SpT", &Push::SpT)
        .def_readwrite("N_data", &Push::N_data)
        .def(py::init())
        .def("set_data", &Push::set_data)
        .def("predict", &Push::predict)
        .def("predict_all", (XVec (Push::*)(RXVec)) &Push::predict_all)
        .def("predict_all", (XVec (Push::*)(RXVec, RXVec, RXVec)) &Push::predict_all)
        .def("loss", &Push::loss)
        .def("predict_grad", &Push::predict_grad)
        .def("predict_grad_all", (std::tuple<XVec, XMat> (Push::*)(RXVec)) &Push::predict_grad_all)
        .def("predict_grad_all", (std::tuple<XVec, XMat> (Push::*)(RXVec, RXVec, RXVec, RXVec)) &Push::predict_grad_all)
        .def("loss_grad", &Push::loss_grad);

    py::class_<PushPull>(m, "PushPull")
        .def_readwrite("WT", &PushPull::WT)
        .def_readwrite("ET", &PushPull::ET)
        .def_readwrite("ST", &PushPull::ST)
        .def_readwrite("SpT", &PushPull::SpT)
        .def(py::init())
        .def("set_data", &PushPull::set_data)
        .def("set_hyperparams", &PushPull::set_hyperparams)
        .def("predict", &PushPull::predict)
        .def("predict_all", (XVec (PushPull::*)(RXVec)) &PushPull::predict_all)
        .def("predict_all", (XVec (PushPull::*)(RXVec, RXVec, RXVec, RXVec, RXVec)) &PushPull::predict_all)
        .def("loss", &PushPull::loss);
    
    
//     py::class_<NoiseModel>(m, "NoiseModel")
//         .def(py::init<RXVec, RXVec, int, int>());
    
//     py::class_<NoisyPushPull>(m, "NoisyPushPull")
//         .def_readwrite("WT", &NoisyPushPull::WT)
//         .def_readwrite("ET", &NoisyPushPull::ET)
//         .def_readwrite("ST", &NoisyPushPull::ST)
//         .def_readwrite("SpT", &NoisyPushPull::SpT)
//         .def(py::init<RXVec, RXVec, RXVec, RXVec, NoiseModel, NoiseModel, NoiseModel>())
//         .def("predict", &NoisyPushPull::predict)
//         .def("predict_all", &NoisyPushPull::predict_all)
//         .def("loss", &NoisyPushPull::loss);
    
    

};
