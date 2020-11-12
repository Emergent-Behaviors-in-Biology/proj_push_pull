#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "eigen_macros.hpp"
#include "push_pull.hpp"


// int add(int i, int j) {
//     return i + j;
// }


PYBIND11_MODULE(push_pull_amp, m) {
    
//     m.def("add", &add);
    

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
