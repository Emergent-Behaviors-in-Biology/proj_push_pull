#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "eigen_macros.hpp"

#include "thermo_models.hpp"



PYBIND11_MODULE(push_pull, m) {
    
    
    py::class_<ThermoModel>(m, "ThermoModel")
        .def("predict_all", &ThermoModel::predict_all)
        .def("loss", &ThermoModel::loss, py::arg("target"), py::arg("data"), py::arg("params"), py::arg("use_log")=false)
        .def("loss_mixture", &ThermoModel::loss_mixture);
    
    py::class_<PushAmp, ThermoModel>(m, "PushAmp")
        .def(py::init<>())
        .def("predict", &PushAmp::predict);
    
    py::class_<Background, ThermoModel>(m, "Background")
        .def(py::init<>())
        .def("predict", &Background::predict);
    

};
