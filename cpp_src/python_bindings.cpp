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
            .def(py::init<RXVec, RXVec, RXVec, RXVec>())
            .def("predict", &PushPull::predict);

};
