#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "eigen_macros.hpp"
#include "push_pull.hpp"
// #include <string>

// class Test {
    
//     public:
    
//     int a;
    
//     Test(int a) : a(a) {}
    
//     int geta(){
        
//         return a;
//     }
// };

// struct Pet {
//     Pet(const std::string &name) : name(name) { }
//     void setName(const std::string &name_) { name = name_; }
//     const std::string &getName() const { return name; }

//     std::string name;
// };

// int add(int i, int j) {
//     return i + j;
// }


PYBIND11_MODULE(push_pull_amp, m) {
    
//     py::class_<Test>(m, "Test")
//         .def("a", &Test::a)
//         .def(py::init<int>())
//         .def("geta", &Test::geta);
    
//     m.def("add", &add);
    
//     py::class_<Pet>(m, "Pet")
//         .def(py::init<const std::string &>())
//         .def("setName", &Pet::setName)
//         .def("getName", &Pet::getName);
    
// };
    
        py::class_<PushPull>(m, "PushPull")
            .def_readwrite("WT", &PushPull::WT)
            .def_readwrite("ET", &PushPull::ET)
            .def_readwrite("ST", &PushPull::ST)
            .def_readwrite("SpT", &PushPull::SpT)
            .def(py::init())
            .def(py::init<RXVec, RXVec, RXVec, RXVec>())
            .def("predict", &PushPull::predict);

//     py::class_<CellComplex>(m, "CellComplex")
//         .def_readonly("dim", &CellComplex::dim)
//         .def_readonly("ncells", &CellComplex::ncells)
//         .def_readonly("ndcells", &CellComplex::ndcells)
//         .def_readonly("dcell_range", &CellComplex::dcell_range)
//         .def_readonly("regular", &CellComplex::regular)
//         .def_readonly("oriented", &CellComplex::oriented)
//         .def(py::init<int, bool, bool>(), py::arg("dim"), py::arg("regular")=true, py::arg("oriented")=false)
//        . def(py::init<CellComplex>())
//         .def("add_cell", (void (CellComplex::*)(int, int, std::vector<int>&, std::vector<int>&)) &CellComplex::add_cell)
//         .def("get_dim", &CellComplex::get_dim)
//         .def("get_label", &CellComplex::get_label)
//         .def("get_facets", &CellComplex::get_facets)
//         .def("get_cofacets", &CellComplex::get_cofacets)
//         .def("get_coeffs", &CellComplex::get_coeffs)
//         .def("get_faces", &CellComplex::get_faces, py::arg("alpha"), py::arg("target_dim")=-1)
//         .def("get_cofaces", &CellComplex::get_cofaces, py::arg("alpha"), py::arg("target_dim")=-1)
//         .def("get_star", &CellComplex::get_star, py::arg("alpha"), py::arg("dual"), py::arg("target_dim")=-1)
//         .def("get_labels", &CellComplex::get_labels)
//         .def("make_compressed", &CellComplex::make_compressed)
//         .def("construct_cofacets", &CellComplex::construct_cofacets);


//     m.def("prune_cell_complex", &prune_cell_complex);
//     m.def("prune_cell_complex_map", &prune_cell_complex_map);
//     m.def("prune_cell_complex_sequential", &prune_cell_complex_sequential,
//           py::arg("comp"), py::arg("priority"), py::arg("preserve"), py::arg("preserve_stop")=true, py::arg("allow_holes")=false, py::arg("threshold")=0.0, py::arg("target_dim")=-1);
//     m.def("prune_cell_complex_sequential_surface", &prune_cell_complex_sequential_surface,
//           py::arg("comp"), py::arg("priority"), py::arg("preserve"), py::arg("surface"), py::arg("preserve_stop")=true, py::arg("allow_holes")=false, py::arg("threshold")=0.0, py::arg("target_dim")=-1, py::arg("verbose")=false);
//     m.def("check_boundary_op", &check_boundary_op,
//           "Checks the boundary operator of a complex to ensure that \\delta_d\\delta_(d-1) = 0 for each cell.");
// //         m.def("get_boundary", &get_boundary);


};
