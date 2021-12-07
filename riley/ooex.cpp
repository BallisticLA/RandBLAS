#include <pybind11/pybind11.h>
#include "pybind11/stl.h"

namespace py = pybind11;

struct Pet {
    Pet(const std::string &name) : name(name) {}
    void setName(const std::string &name_) {name = name_;}
    const std::string &getName() const {return name;}

    std::string name;
};

PYBIND11_MODULE(ooex, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def_readwrite("name", &Pet::name)
        .def("__repr__",
            [](const Pet &a) {
                return "<ooex.Pet named '" + a.name + "'>";
            }
        );
}

