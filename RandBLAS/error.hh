#ifndef RandBLAS_error_h
#define RandBLAS_error_h

#include <sstream>
#include <stdexcept>

#define RB_RUNTIME_ERROR(_msg)                                      \
{                                                                   \
    std::ostringstream oss;                                         \
    oss << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl    \
        << "" _msg << std::endl;                                    \
    throw std::runtime_error(oss.str());                            \
}

#endif
