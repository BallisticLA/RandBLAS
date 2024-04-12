#pragma once

#ifndef RandBLAS_EXCEPTIONS_HH
#define RandBLAS_EXCEPTIONS_HH

#include <exception>
#include <cstdarg>
#include <string>

namespace RandBLAS::exceptions {
// Code below copy-pasted from BLAS++ with minimal changes.

#define UNUSED(x) (void)(x)
#define SET_BUT_UNUSED(x) (void)(x)
// ^ Use to suppress compiler warnings for unused variables in functions.

// -----------------------------------------------------------------------------
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const &msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLAS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLAS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
}; 

namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by blas_error_if macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

#if defined(_MSC_VER)
    #define RandBLAS_ATTR_FORMAT(I, F)
#else
    #define RandBLAS_ATTR_FORMAT(I, F) __attribute__((format( printf, I, F )))
#endif

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
    RandBLAS_ATTR_FORMAT(4, 5);

inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... ) {
    UNUSED(condstr);
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}


// -----------------------------------------------------------------------------
// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
inline void abort_if( bool cond, const char* func,  const char* format, ... )
    RandBLAS_ATTR_FORMAT(3, 4);

inline void abort_if( bool cond, const char* func,  const char* format, ... ) {
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );

        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
    }
}

#undef RandBLAS_ATTR_FORMAT

}  // namespace internal

// -----------------------------------------------------------------------------
// internal macros to handle error checks
#if defined(RandBLAS_ERROR_ASSERT)

    // RandBLAS aborts on error
    #define randblas_error_if( cond ) abort_if( cond, __func__, "%s", #cond )
    #define randblas_error_if_msg( cond, ... ) abort_if( cond, __func__, __VA_ARGS__ )

    #define randblas_require( cond ) \
        abort_if( !(cond), __func__, "%s", "("#cond") was required, but did not hold")

#else

    // RandBLAS throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // ex: randblas_error_if( a < b );
    #define randblas_error_if( cond ) \
        RandBLAS::exceptions::internal::throw_if( cond, #cond, __func__ )

    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // ex: randblas_error_if_msg( a < b, "a %d < b %d", a, b );
    #define randblas_error_if_msg( cond, ... ) \
        RandBLAS::exceptions::internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

    #define randblas_require( cond ) \
        RandBLAS::exceptions::internal::throw_if( !(cond), "("#cond") was required, but did not hold", __func__ )

#endif

} // namespace RandBLAS::exceptions

#endif