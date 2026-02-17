// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <exception>
#include <cstdarg>
#include <string>

// Mose code in this file is adapted from BLAS++.

namespace RandBLAS {

// -----------------------------------------------------------------------------
/// Minimalist exception class for RandBLAS errors.
///
/// @verbatim embed:rst:leading-slashes
///  These are typically triggered by statements of the form ``randblas_require(cond)``
///  where ``cond`` was false. The curious can examine RandBLAS/exceptions.hh
///  for the definition of ``randblas_require``.
///  
///  The vast majority of errors thrown by RandBLAS are due to invalid
///  parameters for matrix dimensions or strides. These error messages 
///  only say what went wrong; they offer no suggestions on how to fix
///  what went wrong or why it might have happened. Please get in touch
///  with us on GitHub if you've having trouble understanding or resolving
///  an error.
///
/// @endverbatim
///
class Error: public std::exception {
private:
    std::string msg_;
public:
    // Constructs RandBLAS error
    Error():
        std::exception()
    {}

    // ---------------------------------------
    /// Constructs error with message
    Error( std::string const &msg ): Error() { msg_ = msg; }

    // Constructs RandBLAS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        Error(std::string(msg) + ", in function " + func)
    {}

    // ---------------------------------------------------------------
    /// Returns a C-string representation of this error's message.
    /// It's common to wrap this function's return value with std::string.
    /// For example ...
    ///
    /// @verbatim embed:rst:leading-slashes
    /// .. code:: c++
    ///
    ///     try {
    ///         /* do something that might raise a RandBLAS error */
    ///         randblas_require(1 < 0);
    ///     } catch (RandBLAS::Error &e) {
    ///         std::string message{e.what()};
    ///         /* inspect the message */
    ///         std::cout << message << std::endl;
    ///     }
    ///
    /// @endverbatim
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }
}; 

} // end namespace RandBLAS

namespace RandBLAS::exceptions {

#define UNUSED(x) (void)(x)
#define SET_BUT_UNUSED(x) (void)(x)
// ^ Use to suppress compiler warnings for unused variables in functions.

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

#define RandBLAS_ERROR_MESSAGE_SIZE 256

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
        char buf[RandBLAS_ERROR_MESSAGE_SIZE];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

#undef RandBLAS_ATTR_FORMAT

}  // namespace internal


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


} // namespace RandBLAS::exceptions
