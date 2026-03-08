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

#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>


namespace RandBLAS::testing {

/** Tests two floating point numbers for approximate equality.
 * See https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *
 * @param[in] A    one number to compare
 * @param[in] B    the second number to compare
 * @param[in] atol is an absolute tolerance that comes into play when
 *                 the values are close to zero
 * @param[in] rtol is a relative tolerance, which should be close to
 *                 epsilon for the given type.
 * @param[inout] str a stream to send a descriptive error message to
 *
 * @returns true if the numbers are atol absolute difference or rtol relative
 *          difference from each other.
 */
template <typename T>
bool approx_equal(T A, T B, std::ostream &str,
    T atol = T(10)*std::numeric_limits<T>::epsilon(),
    T rtol = std::numeric_limits<T>::epsilon())
{
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    T diff_ab = abs(A - B);
    if (diff_ab <= atol)
        return true;

    T max_ab = std::max(abs(B), abs(A));

    if (diff_ab <= max_ab * rtol)
        return true;

    str.precision(std::numeric_limits<T>::max_digits10);

    str << A << " != " << B << " with absDiff=" << diff_ab
        << ", relDiff=" << max_ab*rtol << ", atol=" << atol
        << ", rtol=" << rtol;

    return false;
}

} // end namespace RandBLAS::testing
