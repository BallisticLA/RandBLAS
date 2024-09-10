import re

def transform_param_line(line):
    # Match the parameter line with the format "///      param_name - [direction]"
    match = re.match(r'///\s+(\w+)\s+-\s+\[(.+)\]', line)
    if match:
        param_name = match.group(1)
        direction = match.group(2)
        return f"///  @param[{direction}] {param_name}"
    return line

def transform_math_expressions(text):
    # Use regex to find and replace :math:`<expr>` with \math{<expr>}
    return re.sub(r':math:`([^`]+)`', r'\\math{\1}', text)

def transform_bullet_points(text, param_name):
    # Calculate the offset for bullet points
    offset = len(param_name) + 2
    bullet_point_pattern = re.compile(r'(///\s+)[*-]\s+')
    
    def replace_bullet(match):
        leading_slashes = match.group(1)
        return leading_slashes + ' ' * (offset - len(leading_slashes)) + '- '
    
    return bullet_point_pattern.sub(replace_bullet, text)

def transform_documentation(doc):
    # Remove leading @verbatim and trailing @endverbatim lines
    doc = re.sub(r'///\s*@verbatim.*\n', '', doc)
    doc = re.sub(r'///\s*@endverbatim.*\n', '', doc)
    
    lines = doc.split('\n')
    transformed_lines = []
    param_name = None
    
    for line in lines:
        if re.match(r'///\s+\w+\s+-\s+\[.+\]', line):
            param_name = re.match(r'///\s+(\w+)\s+-\s+\[.+\]', line).group(1)
            transformed_lines.append(transform_param_line(line))
        elif param_name and re.match(r'///\s+[*-]\s+', line):
            # Transform bullet points and math expressions within the bullet points
            transformed_line = transform_bullet_points(line, param_name)
            transformed_line = transform_math_expressions(transformed_line)
            transformed_lines.append(transformed_line)
        else:
            transformed_lines.append(transform_math_expressions(line))
    
    return '\n'.join(transformed_lines)


def example_rstfullparams():
    params = \
r"""
/// @verbatim embed:rst:leading-slashes
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(C)`.
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\mat(A)) = \mat(A)`.
///       * If :math:`\opA` = Trans, then :math:`\op(\mat(A)) = \mat(A)^T`.
///
///      opB - [in]
///       * If :math:`\opB` = NoTrans, then :math:`\op(\mtxB) = \mtxB`.
///       * If :math:`\opB` = Trans, then :math:`\op(\mtxB) = \mtxB^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(C)`.
///       * The number of rows in :math:`\op(\mat(A))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(C)`.
///       * The number of columns in :math:`\op(\mtxB)`.
///
///      k - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mat(A))`
///       * The number of rows in :math:`\op(\mtxB)`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A`. 
///
///      B - [in]
///       * A RandBLAS sparse matrix object.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`C` need not be set on input.
///
///      C - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(C)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(C)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldc - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(C)` when reading from :math:`C`.
///
/// @endverbatim
"""
    return params


if __name__ == '__main__':
    out = transform_documentation(example_rstfullparams())
    print(out)
    print()
