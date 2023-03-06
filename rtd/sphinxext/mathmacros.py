r"""Sphinx extension providing a new directive ``mathmacro``
============================================================

This extension has to be added after the other math extension since it
redefined the math directive and the math role. For example, like this
(in the conf.py file)::

  extensions = [
      'sphinx.ext.autodoc', 'sphinx.ext.doctest',
      'sphinx.ext.mathjax',
      'sphinx.ext.viewcode', 'sphinx.ext.autosummary',
      'numpydoc',
      'fluiddoc.mathmacro']


Here, I show how to use a new mathmacro substitution directive in
reStructuredText. I think even this small example demonstrates that it
is useful.

Example
-------

First some math without math macros.  Let's take the example of the incompressible
Navier-Stokes equation:

.. math::
  \mbox{D}_t \textbf{v} =
  -\boldsymbol{\nabla} p + \nu \boldsymbol{\nabla} ^2 \textbf{v}.

where :math:`\mbox{D}_t` is the convective derivative, :math:`\textbf{v}` the
velocity, :math:`\boldsymbol{\nabla}` the nabla operator, :math:`\nu` the
viscosity and :math:`\boldsymbol{\nabla}^2` the Laplacian operator.

The code for this is:

.. code-block:: rest

  .. math::
    \mbox{D}_t \textbf{v} =
    -\boldsymbol{\nabla} p + \nu \boldsymbol{\nabla} ^2 \textbf{v}.

  where :math:`\mbox{D}_t` is the convective derivative, :math:`\textbf{v}` the
  velocity, :math:`\boldsymbol{\nabla}` the nabla operator, :math:`\nu` the
  viscosity and :math:`\boldsymbol{\nabla}^2` the Laplacian operator.


.. |Dt| mathmacro:: \mbox{D}_t
.. |bnabla| mathmacro:: \boldsymbol{\nabla}
.. |vv| mathmacro:: \textbf{v}

Now, let's use some math macros and try to get the same result... We use the
following code:

.. code-block:: rest

  The Navier-Stokes equation can now be written like this:

  .. math:: \Dt \vv = - \bnabla p + \nu \bnabla^2 \vv

  where |Dt| is the convective derivative, |vv| the velocity, |bnabla| the nabla
  operator, :math:`\nu` the viscosity and :math:`\bnabla^2` the Laplacian operator.

It gives:

The Navier-Stokes equation can now be written like this:

.. math:: \Dt \vv = - \bnabla p + \nu \bnabla^2 \vv

where |Dt| is the convective derivative, |vv| the velocity, |bnabla| the nabla
operator, :math:`\nu` the viscosity and :math:`\bnabla^2` the Laplacian operator.

"""

import re

from docutils.parsers.rst.directives.misc import Replace
from docutils.parsers.rst.roles import math_role as old_math_role
from sphinx.directives.patches import MathDirective


def multiple_replacer(replace_dict):
    """Return a function replacing doing multiple replacements.

    The produced function replace `replace_dict.keys()` by
    `replace_dict.values`, respectively.

    """

    def replacement_function(match):
        s = match.group(0)
        end = s[-1]
        if re.match(r"[\W_]", end):
            return replace_dict[s[:-1]] + end
        else:
            return replace_dict[s]

    pattern = "|".join(
        [re.escape(k) + r"[\W_$]" for k in replace_dict.keys()]
        + [re.escape(k) + "$" for k in replace_dict.keys()]
    )
    pattern = re.compile(pattern, re.M)

    def _mreplace(string):
        return pattern.sub(replacement_function, string)

    return _mreplace


def multiple_replace(string, replace_dict):
    mreplace = multiple_replacer(replace_dict)
    return mreplace(string)


class MathMacro(Replace):
    """Directive defining a math macro."""

    def run(self):
        if not hasattr(self.state.document, "math_macros"):
            self.state.document.math_macros = {}

        latex_key = "\\" + self.state.parent.rawsource.split("|")[1]
        self.state.document.math_macros[latex_key] = "".join(self.content)

        self.content[0] = ":math:`" + self.content[0]
        self.content[-1] = self.content[-1] + "`"

        return super().run()


class NewMathDirective(MathDirective):
    """New math block directive parsing the latex code."""

    def run(self):
        try:
            math_macros = self.state.document.math_macros
        except AttributeError:
            pass
        else:
            if math_macros:
                mreplace = multiple_replacer(math_macros)
                for i, c in enumerate(self.content):
                    self.content[i] = mreplace(c)
                for i, a in enumerate(self.arguments):
                    self.arguments[i] = mreplace(a)
        return super().run()


def new_math_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """New math role parsing the latex code."""
    try:
        math_macros = inliner.document.math_macros
    except AttributeError:
        pass
    else:
        if math_macros:
            rawtext = multiple_replace(rawtext, math_macros)
            text = rawtext.split("`")[1]

    return old_math_role(
        role, rawtext, text, lineno, inliner, options=options, content=content
    )


def setup(app):

    app.add_role("math", new_math_role, override=True)
    app.add_directive("math", NewMathDirective, override=True)

    app.add_directive("mathmacro", MathMacro)



if __name__ == "__main__":
    math_macros = {"\\vv": "\\textbf{v}"}
    s = "\\vv"
    mreplace = multiple_replacer(math_macros)
    print(mreplace(s))
