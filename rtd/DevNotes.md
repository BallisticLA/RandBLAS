# What goes into the RandBLAS website

## 1. Dependencies
* sphinx (``pip install sphinx``)
* doxygen (e.g., on macOS: ``brew install doxygen``)
* breathe (``pip install breathe``)
* sphinx_rtd_theme (``pip install sphinx-rtd-theme``). 
* bibtex support (``pip install sphinxcontrib-bibtex``).

Note: ``sphinx_rtd_theme`` is somewhat optional; alternative styles could be used with modest changes to the files described below.

## 2. Building the documentation

Once dependencies are installed, one builds the documentation by running the command
```
sphinx-build source build
```
within the ``rtd`` folder. (The role of this folder is explained below.)

## 3. Project file and folder structure

We have one directory that contains all files needed for web documentation. We call this directory "``rtd``", as an abbreviation for "read the docs". Here is the project structure relative to that directory.
```
rtd/
├── source/
│   ├── conf.py
│   ├── Doxyfile
│   ├── DoxygenLayout.xml
│   ├── index.rst
│   └── <OTHER_FOLDERS>
├── build/
├── sphinxext/
│   └── mathmacros.py
├── themes/
│   └── randblas_rtd/
│       ├── static/
│       │   └── theme_overrides.css
│       └── theme.conf
├── requirements.txt
└── howwebuiltthis.md
```
The files ``conf.py``, ``Doxyfile``, ``theme.conf``, and ``theme_overrides.css`` are all obtained by customizing boilerplate code. We explain the customizations below.

The file ``mathmacros.py`` contains custom, nontrivial code. It's needed so that we can define LaTeX macros and use them in the C++ source code documentation. We clarify its role and explain how it's accessed below.

The file ``index.rst`` defines the landing page of the sphinx website. This file is also used together with ``rtd/source/<OTHER_FOLDERS>`` to define *all* pages on the sphinx website. Because there's a ton of information about general sphinx websites out there we won't go into much more detail about this file.

## 4. Where did these files come from, and what goes into them?

### 4.1. What goes into conf.py

The following lines are pretty basic, and apply to just about any sphinx project that uses breathe.
```
import subprocess, os, sys
if not os.path.exists('../build/html'):
    os.makedirs('../build/html')
subprocess.call('doxygen --version', shell=True)
subprocess.call('doxygen', shell=True)
breathe_projects = {
    "RandBLAS": "../build/xml"
}
breathe_default_project = "RandBLAS"
source_suffix = '.rst'
master_doc = 'index'
```
The values specified for ``source_suffix`` and ``master_doc`` tell ``sphinx`` that it should look at ``rtd/source/index.rst`` for the definition of the website landing page. As mentioned before, that file will also tell sphinx where to look within ``rtd/source/<OTHER_FOLDERS>`` for files that define other pages of the website.

``rtd/source/conf.py`` also specifies our desired sphinx extensions by name, with a list of strings:
```
extensions = [
    'breathe',               # required!
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',  # optional
    'mathmacros'             # required! Must be last!
]
sys.path.insert(0, os.path.abspath('../sphinxext'))
# ^ Needed for the custom mathmacros sphinx extension. 
```
Finally, ``rtd/source/conf.py`` specifies the base theme for the website, and it gives the name for the customized theme that we build on top of the base.
```
import sphinx_rtd_theme
base_theme_html_path = sphinx_rtd_theme.get_html_theme_path()
base_theme_extension = 'sphinx_rtd_theme'

html_theme_path = [base_theme_html_path, "../themes"]
extensions += [base_theme_extension]
html_theme = 'randblas_rtd'
```

Note: a complete ``conf.py`` file may contain additional lines beyond those indicated above.

### 4.2. What to put in theme.conf

Sphinx will look in the directory ``rtd/themes/randblas_rtd`` for a file called ``theme.conf``. Here is a minimal example for that file:
```
[theme]
inherit = sphinx_rtd_theme
stylesheet = theme_overrides.css
```

### 4.3. What to put in theme_overrides.css
Sphinx will look for ``theme_overrides.css`` in the directory ``rtd/themes/randblas_rtd/static/``. Here is a minimal example for that file.
```
@import 'css/theme.css';

/* override specific CSS settings, as desired. */
// CSS code ...
// CSS code ...
```
The import statement will always be needed. The argument of the import statement can change depending on your base theme (e.g., a base theme of ``sphinx_rtd_theme`` versus ``alabaster``).

### 4.4. mathmacros.py : a custom Sphinx extension

Sphinx and reStructuredText files do a great job of providing basic LaTeX support. However, these are not enough if you want to define custom LaTeX commands for use in your C++ source code documentation. Someone posted [a question on StackExchange](https://stackoverflow.com/questions/25729537/math-latex-macros-to-make-substitutions-in-restructuredtext-and-sphinx) about how to do this, and later followed up with [an answer](https://stackoverflow.com/a/25818305/2664946). We've adopted their answer into this file.

Note: this file needs to be located in ``rtd/sphinxext``. This is because ``conf.py`` added ``../sphinxext`` to the system path, and that directory relatve to the ``rtd/source`` directory that contains ``conf.py``.

### 4.5. Customizations in the Doxyfile

``rtd/source/Doxyfile`` contains many settings. We customize the following settings:
* Basic project information:
  ```
    PROJECT_NAME           = RandBLAS
    # ... more settings ... 
    PROJECT_BRIEF          = "RandBLAS, a C++ library for sketching in randomized linear algebra."
    # ... more settings ... 
    OUTPUT_DIRECTORY       = ../build
    # ... more settings ... 
    INPUT                  = ../../README.md \
                            ../../ \
                            ../../RandBLAS/
    # ... more settings ...
    HTML_OUTPUT            = doxygen
  ```
* Define a succinct alias for using in-line reStructured Text math mode:
  ```
    ALIASES = math{1}="@verbatim embed:rst:inline :math:`\{\1\}` @endverbatim "
  ```
  This lets you define LaTeX macros in one part of a C++ source code docstring and then cleanly use the macros in another part of the same docstring. For example, in a long docstring that needs LaTeX math operator typesetting for functions called "op" and "mat", we can do
  ```
    /// @verbatim embed:rst:leading-slashes
    ///
    ///   .. |op| mathmacro:: \operatorname{op}
    ///   .. |mat| mathmacro:: \operatorname{mat}
    ///
    /// @endverbatim
    /// The matrix \math{\op(\mat(A))} is \math{m \times n}.
  ```
  