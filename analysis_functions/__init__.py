"""
analysis_functions
~~~~~~~~~~~~~~~~~~

A series of functions that I used in my celerity analysis.

Basic Usage::

    >>> import analysis_functions as my
    >>> print("Analysis_functions version: ", my.__version__)
    Analysis_functions version: 2025.06.09
    

    >>> dir(my)
    produces list of available functions

    >>> help(my)
    prints this message, which you are reading right now.
    
    >>> help(my.datagaps)
    prints the help docs for the function datagaps.

    >>> my.markdown("# A new title")
    <H1> A new title </H1>  --> this renders inside of a Jupyter cell.
    
    >>> my.MINUTE
    60

    >> my.HOUR
    3600

"""

__title__ = 'analysis_functions'
__version__ = '2025.06.27'
__author__ = 'Martin Roberge'
__email__ = 'mroberge@towson.edu'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025 Martin Roberge'


# Bring all of the functions into the main namespace
from .analysis_functions import *
from .constants import *
from .cross_correlation import *
from .display_helpers import *

def print_versions():
    print("Package Versions")
    print("----------------")
    print(f"Python: {platform.python_version()}")
    main_dependencies = ['numpy', 'pandas', 'scipy', 'statsmodels', 'pangoin','matplotlib', 'seaborn', 'hydrofunctions']

    for dependency in main_dependencies:
        try:
            vers = version(dependency)
        except PackageNotFoundError:
            vers = 'not imported'
        finally:
            print(f"{dependency}: {vers}")

    try:
        print(f"analysis_funtions: {__version__}")
    except:
        pass

# The following code can be used to force re-loading of these modules after editing.
#import importlib
#importlib.reload(analysis_functions)
#importlib.reload(constants)
#importlib.reload(cross_correlation)
#importlib.reload(display_helpers)
