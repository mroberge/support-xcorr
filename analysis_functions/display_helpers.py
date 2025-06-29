"""
analysis_funtions.display_helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains functions that help with the Jupyter display.

-----
"""
from IPython.display import display
from IPython.display import Markdown as md
import platform
from importlib.metadata import version
from importlib.metadata import PackageNotFoundError
import pandas as pd



def markdown(text):
    """Add markdown text to cells.
    """
    return display(md(text))


def big_df(df, max_rows=500, max_cols=40, min_rows=200):
    """Temporarily increase the number of rows that Pandas will display in a Jupyter notebook.
    """
    if len(df) > max_rows:
        print(f"Datafame exceeds {max_rows} rows; truncating to {min_rows} rows...")
    with pd.option_context('display.max_rows', max_rows, 'display.min_rows', min_rows, 'display.max_columns', max_cols):
        df.style.set_sticky(axis="index")
        df.style.set_sticky(axis="columns")
        display(df)