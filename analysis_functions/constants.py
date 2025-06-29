"""
analysis_funtions.constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the time constants used in this package. Each value is the number
of seconds in that unit. 

Examples:

Use this for calculating sample rates in Hertz:
- `import analysis_functions as my`
- `sample_rate_Hz = 1 / (15 * my.MINUTE)`
- `print(f"Sample rate is {sample_rate_Hz} Hz")`
- `>> Sample rate is 0.0011111 Hz`

Calculate hours in a year:
- `import analysis_functions as my`
- `my.YEAR/my.HOUR`
- `8765.82`

exports:
- SECOND
- MINUTE
- HOUR
- DAY
- WEEK
- MONTH
- YEAR

-----
"""

SECOND = 1
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY
YEAR = 365.2425 * DAY
MONTH = YEAR / 12