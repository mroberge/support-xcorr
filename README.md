# support-xcorr: Supplemental materials for cross-correlation methods analysis

This collection contains the source code and data in support of a forthcoming analysis of cross-correlation methods (under review).

## How to Cite

Roberge, M.C. (2025). mroberge/support-xcorr: supplemental materials for cross-correlation methods analysis (v0.0.1). [Collection] https://github.com/mroberge/support-xcorr/ Zenodo. https://doi.org/10.5281/zenodo.15765398

## Installation

All of the files in this repo can be downloaded and run immediately if you have the correct software set up already. However, if you are starting from scratch, follow these steps to install Python, download this project, and set up a virtual environment:

1. Create a new working directory on your computer. For example, I created `D:\my-example-dir`
2. Click on the green 'Code' button on the [github page for this project](https://github.com/mroberge/support-xcorr/) and select 'Download ZIP'.
3. Extract the zip file into your new working directory.
4. If you don't have Python installed on your computer, [do it now](https://www.python.org/downloads/).
5. Open a command line prompt in your new working directory.
6. Create a new environment called "xcorr_project":
   - Type: `python -m venv xcorr_project`
   - this creates a new directory called "xcorr_project" inside of your working directory that holds all of the software packages we'll be installing.
   - delete this directory if you want to get rid of this project and its environment.
7. Activate the environment:
   - Windows: `xcorr_project\Scripts\activate`
   - Linux/MacOS: `source xcorr_project/bin/activate`
8. Install the required software packages: `python -m pip install -r requirements.txt`
9. Register the kernel: `ipython kernel install --user --name=xcorr_project `
   - A kernel refers to a particular installation of Python. Our new kernel is installed inside of our new virtual environment and has the same name as the environment.
   - When you run Jupyter, you can select this kernel to access everything in the new virtual environment.
10. To run Jupyter lab: `jupyter lab`
    - You may need to specify that you want jupyter to open in your working directory:
    - `jupyter lab --notebook-dir path-of-your-directory`
    - For example, I used: `jupyter lab --notebook-dir D:\my-example-dir`
11. Jupyterlab provides a GUI interface through your web browser. Use it to open and run the notebooks. If you have more than one version of Python installed, choose the version (kernel) we just created: `xcorr_project`.

Repeat steps 7 & 10 ('Activate the Environment' & 'Run jupyter lab') from inside your working directory whenever you want to run Jupyterlab again.

## Notebooks

These notebooks run the analysis. You can click on the file names to view a rendered version.

**Essential notebooks:**

- **`0_Data-Download.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Download, process, and save stream stage and discharge data from the USGS.
- **`1_Stations.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Creates a table that describes each station and saves as stations.parquet
- **`2_Reaches.ipynb**`\*\*  
  &nbsp;&nbsp;&nbsp;&nbsp;Creates a table that describes each reach and saves as reaches.parquet
- **`3_Validation-data.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Creates a dataset of wave observations by manually matching peaks. Results are saved in validation_reaches.parquet
- **`5a_Execute-analysis.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Defines the parameters for each method, then runs the cross-correlation analysis using these parameters. It uses 5b_xcorr-method-template.ipynb to create a series of notebooks, one for each method. Saves output as xcorr_output.parquet; session info saved to session-parameters.json and parameters.json.
- **`5b_xcorr-method-template.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;The template for each analysis run.
- **`6_XCorr_band1d130m-full.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Analysis that uses a 1-day, 130-minute bandpass filter on the depth data before running the cross-correlation analysis. Results saved to xcorr-out-band1d130m-full-{YEAR}-{MO}-{DAY}-T{HOUR}.json
- **`6_XCorr_{{method-name}}.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Each data-preparation technique will get its own notebook following this naming convention.
- **`7_Compare-methods.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Quick comparison analysis of xcorr_output.parquet
- **`8_Tables.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Renders all tables used in paper.
- **`9_Figures.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Renders all figures used in paper.

## Additional Notebooks

These notebooks explore side issues or provide factoids for the text.

**Additional notebooks:**

- **`X_Examine-missing-data.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;
- **`X_Check-for-overbank-flows.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;
- **`X_Validation-descriptive-statistics.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;
- **`X_Compare-filters.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Plots examples of filters used in this analysis.
- **`X_Outlier-analysis.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;
- **`X_Examine-segments.ipynb`**  
  &nbsp;&nbsp;&nbsp;&nbsp;

## Essential Files & Data

These files are used to set up the environment, provide code snippets, back up input data, and store results.

**Essential files:**

- **`/analysis_functions/`**  
  &nbsp;&nbsp;&nbsp;&nbsp;A local Python package containing some of the code used in this analysis. All other code is located in the notebooks or are installed automatically as packages from PyPI.
- **`WBdata-depth.parquet`**  
  &nbsp;&nbsp;&nbsp;&nbsp;USGS stream gauge stage data, converted to depths in meters. Created in notebook #0.
- **`stations.parquet`** & **`reaches.parquet`**
  &nbsp;&nbsp;&nbsp;&nbsp;Site information about the five stations and four reaches. These files are created in notebooks #1 & 2.
- **`xcorr-output.parquet`**  
  &nbsp;&nbsp;&nbsp;&nbsp;Results from the cross-correlation analysis, saved as a dataframe with a multi-index. Created by notebook #5a.
- **`requirements.txt`**  
  &nbsp;&nbsp;&nbsp;&nbsp;List of dependencies.
