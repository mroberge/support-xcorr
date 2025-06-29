"""
analysis_functions.py

A series of functions that get used in the accompanying analysis of celerity.

To import from this file, just do this:
`import analysis_functions
`print("Analysis_functions version: ", analysis_functions.__version__)

To use a function, do this:
`analysis_functions.name_of_the_function()

conda install scipy, pyarrow
"""
from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as inset
import scipy.signal as signal
from scipy.signal import butter, bessel
from scipy.signal import sosfiltfilt
from scipy.signal import detrend as sig_detrend
import gzip
from IPython.display import display
from IPython.display import Markdown as md
import hydrofunctions as hf

__title__ = 'analysis_functions'
__version__ = '2025.06.08'
__author__ = 'Martin Roberge'
__email__ = 'mroberge@towson.edu'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025 Martin Roberge'


def markdown(text):
    """Add markdown text to cells.
    """
    return display(md(text))

def clean (DF, nans=False):
    DF.index.name = 'time'
    
    # Remove all duplicated index values
    print(f'Original data has {len(DF.index)} entries. There are no repeated index values: {DF.index.is_unique}')
    DF = DF[~DF.index.duplicated(keep='first')]
    print(f'There are now {len(DF.index)} entries.')
    DF.sort_index(axis=0, inplace=True)
    print(f'New index has no repeated values: {DF.index.is_unique}, and it is monotonic: {DF.index.is_monotonic}.')
    
    
    new_freq = DF.index.freq
    if DF.index.freq is None:
        new_freq = pd.infer_freq(DF.index)
        if new_freq is None:
            guess1 = (DF.index.max() - DF.index.min())/len(DF.index)
            if pd.Timedelta('13 minutes') < guess1 < pd.Timedelta('17minutes'):
                new_freq = pd.Timedelta('15 minutes')
            else:
                raise ValueError("Cannot calculate the dataset frequency.")
            
    print(f'Original frequency: {DF.index.freq}; New Frequency: {new_freq}')
    orig_len = len(DF)
    print(f'Re-organizing the time index. There should be {((DF.index.max()-DF.index.min())/new_freq) + 1} records.')
    DF = DF.asfreq(new_freq)
    new_len = len(DF)
    print(f'original length: {orig_len}; new length: {new_len}; missing records: {new_len - orig_len}')
    
    nulls = DF.isnull().sum()
    nulls = pd.DataFrame(nulls, columns=['NaN_count'])
    nulls.index.name = 'columns'
    nulls['percent'] = nulls / len(DF)
    print(f'Missing data: ')
    print(nulls)
    

    
    # create a data dataframe with the discharge data.
    # TODO: make this work when you ask for more than one type data (right now it assumes that only IV discharge has been requested.)
    # TODO: It would be better to work with the original USGS JSON file than to re-parse this dataframe.
    # TODO: for now, make a separate request for each parameter code.
    
    #     parse the column names
    #        strip the 'USGS:'  (or check for it and raise an error if it is not there)
    #        check for any summary type that isn't '0000'; raise error if they exist; get rid of them. I'll worry about these later.
    #        separate columns into different stations
    #        For each station name:
    #            separate columns into different parameter codes
    #            check that the _qualifiers columns for a station match; raise an error if they don't.
    #            Add one of the _qualifiers columns into a new meta dataframe. get rid of the others.
    #            For each parameter code:
    #                Add the data column to a dataframe for that parameter.
    data = DF.iloc[:, ::2] # Select all rows, select all columns stepping by two.
    
    # rename data columns
    cols = data.columns.values
    for i, col in enumerate(cols):
        cols[i] = col[5:-12] # This gets rid of everything in the name except for the station ID.
        #cols[i] = col[5:-6] # This gets rid of everything in the name except the station ID and the parameter code.
    data.columns = cols
    
    # create a metadata dataframe with data flags.
    meta = DF.iloc[:, 1::2] # Select all rows, select all columns starting at 1 and stepping by two.
    
    # rename meta columns
    #cols = meta.columns.values
    #for i, col in enumerate(cols):
    #    cols[i] = col[5:-23]+'_qualifiers'
    
    #data = data[~data.index.duplicated(keep='first')]
    #data.sort_index(axis=0, inplace=True)
    data.sort_index(axis=1, inplace=True)
    meta.sort_index(axis=1, inplace=True)
    #data = data[sorted(data.columns)]
    #meta = meta[sorted(meta.columns)]
        
    # Create new data structure
    result = {'data':data, 'meta': meta}
    
    print(f'First observation: {data.index.min()}')
    print(f'Last observation: {data.index.max()}')
    print(f'Length: {data.index.max()-data.index.min()};   {len(data)} records x {len(data.columns)} sites.')
    
    return result


def track_peak(data, start, period, manual={}, legend_loc=None, title_text=None, colors=None, xlabel=None, ylabel=None):
    """plot a window of stream discharge data and select the peak discharge.
    
    This function was designed to help in the selection of individual flood waves for the validation dataset. It plots a series of
    stream gage data from a start date through a designated period; it then selects the maximum value for each series and returns
    the time and value of the peak. It allows the user to manually select a peak if the maximum value in the data is not the desired
    peak. It plots the final peaks and the peaks it would have selected automatically.
    
    Args:
        data (df):
            A dataframe of USGS stream gage data, with each column representing a different station in a series of gages along a river.
        start (str, 'year-month-day hour:min:second'): 
            the start date and time for use in slicing a time segment from data.
        period (str):
            describes the length of time from start that the slice will extend. Should be in the format of '1 day' or '5 hours'.
        manual (dict):
            a dict of peaks that have been manually entered to replace the auto-peak detecting function in cases where the highest value
            in the segment is not the desired peak. Uses the gage number from the data columns as a key, and a date and time of the peak,
            following the same format as 'start'. It also accepts the value of 'skip' if no peak should be identified, and returns Nan.
            Example: manual={'01541200':'2016-04-04 15:45:00', '01541303':'skip'}
        legend_loc (str):
            The location for the legend if you are not pleased with the automatic placement of the legend. Accepts 'center left', 
            'upper center', etc.
        title_text (str):
            A string to be used as the title for the graph.
        colors (dict):
            A dict of colors to be used in plotting; site ID's are used as key.
        xlabel (str):
            A string to be used as the label for the x axis.
        ylabel (str):
            A string to be used as the label for the y axis.

    Returns:
        max_idx:
            a dataframe of times for the peak in each series.
        max_data:
            a dataframe of the peak value for each series.
    """
    start = pd.Timestamp(start)
    period = pd.Timedelta(period)
    stop = start + period
    
    sites = sorted(data.columns)
    max_idx = data[start:stop].idxmax() # Collect times for every site all at once.
    max_data = data[start:stop].max() # Collect max values for every site all at once.
    max_idx_plot = max_idx.copy()
    
    fig, ax = plt.subplots(figsize= (14,10))             
    for site in sites:
        if colors is not None:
            site_color = colors[site]
        else:
            site_color = None
        ax.plot(data[site][start:stop], label=site, color=site_color)

        if site in manual:
            # if there is a manually chosen wave for this site:
            #    plot the auto max using a different symbol
            #    if the site is to be skipped:
            #        set the time and discharge to 'Nan'
            #    store the manually chosen peak in the place of the automatically chosen peak.
            # 
            ax.plot(max_idx[site], max_data[site], 'rx', markersize=10, markeredgewidth=2)
            if manual[site] == 'skip':
                # Nothing should be plotted in this circumstance: so keep a timestamp but remove Y value so this point doesn't plot.
                # But also, the timestamp should be removed from the list that gets returned.
                max_idx[site] = pd.Timestamp('nan') # remove time from return list, but not plot list.
                max_data[site] = np.nan
            else:
                chosen_time = pd.Timestamp(manual[site])
                max_idx[site] = chosen_time 
                max_idx_plot[site] = chosen_time
                max_data[site] = data[site][chosen_time]
            
    ax.plot(max_idx_plot, max_data, 'k+:', markersize=20, markeredgewidth=2, label='selected peak')
    # A dummy entry for the legend, so this will only appear once if there is more than one manually selected peak.
    ax.plot([], [], 'rx', markersize=10, markeredgewidth=2, label='auto-peak (not selected)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    ax.set_title(title_text)
    plt.show()
    plt.close(fig)
    
    return max_idx, max_data


def Nfreq(wavelength_days, sample_rate_per_hour):
    """
    Calculates the normalized frequency
    # Calculate the critical frequency for a dataset based on the sample rate.
    # This uses my imaginary unit of 'Az', or cycles per year. A week has a frequency of 52 Az, and days are 365 Az.
    # To aid in translation from SI units, 
    #    1 cycle / year = 0.032 µHz  (microHz, Hz x 10⁻⁶)
    #    1 cycle / day  = 11.574 µHz
    #    1 cycle / hour = 277.8 µHz
    """
    Nyquist_freq_Az = sample_rate_per_hour * 24 * 365 / 2    # Nyquist_freq_Az should be 17520 if samples are taken every 15 minutes.
    crit_freq_Az = 365/wavelength_days
    Nratio = crit_freq_Az / Nyquist_freq_Az
    return Nratio


def norm_freq(input_period, sample_period):
    """Calculate the normalized frequency from inputs given as periods.
    
    This function accepts inputs as periods instead of frequencies in order to facilitate earth science applications, which frequently
    deal with longer cycles that are inconvenient to express in Hertz.
    
    Examples expressed in SI units:
    - 1 cycle / year = 0.032 µHz  (microHz, Hz x 10⁻⁶) 'annual' or 'seasonal' cycle
    - 1 cycle / day  = 11.574 µHz 'diurnal' cycle
    - 1 cycle / hour = 277.8 µHz
    
    Args:
        input_period (str):
            a string describing the length of the cycle being normalized. Accepts strings such as '1 day', '3 hours 30 seconds', '2 weeks'.
        sample_period (str):
            a string describing the length of time between samples. USGS 'real-time' data frequently have sample rates of '15 minutes'.
    
    Returns:
        Nratio (float):
            a floating point value representing the normalized frequency (or 1/period) of the input period. Value is calculated as the
            ratio of the input frequency divided by the Nyquist frequency.
    """
    nyquist_period = pd.Timedelta(sample_period) * 2
    normalized_frequency = nyquist_period / pd.Timedelta(input_period)
    
    return normalized_frequency

def estimate_filter_response_length(b, a):
    # This is an approximation of how many samples it takes for an impulse to quiet down to almost zero using a particular filter.
    # from bottom of https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
    z, p, k = tf2zpk(b, a)
    eps = 1e-9
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    return approx_impulse_len



def filter(data, order=2, 
           cutoff='130 minutes', 
           sample_period='auto', 
           filter_type='lowpass', 
           filter_method='butterworth'
          ):
    """Filter a Pandas dataframe with a Butterworth filter and return a dataframe.

    Parameters
    ----------
        data : pandas.DataFrame
            The data that you would like to have filtered. Data will be filtered along columns. Multiple columns can be filtered, but
            they must all have a common index.
        order : int, default: 2
            The order of the filter. Higher values will have a tighter fit, but are less stable.
        cutoff : str, default: '130 minutes'
            The filter cutoff parameters, given as a period written as a string. For example, '1 day' or '5 hours 27 seconds'.
            For bandpass or bandstop filters, this should be given as a list of the low and high cutoffs, with the longer period listed first.
        sample_period : str, optional
            How often the observations in the time series are sampled, given as a period written as a string. Example: '15 minutes'
            If not specified, the default is 'auto', which will attempt to figure out the sample period from the data.
        filter_type : str, {'lowpass', 'highpass', 'bandpass', 'bandstop'}
            The type of filter. Default is a lowpass filter with a 130 minute cutoff, which is ideal for smoothing USGS 15-minute stream gauge data.
            Bandpass and bandstop filter need two cutoff periods, given as a list.
        filter_method : str {'butterworth', 'bessel'}

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe with the same index and column names as the input data.
    """
    if sample_period == 'auto':
        sample_period = data.index.freq
    assert sample_period is not None, "The sample_period was set to None. Instead, it should be set to the amount of time between measurements."
    
    if filter_type == 'lowpass' or filter_type == 'highpass':
        Nratio = norm_freq(cutoff, sample_period)
    elif filter_type == 'bandpass' or filter_type == 'bandstop':
        low = cutoff[0]
        high = cutoff[1]
        Nratio = [norm_freq(low, sample_period), norm_freq(high, sample_period)]
        assert Nratio[0] < Nratio[1], f"For filter_type {filter_type}, the cutoff should be a pair of values, with the longer period first. You sent cutoff=({low}, {high}) instead."
    else:
        print(f"filter_type was set to '{filter_type}', valid types are 'lowpass', 'highpass', 'bandpass', and 'bandstop'.")

    if filter_method == 'bessel':
        sos = bessel(order, Nratio, btype=filter_type, analog=False, output='sos', norm='phase')
    elif filter_method == 'butterworth':
        sos = butter(order, Nratio, btype=filter_type, output='sos')
    else:
        raise ValueError(f"filter() was expecting filter_method of either 'butterworth' or 'bessel', but received '{filter_method}' instead.")
    y = sosfiltfilt(sos, data, axis=0)
    result = pd.DataFrame(data=y, index=data.index, columns=data.columns)

    return result

def butterworth(data, order=2, cutoff='130 minutes', sample_period='auto', filter_type='lowpass'):
    """Filter a Pandas dataframe with a Butterworth filter and return a dataframe.
    """
    return filter(data, order=order, cutoff=cutoff, 
                  sample_period=sample_period, filter_type=filter_type,
                  filter_method='butterworth',
                 )


def replace_na(dataset, method='full'):
    """Replaces Nans with other values in preparation for cross-correlation analysis.
    
    Input:
    ======
        dataset: a dataframe or series.
        method: {full | inside | zero | mean }
            full: interpolate for Nans at start, internal, and trailing nans
            inside: only interpolate Nans surrounded by valid data.
            zero: replace all Nans with zeroes
            mean: replace all Nans with the mean.
            no: do NOT replace any Nans.
    """
    print(f"replacing {np.asarray(dataset.isna().sum())} Nans in dataset using the '{method}' method.")
    if (method == 'full'):
        dataset = dataset.interpolate(limit_direction='both')
    elif (method == 'inside'):
        print("The inside method has not been implemented; full interpolation run instead.")
        dataset = dataset.interpolate(limit_direction='both')
        # The following doesn't work: limit_area only works when a limit has been declared.
        # All it does is leave Nans at the start if limit_direction=forward,
        # or leave Nans at the end if limit_direction=backward.
        # In any case, a proper implementation for my purposes would have to crop the dataset so that it
        # started on a valid value.
        #dataset = dataset.interpolate(limit_area='inside')
    elif (method =='mean'):
        dataset = dataset.fillna(dataset.mean())
    elif (method == 'zero'):
        dataset = dataset.fillna(0)
    elif (method == 'no'):
        # Do not replace the nans.
        pass
    elif (method == 'chop'):
        print("The 'chop' method is not implemented yet. NaNs not removed.")
        # if dataset is a df, not a series, then it won't trim down to the shortest valid segment.
        valid = pd.notnull(dataset)
        start = dataset[valid].index.min()
        stop = dataset[valid].index.max()
        newdf = dataset[start:stop]
    else:
        print("This method is not recognized. Nans not removed.")
    print(f"There are {np.asarray(dataset.isna().sum())} Nans remaining.")
    
    return dataset


# Version 4: window takes tuple argument
def xcorr(x, upstream, downstream, window='minimum', replace_nans = 'full', plotting=True):
    """Calculate the lag between an upstream and a downstream site using cross-correlation.
    
    Inputs:
    =======
        x: the index for upstream & downstream. data.index 
        upstream: a data series for the upstream site.
        window: { 'minimum' | 'full' | int | (int, int) } Defines a window of lags that will be considered for a match. 
            This prevents the selection of really large, improbable lags for a cross-correlation match of the two arrays.
            'minimum': uses the length of the smaller of the upstream or downstream array to size the window.
            'full': no window is used, and the full length of the overlap is used to find the best match.
            int: any integer will get used to create a window that runs from [-value, value].
            (int, int): a tuple of integers, where the window will run from (start, stop). An example is (1, 100) where 
            xcorr will find the maximum correlation between 1 and 100 lags.
             
    Notes:
    ======
        The inputs can be of different lengths, and they can start at different times. However, I've found that the results
        of the cross-correlation aren't great in this situation.
        
        The behavior now is to fill in all Nans with interpolated values, and fill the starting and ending Nans with the
        first or last valid value of the series.
        
        This version will also limit the range of possible lags it will consider to a window.
    """
    maxlen = max(len(x), len(downstream), len(upstream)) #old npts
    """
    print('number of upstream points: ', len(upstream))
    print('number of downstream points: ', len(downstream))
    print('number of index points: ', len(x))
    print("original freq: ", x.freq)
    """
    if (x.freq is None):
        from pandas.tseries.frequencies import to_offset
        x.freq = pd.to_timedelta(to_offset(pd.infer_freq(x)))
        print("new freq: ", x.freq)
    #start_offset is the difference in start time between upstream and downstream.
    start_offset = downstream.index.min()-upstream.index.min()
    #print("start time offset: ", start_offset)
    start_offset_lag = int(start_offset / x.freq)
    #print("start offset lag: ", start_offset_lag)
    # lags should have a length of len(upstream)+len(downstream)-1
    # but the value of lags should start at a negative, hit a zero where the two time indexes line up, and go to positive
    lags = np.arange(-len(upstream) + start_offset_lag + 1, len(downstream) + start_offset_lag)   
    #print(lags)
    #print(f'lags length equals us len + ds len -1: {len(lags)} = {len(upstream)} + {len(downstream)} -1 = {len(upstream) + len(downstream) -1}')
    
    if replace_nans:
        #np.correlate returns nan if any nan are present.
        downstream = replace_na(downstream, method=replace_nans)
        upstream = replace_na(upstream, method=replace_nans)
        #assert(not np.any(np.isnan(downstream)))
        #assert(not np.any(np.isnan(upstream)))
    else:
        print(f"Number of NaNs in US dataset: {np.asarray(upstream.isna().sum())}")
        print(f"Number of NaNs in DS dataset: {np.asarray(downstream.isna().sum())}")
    
    ds_mean = downstream.mean()
    us_mean = upstream.mean()
    ccov = np.correlate(downstream - ds_mean, upstream -  us_mean, mode='full')
    ccor = ccov / (maxlen * np.nanstd(downstream) * np.nanstd(upstream))
    print(f'Length of lags: {len(lags)}; length of ccov: {len(ccov)}; length of ccor: {len(ccov)}; length of US: {len(upstream)}')
    print(f"ccov: {ccov}")
    print(f"ccov.max(): {ccov.max()}")
    print(f"ccor: {ccor}")
    print(f"ccor.max(): {ccor.max()}")    
        
    # if the two arrays are the same length, then the zero is at index len(maxlen)
    zero_index = (len(upstream) - 1) - start_offset_lag
    print(f'zero_index: {zero_index}')
        
    # Limit the range of ccor that argmax will consider
    # set the window size
    if ( isinstance(window, int)):
        window_half = window
        window_size = (window_half * 2) + 1 # This is for temp testing purposes
        start = zero_index - window_half
        stop = zero_index + window_half
        print(f'window is an integer: {window}; start: {start}, stop: {stop}', )
    elif( isinstance(window, (tuple, list))):
        window_start, window_stop = window
        window_size = (window_stop - window_start) #for testing purposes
        start = zero_index + window_start
        stop = zero_index + window_stop
        print(f'window is a tuple; start: {start}, stop: {stop}')
    elif (window == 'minimum'): 
        window_half = min(len(upstream), len(downstream)) - 1
        window_size = (window_half * 2) - 1
        start = zero_index - window_half
        stop = zero_index + window_half
        print(f'minimum window size: {window_half}; start: {start}, stop: {stop}')
    elif (window == 'full'): 
        window_size = len(ccov)
        start = 0
        stop = len(ccov)-1
        print(f'full window size: {window_size}; start: {start}, stop: {stop}')
    else:
        print(f"This window parameter is not recognized: {window}  of type {type(window)}.")
        raise ValueError(f"This window parameter is not recognized: {window}  of type {type(window)}.")
        
        #print(f'full window size: {window_size}; start: {start}, stop: {stop}')

    print(f"start, zero_index, stop: {start}, {zero_index}, {stop}. zero value: {ccor[zero_index]}")
    #If correlate came across any NaNs, it will put a NaN in the ccor array; this will then sum to np.NaN.
    if np.isnan(ccor.sum()):
        print(f"Cross-Correlation was not able to find a match due to missing data. ccor.sum(): {ccor.sum()}")
        maxlag = np.nan
        maxccor = np.nan
    else:
        max_index = np.argmax(ccor[start:stop + 1]) + start # only search from start to stop, but keep the old index.
        maxlag = lags[max_index]
        maxccor = ccor[max_index]
        #print(f"ccor values before max: {ccor[max_index-5:max_index]}")
        #print(f"lag  values before max: {lags[max_index-5:max_index]}")
        #print(f"ccor values after max: {ccor[max_index:max_index+5]}")
        #print(f"lag  values after max: {lags[max_index:max_index+5]}")
        width = np.argmax(ccor[max_index:] < maxccor*0.95) # What is the index for the first corr that is < maxccor*0.95 ?
        print(f"Cross-Correlation found a match: max_index:{max_index}  maxlag:{maxlag}  maxccor:{maxccor}  xwidth:{width}")
    """
    print(f'maxlag: {maxlag}, maxccor: {maxccor}')
    print('   i:   lags   --   ccor')
    print('|---|---------|--|----------')
    
    for i, value in enumerate (ccor):
        if (i < start):
            string = ''
        elif (i==start):
            string = f' ********* window start'
        elif (i < stop):
            string = ' **'
        elif (i== stop):
            string = f' ********* window stop '
        else:
            string = ''
        
        if (i == max_index):
            string = string + '<---------------- MAX'
        print(f'{i:>4}: {lags[i]:>8} -- {ccor[i]:10.7f}{string}')

    print(f"The upstream array must be shifted by a lag of {maxlag} to match the downstream array.")
    print(f"The maximum correlation is {maxccor}.")
    """   
    if plotting:
        fig, ax_xc = plt.subplots(figsize=(12, 7))
        # Cross-Correlation plot
        ax_xc.plot([0, 0], [-1, 1], color='r', linestyle='-', linewidth=1)
        ax_xc.plot(lags, ccor, label='correlation')
        if not np.isnan(maxlag):
            ax_xc.plot(maxlag, maxccor,'r+', markersize=12, markeredgewidth=1, label='maximum correlation')

        ax_xc.set_ylim(-0.6, 1.3)
        ax_xc.set_ylabel('cross-correlation')
        ax_xc.set_xlabel('upstream lag')
        ax_xc.legend(loc='upper left')
        ax_xc.set_title(f'cross-correlation and lag time from {upstream.name} to {downstream.name}')
    
        ax_xc_inset = inset.inset_axes(ax_xc, width="40%",height="50%",loc=1)
        ax_xc_inset.set_xlim(-100, 100)
        ax_xc_inset.set_ylim(0, 1.1)
        ax_xc_inset.plot([0, 0], [-1, 1], color='r', linestyle='-', linewidth=1)
        ax_xc_inset.plot(lags, ccor)
        ax_xc_inset.plot(maxlag, maxccor,'r+', markersize=12, markeredgewidth=2)
        annotation = f'lag: {maxlag}  xcorr: {maxccor:.2f}'
        ax_xc_inset.annotate(annotation, xy=(maxlag, maxccor), xytext=(3, 9), textcoords='offset points')
        inset.mark_inset(ax_xc, ax_xc_inset, 2, 4)
    
        plt.show()
        plt.close()

    return maxlag, maxccor, width


def length_analysis(data, upstream, downstream, level, length = None, reach=None, raw_data_label=None, raw_data=None, window='full', replace_nans='full', detrend=False, plotting=True):
    """Calculate the lag and cross-correlation for a dataset that has been broken into 2^level number of segments.
    
    The purpose of this function is to see how long of a data sequence is needed to produce lags of acceptable quality.
    It will split a dataset into smaller pieces, run xcorr on them, and then record the lag with the best correlation
    into a list, with each element of the list corresponding to one of the pieces of the original dataset.
    
    Inputs:
    =======
    
        data: a dataframe with labelled columns
        upstream: a selection label for a column in data. data[upstream] 
        downstream: the selection label for the downstream station column.
        level: the number of segments that data will be split into, number of segments = 2^level
        length: the distance in meters between the upstream and downstream stations.
        reach: the index value for the reach
        raw_data_label: (str) string to be used as the Y-axis label on plot of the raw data.
        raw_data: If data is transformed for analysis, then this is a dataframe with the untransformed values.
        window: either 'full' for accepting matches of the entire dataset, or a tuple indicating the min and max number of lags in which a search for a match will occur.
        replace_nans: policy for how to replace nans. Uses the options for replace_na()
        plotting (default: True): If true, plot the data in the output.
    
    Returns:
    ========
        Will return 'results', a list of dicts, each dict containing stats about a segment of the data.
        Warning: the US_
    """
    colors = {
        'us' : '#F27049',
        'us2': '#F2A891',
        'ds' : '#498DF2',
        'ds2': '#91B8F2',
    }
    
    if (data.index.freq is None):
        from pandas.tseries.frequencies import to_offset
        data.index.freq = pd.to_timedelta(to_offset(pd.infer_freq(data.index)))
        print("new freq: ", data.index.freq)

    # Fill NaNs right away using approprite option
    clean_data = replace_na(data, replace_nans)
    
    num_of_segments = pow(2, level)
    segments = np.array_split(clean_data, num_of_segments)
    seg_length = len(segments[0])
    results = []
    print(f"Analysis from site {upstream} to {downstream} in {num_of_segments} parts; segment length: {seg_length} ({pd.Timedelta('15 minutes') * seg_length})")
    for i, segment_df in enumerate(segments):
        print("\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"RUN {i} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        us_df = segment_df[upstream]
        ds_df = segment_df[downstream]
        if detrend == True:
            us_df = replace_na(us_df, method=replace_nans)
            ds_df = replace_na(ds_df, method=replace_nans)
            us_df = pd.Series(sig_detrend(us_df), index=segment_df.index) # Keep the same index pd.Series(detrend, 
            us_df.name = f"{upstream}-detrended"
            ds_df = pd.Series(sig_detrend(ds_df), index=segment_df.index)
            ds_df.name = f"{downstream}-detrended"
        start = segment_df.index.min() # The start time for this segment.
        stop = segment_df.index.max() # The end time for this segment.
        if raw_data is not None:
            US_nulls = raw_data.loc[start:stop, upstream].isna().sum()
            DS_nulls = raw_data.loc[start:stop, downstream].isna().sum()
        else:
            US_nulls = data.loc[start:stop, upstream].isna().sum()
            DS_nulls = data.loc[start:stop, downstream].isna().sum()
        us_data_max = us_df.max()
        us_data_mean = us_df.mean()
        ds_data_max = ds_df.max()
        ds_data_mean = ds_df.mean()
        us_idxmax = pd.to_datetime(us_df.idxmax(), unit='ms')  
        ds_idxmax = pd.to_datetime(ds_df.idxmax(), unit='ms')
        comments = f''
        detrendUS = None
        detrendDS = None
        # filter out maxes that occur at start or end of the sequence.
        if (((us_idxmax - start) < pd.Timedelta('10 hours')) | ((stop - us_idxmax) < pd.Timedelta('10 hours'))):
            lagx = np.nan
            note = f'| US peak is near start or end of sequence. us_idxmax: {us_idxmax}'
            comments = comments + note
            print(note)
        elif (((ds_idxmax - start) < pd.Timedelta('10 hours')) | ((stop - ds_idxmax) < pd.Timedelta('10 hours'))):
            lagx = np.nan
            note = f'| DS peak is near start or end of sequence. ds_idxmax: {ds_idxmax}'
            print(note)
            comments = comments + note
        else:
            # If the maximum values for US or DS sequence are not near the start or end, then find the time different between the US & DS peaks.
            lagx = ds_idxmax - us_idxmax # lagx will be NaN if either us or ds is NaN
            print(f"ds_idxmax: {ds_idxmax}- us_idxmax: {us_idxmax} = lagx: {lagx}")
        if lagx is not np.nan:
            lagmax = lagx / pd.Timedelta('15 minutes')
        else:
            lagmax = np.nan
        
        n_valid = min(us_df.count(), ds_df.count())
        lag = np.nan
        xc = np.nan

        # n_valid is the count of non-null values
        if n_valid < 10:
            note = f'ERROR. Number of valid observations is too low. n_valid: {n_valid}'
            print(note)
            comments = comments + "| " + note
            print(f'segment i: {i}; start: {start}; stop: {stop}')
            #raise RuntimeError(f'ERROR. Number of valid observations is too low. n_valid: {n_valid}')
            lag = np.nan
            xc = np.nan
            xwidth = np.nan
            celerity = np.nan
        else:
            if pd.isnull(lagx):
                note = "Possible error. One end of the segment is on a peak."
                print(note)
                comments = comments + "| " + note
                
            lag, xc, xwidth = xcorr(segment_df.index, 
                                    us_df, ds_df, 
                                    window=window, 
                                    replace_nans=replace_nans,
                                    plotting=plotting,
                                    )
            
            # If lag is 1 or less, set a comment and re-try xcorr with de-trended data.
            if lag < 2:
                note = f"| ERROR. original lag was {lag}. Will re-run xcorr after de-trending."
                comments = comments + note
                us_df = replace_na(us_df, method=replace_nans)
                ds_df = replace_na(ds_df, method=replace_nans)
                detrendUS = pd.Series(sig_detrend(us_df), index=segment_df.index) # Keep the same index pd.Series(detrend, 
                detrendUS.name = f"{upstream}-detrended"
                detrendDS = pd.Series(sig_detrend(ds_df), index=segment_df.index)
                detrendDS.name = f"{downstream}-detrended"
                old_lag = lag
                print("\n\n\nSecond Run of XCORR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                
                lag, xc, xwidth = xcorr(segment_df.index, 
                                        detrendUS, detrendDS, 
                                        window=window, 
                                        replace_nans=replace_nans,
                                        plotting=plotting,
                                       )
                
                print(f"~~~  RE-RUN {i}: lag hit threshold. Old lag: {old_lag}  New lag: {lag} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
                comments = comments + f" New lag:  {lag}. "
                old_lag = None
                if lag < 2:
                    lag = np.nan
            celerity = length / (lag * 15 * 60)
        
        result = {'us': upstream, 'ds': downstream, 'reach': reach, 'level': level, 'segment': i, 
                  'US_nulls': US_nulls, 'DS_nulls': DS_nulls, 'start': start, 'stop': stop, 
                  #'us_data_max': us_data_max, 'us_data_mean': us_data_mean, 
                  #'ds_data_max': ds_data_max, 'ds_data_mean': ds_data_mean,
                  'us_idxmax': us_idxmax, 'ds_idxmax': ds_idxmax, 'lagmax': lagmax,
                  'lag': lag, 'xcorr': xc, 'xwidth': xwidth, 'celerity': celerity,
                  'raw_us_data_max': raw_data.loc[start:stop, upstream].max(),
                  'raw_ds_data_max': raw_data.loc[start:stop, downstream].max(),
                  'raw_us_data_mean': raw_data.loc[start:stop, upstream].mean(),
                  'raw_ds_data_mean': raw_data.loc[start:stop, downstream].mean(),
                  'comments': comments
                 }
        results.append(result)
        print(result)
        print(f"\nlevel: {level}, i: {i}, start: {start}, stop: {stop}, lag: {lag}, xc: {xc}")
        print(f'Calculating the lag based on the maximum values: lagmax: {lagmax} ({lagx})')
        print(f"Calculating the lag based on the cross-correlation: lag: {lag} ({pd.Timedelta('15 minutes') * lag})")

        # Plotting
        if plotting:
            fig, [ax0, ax1] = plt.subplots(nrows=2,figsize=(12, 14))
            fig.subplots_adjust(hspace=0.2)
    
            ax0.plot(raw_data.loc[start:stop, downstream], colors['ds'], label=f"DS: {downstream}")
            ax0.plot(raw_data.loc[start:stop, upstream], colors['us'], label=f"US: {upstream}")
            if not np.isnan(lag):
                lag_time=pd.Timedelta(data.index.freq * lag)  # ...* abs(lag)
                x = raw_data.loc[start:stop, upstream].index
                y = raw_data.loc[start:stop, upstream].shift(lag)
                ax0.plot(x, y, colors['us'], linestyle='dotted', label=f'US:{upstream} lagged {lag_time}')
            ax0.set_xlim([start,stop])
            ax0.set_ylabel(raw_data_label)
            ax0.set_title(f"Original Untransformed data")
            ax0.legend()
            
            ax1.plot(ds_df.index, ds_df, colors['ds'], label = f'DS:{ds_df.name}')
            ax1.plot(us_df.index, us_df, colors['us'], label = f'US:{us_df.name} (original)')
            # Only plot the upstream data with a shift if the correlation was successful.
            if not np.isnan(lag):
                lag_time=pd.Timedelta(data.index.freq * lag)  # ...* abs(lag)
                ax1.plot(us_df.index, us_df.shift(lag), colors['us'], linestyle='dotted', label=f'US:{us_df.name} lagged {lag_time}')

            if detrendUS is not None:
                ax1.plot(detrendUS.index, detrendUS, colors['us2'], label=f"US {detrendUS.name}")
                ax1.plot(detrendDS.index, detrendDS, colors['ds2'], label=f"DS {detrendDS.name}")
                ax1.legend(loc='upper left')
            ax1.set_xlim([start,stop])
            ax1.set_ylabel(f"Transformed values")
            ax1.set_title(f'Transformed data at sites {us_df.name} and {ds_df.name}')
        
            # Plot Inset Axis
            # suppress the inset axis if there are fewer than about 1200 data points. You can see the lag in the main plot
            # if there are fewer than this number of points.
            draw_inset = True  # I took this out of the signature for xcorr when I moved this plot out of xcorr
            #draw_inset: {True, False, 'auto'} Plot an inset with a closeup of the data series. On auto, 
            #    it won't plot if the timeseries is so short that you can see the lag in the main plot.
            #    The inset will plot a four-day window focused on the maximum value in the dataset.
            if (draw_inset == 'auto'):
                if (maxlen < 1200):
                    draw_inset = False
                else:
                    draw_inset = True
            # Don't draw inset if xcorr couldn't produce a lag.
            #if np.isnan(lag):
            #    draw_inset = False
            if (draw_inset):
                axins = inset.inset_axes(ax1, width="40%", height="50%", loc=1)
                if detrendUS is not None:
                    axins.plot(detrendDS.index, detrendDS, colors['ds2'])
                    axins.plot(detrendUS.index, detrendUS, colors['us2'])
                    if not np.isnan(lag):
                        axins.plot(detrendUS.index, detrendUS.shift(lag), colors['us2'], ':')
                else:
                    axins.plot(ds_df.index, ds_df, colors['ds'])
                    axins.plot(us_df.index, us_df, colors['us'])
                    if not np.isnan(lag):
                        axins.plot(us_df.index, us_df.shift(lag), colors['us'], linestyle='dotted')                   
        
                # sub region of the original image will center on a peak
                ds_filled = replace_na(ds_df, 'full')
                center_x = ds_filled.idxmax()  #df.idxmax() no longer works if array has nans.
                xx = pd.Timedelta('2 days')
                xmin = center_x - xx
                xmax = center_x + xx
                ymin = np.nanmin([us_df.min(), ds_df.min()])  #np.nanmin() will ignore nans
                ymax = np.nanmax([us_df.max(), ds_df.max()]) * 1.02
                axins.set_xlim(xmin, xmax)
                axins.set_ylim(ymin, ymax)
                inset.mark_inset(ax1, axins, 2, 4)
            plt.show()
            plt.close()
    
    
    return results

# A function to help create a multiindex dataframe
def tupleizer(col):
    """Breaks long USGS labels into useful chunks.
    """
    parts = col.split(":")
    site = parts[1]
    param = parts[2]
    end = parts[3]
    if end == '00000_qualifiers':
        meta = 'flag'
    else:
        meta = 'data'
    return (site, param, meta)

def create_multi(df):
    """Convert a large hydrofunctions dataframe into a multi-index dataframe, with separate sites, parameters, and data vs metadata.
    """
    df = df.copy().rename(columns=tupleizer)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['site', 'param', 'meta'])
    return df

def print_missing(multi_df):
    # It might be usefull to have this return a dataframe
    # It might also be usefull to have it figure out how many different flags there are and include those too. maybe in a multiindex dataframe
    sites = multi_df.columns.levels[0]
    params = multi_df.columns.levels[1]
    for site in sites:
        print(site)
        for param in params:
            counts = multi_df.loc[:,(site, param, 'flag')].value_counts(dropna=False)
            data_nulls = multi_df.loc[:,(site, param, 'data')].isnull().sum()
            print(f"  {param}:  #nulls (%): {data_nulls:6} ({(data_nulls/(counts['hf.missing']+counts['A'])):6.2%})  flag:'hf.missing' (%): {counts['hf.missing']:6} ({(counts['hf.missing']/(counts['hf.missing']+counts['A'])):6.2%})  flag:'A': {counts['A']}   total: {counts['hf.missing']+counts['A']:8}")

def multi_missing(multi_df):
    sites = multi_df.columns.levels[0]
    params = multi_df.columns.levels[1]
    meta = multi_df.columns.levels[2]
    numbers = []
    index_list = []
    null_vals = []
    for site in sites:
        for param in params:
            # See multi_missing2() for a simpler version that doesn't use nested loops!
            counts = multi_df.loc[:,(site, param, 'flag')].value_counts(dropna=False)
            for count_idx in counts.index:
                my_tuple = (site, param, count_idx)
                index_list.append(my_tuple)
                numbers.append(counts[count_idx])
            index_list.append((site, param, 'nulls'))
            null_count = multi_df.loc[:,(site, param, 'data')].isnull().sum()
            numbers.append(null_count)
    new_df = pd.DataFrame(data=numbers, index=index_list, columns=['counts'])
    new_df.index = pd.MultiIndex.from_tuples(new_df.index, names=['site', 'param', 'flags'])
    new_df['ratio'] = new_df/len(multi_df)
    #new_df.unstack(level=-1)
    return new_df

def multi_missing2(multi_df):
    """A simpler version of multi_missing that doesn't count nulls"""
    # Unstack the columns into the rows
    test = multi_df.loc[:,(slice(None),slice(None),'flag')].stack(future_stack=True).stack(future_stack=True).stack(future_stack=True)
    
    # Reorganize the rows to clean up
    test = test.reorder_levels(order=['site', 'param', 'meta','datetimeUTC']).droplevel('meta').sort_index()
    
    counts = test.groupby(level=['site','param']).value_counts(dropna=False)
    
    new_df = pd.DataFrame(counts)
    
    new_df.index = pd.MultiIndex.from_tuples(new_df.index, names=['site', 'param', 'flags'])
    new_df['ratio'] = new_df/len(multi_df)
    return new_df

def datagaps(Qseries):
    """Compiles a dataframe of gaps in a timeseries.
    
        Args:
            Qseries (series): A Pandas time series of data
        Returns:
            df (dataframe): A dataframe of the gaps
            Each row is a different gap; 
                'first': a timestamp of the first NaN in the gap
                'last': a timestamp of the last NaN in the gap
                'n': the number of missing measurements
                'length': the length of time between first & last
                
    """
    # Find a null preceded by a value or that is in the first row.
    first_nan_index = (Qseries.isnull()) & ((Qseries.shift().notnull()) | (Qseries.index==0))
    
    # Find a null followed by a value or that is in the last row.
    last_nan_index = (Qseries.isnull()) & ((Qseries.shift(-1).notnull()) | (Qseries.index==Qseries.index.max()))
    
    # Create dataframe
    df = pd.DataFrame(Qseries[first_nan_index].index).rename(columns={'datetimeUTC':'first'})
    df.index.rename('gap_idx', inplace=True)
    df['last'] = Qseries[last_nan_index].index
    
    # Approach: 
    #     - either divide length by period (can I get period?)
    #     - or subtract the index values somehow?
    #df['n'] = #
    
    df['length'] = (df['last']-df['first'])
    
    return df

def easy_gaps(data):
    """Create a dict of datagap dataframes from a multiindex dataframe"""
    myindex = data.loc[:,(slice(None),slice(None),'data')].columns
    gap_set = {}
    for index in myindex:
        t = data.loc[:,index]
        gap_set[index] = datagaps(t)
        
    return gap_set

def multi_gap(multi_df):
    """Create a multi-index list of gaps found in a mulit-index set of data."""
    # This creates a dict of dataframes, each a list of gaps.
    #gaps = easy_gaps(multi_df)
    
    # Select just the data columns; lose the 'flag' columns
    myindex = multi_df.loc[:,(slice(None),slice(None),'data')].columns
    gap_set = {}
    for index in myindex:
        # For each pair of site&param, get the list of values
        #print(index)
        t = multi_df.loc[:,index]
        gap_set[index] = datagaps(t)
    
    for k,v in gap_set.items():
        site, param, meta = k
        v['site'] = site
        v['param'] = param

    df = pd.concat(gap_set.values()).reset_index().set_index(['site', 'param', 'gap_idx'])

    return df

""" Functions for dealing with field measurements"""

def field_meas2(site, verbose=True):
    """Load USGS field measurements of stream discharge into a Pandas dataframe.

    Args:
        site (str):
            The gauge ID number for the site.
        verbose (bool):
            If True (default), will print confirmation messages with the url before and
            after the request.

    Returns:
        a hydroRDB object or tuple consisting of the header and a pandas
        dataframe. Each row of the table represents an observation on a given date of
        river conditions at the gauge by USGS personnel. Values are stored in
        columns, and include the measured stream discharge, channel width,
        channel area, depth, and velocity.

    **Example:**

        >>> test = field_meas('01542500')
        >>> test
        hydroRDB(header=<a mulit-line string of the header>,
                 table=<a Pandas dataframe>)

    You can also access the header, dataframe, column names, and data types
    through the associated properties `header`, `table`, `columns`, `dtypes`::

        >>> test.table
        <a Pandas dataframe>

    **Discussion:**
        The USGS operates over 8,000 stream gages around the United States and
        territories. Each of these sensors records the depth, or 'stage' of the
        water. In order to translate this stage data into stream discharge, the
        USGS staff creates an empirical relationship called a 'rating curve'
        between the river stage and stream discharge. To construct this curve,
        the USGS personnel visit all of the gage every one to eight weeks, and
        measure the stage and the discharge of the river manually.

        The ``field_meas()`` function returns all of the field-collected data for
        this site. The USGS uses these data to create the rating curve. You can use
        these data to see how the site has changed over time, or to
        read the notes about local conditions.

        The ``rating_curve()`` function returns the most recent 'expanded shift-
        adjusted' rating curve constructed for this site. This is the current official
        rating curve.

        To plot a rating curve from the field measurements, use::

            >>> header, data = hf.field_meas('01581830')

            >>> data.plot(x='gage_height_va', y='discharge_va', kind='scatter')

        Rating curves are typically plotted with the indepedent variable,
        gage_height, plotted on the Y axis.
    """
    url = (
        "https://waterdata.usgs.gov/nwis/measurements?site_no="
        + site
        + "&agency_cd=USGS&format=rdb_expanded"
    )
    headers = {"Accept-encoding": "gzip"}

    if verbose:
        print(
            f"Retrieving the field measurements for site #{site} from {url}", end="\r"
        )
    response = hf.get_usgs_RDB_service(url, headers)
    if verbose:
        print(f"Retrieved the field measurements for site #{site} from {url}")
    # It may be desireable to keep the original na_values, like 'unkn' for many
    # of the columns. However, it is still a good idea to replace for the gage
    # depth and discharge values, since these variables get used in plotting
    # functions.
    (
        header,
        outputDF,
        columns,
        dtype,
    ) = hf.read_rdb(response.text)

    try:
        #The only difference between hf.field_meas() and field_meas2() is the format='ISO8601'
        outputDF.measurement_dt = pd.to_datetime(outputDF.measurement_dt, format='ISO8601')
    except ValueError as err:
        print(
            f"Unable to parse the measurement_dt field as a date. reason: '{str(err)}'."
        )
    # Add timezone, then convert to UTC to match discharge data. #new
    #field_table = field_table.tz_localize(tz='US/Eastern')       #new
    #field_table = field_table.tz_convert('utc')                  #new
    
    # An attempt to use the tz_cd column to make measurement_dt timezone aware.
    # outputDF.tz_cd.replace({np.nan: 'UTC'}, inplace=True)
    # def f(x, y):
    #    return x.tz_localize(y)
    # outputDF['datetime'] = outputDF[['measurement_dt', 'tz_cd']].apply(lambda x: f(*x), axis=1)

    outputDF.set_index("measurement_dt", inplace=True)
    
    # Convert to metric  #new
    length = 0.3048      #new
    area = 0.09290304    #new
    volume = 0.0283168   #new

    # see page for meaning of values: https://help.waterdata.usgs.gov/output-formats#streamflow_measurement_data #new

    outputDF.loc[:,'gage_height_va_m'] = outputDF.loc[:,'gage_height_va'] * length   #new
    outputDF.loc[:,'discharge_va_m3s'] = outputDF.loc[:,'discharge_va'] * volume      #new
    outputDF.loc[:,'gage_va_change_m'] = outputDF.loc[:,'gage_va_change'] * length   #new
    outputDF.loc[:,'chan_discharge_m3s'] = outputDF.loc[:,'chan_discharge'] * volume  #new
    outputDF.loc[:,'chan_width_m'] = outputDF.loc[:,'chan_width'] * length           #new
    outputDF.loc[:,'chan_area_m2'] = outputDF.loc[:,'chan_area'] * area              #new
    outputDF.loc[:,'chan_velocity_ms'] = outputDF.loc[:,'chan_velocity'] * length     #new
    
    return hf.hydroRDB(header, outputDF, columns, dtype, response.text)

class FieldMeas:
    """A class for working with the USGS field data."""
    def __init__(self, site, start=None, stop=None, file=None, verbose=True):
        self.site = site
        self.start = start
        self.stop = stop
        if file is None:
            file = f"field_meas_{site}.rdb.gz"
        self.file = file
        self.verbose = verbose
        self.url = f"https://waterdata.usgs.gov/nwis/measurements?site_no={site}&agency_cd=USGS&format=rdb_expanded"
        
        self.aliases = {
            'Q':'chan_discharge',
            'V':'chan_velocity',
            'W':'chan_width',
            'A':'chan_area',
            't':'index',
        }
        self.vars = ['Q', 'V', 'W', 'A', 't']
        
        try:
            # attempt to read file
            with gzip.open(self.file, "rt") as zip_file:
                self.text = zip_file.read()
            if verbose:
                print(f"Retrieved field measurements for site #{site} from {file}")
        except FileNotFoundError:
            # if file doesn't exist, request data from USGS
            headers = {"Accept-encoding": "gzip"}

            if verbose:
                print(f"Retrieving field measurements for site #{site} from {self.url}", end="\r")
                USGSresponse = hf.get_usgs_RDB_service(self.url, headers)
            if verbose:
                print(f"Retrieved field measurements for site #{site} from {self.url}  ")
            # self.text = the text that was returned
            self.text = USGSresponse.text
            # save text to file
            with gzip.open(self.file, "wt") as zip_file:
                zip_file.write(self.text)
        
        # now we can process the text
        (
            self.header,
            outputDF,
            self.columns,
            self.dtypes,
        ) = hf.read_rdb(self.text)
        
        # Additional processing of the dataframe
        self.historical_table = self._process_table(outputDF)
        
    @property
    def table(self):
        # Return a table that only includes valid information, using these criteria:
        # - only measurements within date range of start to stop
        #      - .table will automatically reflect changes to self.start and self.stop change
        #      - and get rid of measurements outside of study period
        #      - returns whole table if start & stop are None.

        # - exclude measurements where the gauge discharge is different from the measured discharge (Qerror > 0)
        error_rows = self.historical_table.loc[:,'Qerror'] > 0
        # - drop quantitative rows that have missing values
        _table = (self.historical_table
                      .loc[~error_rows]
                      .loc[self.start:self.stop, :]
                      .dropna(subset=[  # Drop rows that have missing values in the following fields
                              'discharge_va_m3s', 
                              'chan_discharge_m3s',
                              'gage_height_va_m',
                              'chan_width_m',
                              'chan_velocity_ms',
                              'chan_area_m2',
                            ])
                          )
        # Calculate Seddon celerity for this subset of the data
        # The rating curve (y:discharge x: stage) changes over time, so only fit a curve to short time periods.
        try:
            # c = (dQ/dh)/W by fitting a polynomial curve to field Q&stage data instead of using a rating curve, 
            # then take derivative, divide by width... Ponce 1989 p.285 & Ponce 2024
            S = _table.loc[:,'gage_height_va_m']  # Stage is h
            Q = _table.loc[:,'discharge_va_m3s']
            W = _table.loc[:,'chan_width_m']
            SQfit = np.polynomial.Polynomial.fit(S, Q, 2)
            dSQfit = SQfit.deriv()
            dQfromS = dSQfit(S)
            c2 = dQfromS / W
            _table.loc[:,'c_dQdSW_fd_ms'] = c2 #fd stands for field data
        except np.linalg.LinAlgError as e:
            print(f"Could not fit a polynomial to the field-collected stage-discharge data.",
                  "It might be possible to fit a curve with a shorter span of time.")
            print(e)
        return _table

    @property
    def a(self):
        return self.table.loc[:,'chan_area_m2']

    @property
    def avg_d(self):
        return self.table.loc[:,'avg_depth_m']

    @property
    def q(self):
        return self.table.loc[:,'chan_discharge_m3s']

    @property
    def s(self):
        return self.table.loc[:,'gage_height_va_m']

    @property
    def v(self):
        return self.table.loc[:,'chan_velocity_m2']

    @property
    def w(self):
        return self.table.loc[:,'chan_width_m']
    
    @property
    def error_rows(self):
        # Select rows with errors in them
        error_rows = self.historical_table.loc[:,'Qerror'] > 0
        return self.historical_table.loc[error_rows]
    
    def __repr__(self):
        t_rows, t_cols = self.table.shape
        h_rows, h_cols = self.historical_table.shape
        info_text = f"\n    .table: {t_rows} rows × {t_cols} columns\n    .historical_table: {h_rows} rows × {h_cols} columns"
        return f"FieldMeas(site={self.site}, start={self.start}, stop={self.stop}, file={self.file}, verbose={self.verbose}){info_text}"
        
    def _process_table(self, outputDF):
        """Check for errors, calculate metric values, more?"""
        zone = { # See https://help.waterdata.usgs.gov/code/tz_query?fmt=html
            'nan':'',
            '':'',
            'EST':'-5:00',
            'CST':'-6:00',
            'MST':'-7:00',
            'PST':'-8:00',
            'EDT':'-4:00',
            'CDT':'-5:00',
            'MDT':'-6:00',
            'PDT':'-7:00',
            'AST':'-4:00', #Atlantic Time for Puerto Rico, no DST
            'AKST':'-9:00',
            'AKDT':'-8:00',
            'HST':'-10:00', #No DST in Hawaii
            'GST':'+10:00', # Guam ST; no DST
        }
        try:
            #The only difference between hf.field_meas() and field_meas2() is the format='ISO8601'
            outputDF.measurement_dt = outputDF.apply(lambda x: x.measurement_dt+zone[str(x.tz_cd)], axis=1)
            outputDF.measurement_dt = pd.to_datetime(outputDF.measurement_dt, format='ISO8601', utc=True)
        except ValueError as err:
            print(
                f"Unable to parse the measurement_dt field as a date. reason: '{str(err)}'."
            )
    
        outputDF.set_index("measurement_dt", inplace=True)

        # sort by date of measurement
        outputDF.sort_index(inplace=True)
    
        # see page for meaning of values: https://help.waterdata.usgs.gov/output-formats#streamflow_measurement_data
        # Convert to metric
        length = 0.3048
        area = 0.09290304
        volume = 0.0283168

        outputDF.loc[:,'gage_height_va_m'] = outputDF.loc[:,'gage_height_va'] * length
        outputDF.loc[:,'discharge_va_m3s'] = outputDF.loc[:,'discharge_va'] * volume
        outputDF.loc[:,'gage_va_change_m'] = outputDF.loc[:,'gage_va_change'] * length
        outputDF.loc[:,'chan_discharge_m3s'] = outputDF.loc[:,'chan_discharge'] * volume
        outputDF.loc[:,'chan_width_m'] = outputDF.loc[:,'chan_width'] * length
        outputDF.loc[:,'chan_area_m2'] = outputDF.loc[:,'chan_area'] * area
        outputDF.loc[:,'chan_velocity_ms'] = outputDF.loc[:,'chan_velocity'] * length
        # avg_depth_m has produced values that seem impossibly large; use with caution
        outputDF.loc[:,'avg_depth_m'] = outputDF.loc[:,'chan_area_m2'] / outputDF.loc[:,'chan_width_m']
        
        # Calculate estimates of celerity
        
        # The linear estimate, using average depth (area/width)
        outputDF.loc[:,'c_linearAvgD_ms'] = (outputDF.loc[:,'avg_depth_m'] * 9.8)**0.5

        # The 5/3V estimate Ponce 1989 p.285
        outputDF.loc[:,'c_5V3_ms'] = (5 * outputDF.loc[:,'chan_velocity_ms']) / 3
    
        # Clean up columns
        # 'measurement_nu' is an index for the measurements at each site; when the sites are combined the indecies are no longer unique
        #outputDF.loc[:,'measurement_nu'] = outputDF.loc[:,'measurement_nu'].astype(str) #Convert to string #Why??
        # How long did the measurement take? convert decimal hours into pd.Timedelta; nan converted to zero.
        #outputDF.loc[:,'gage_Timedelta'] = pd.to_timedelta(outputDF.loc[:,'gage_va_time'])
        # Set when the measurement was over.
        #outputDF.loc[:,'gage_va_time_done'] = outputDF.index + outputDF.loc[:,'gage_Timedelta']
        #field_meas['gage_va_time_done'] = field_meas['measurement_dt'] + field_meas['gage_Timedelta']

        # Error processing: measured discharge should equal gauge discharge
        outputDF.loc[:,'Qerror'] = abs(outputDF.loc[:,'chan_discharge'] - outputDF.loc[:,'discharge_va'])

        return outputDF

    def plot(self, x, y):
        """Plot two variables"""
        fig, ax = plt.subplots()
        ax.plot(x, y, 'ko', label="Original Field Data")
        #ax.plot(x_line, velocity_estimator(x_line), 'r-', label="Fitted Curve")
        ax.set(xlabel = x.name, ylabel = y.name)
        ax.legend()
        return fig, ax


def calc_corr(us, ds, plot='full'):
    if plot == 'full':
        select = slice(None)
    if plot == 'half':
        select = slice(len(us)-1, -1)
    ccov = signal.correlate(ds-np.mean(ds), us-np.mean(us), mode='full', method='fft')[select]
    ccor = ccov / (max([len(ds), len(us)]) * np.nanstd(ds) * np.nanstd(us))
    lags = signal.correlation_lags(len(ds), len(us), mode='full')[select]
    return lags, ccor, ccov

def calc_stats(lags, ccor):
    ## Calc Stats
    indx = lambda x: np.where(lags==x)[0][0]
    # Find zero crossings. Crossings occur between elements. index & lags are before zero crossing.
    zeros_indx =  np.where(np.diff(np.sign(ccor)))[0]
    zeros_lags = lags[zeros_indx]
    # Get index of first lag in zeros_lags that is greater than zero, get lag
    first_zeros_lag = zeros_lags[np.where(zeros_lags > 0)[0][0]]  # lag of the first zero crossing

    stats = {
        'ccor_max_indx': ccor.argmax(),
        'ccor_max_lag': lags[ccor.argmax()],
        'ccor_max': ccor.max(),
        'lag_0_indx': indx(0),
        'lag_0_ccor':ccor[indx(0)],
        'lag_11_indx': indx(0),
        'lag_11_ccor':ccor[indx(0)],
        'zeros_indx': zeros_indx,
        'zeros_lags': zeros_lags,
        '1st_0_lag': first_zeros_lag,
    } 
    return stats

def plot_corr(lags, ccor, ax=None):
    # If user supplies an axis to draw to, then work with that ax and return it and an empty fig.
    fig=None
    if ax is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    ax.plot(lags, ccor)
    ax.axhline(0.5, ls=':')
    
    if fig is None:
        # If user supplied an ax to draw to, then only return the ax.
        return ax
    else:
        # If user didn't supply an ax, then create a whole figure and return fig & ax.
        return fig, ax
    
def xcorr2(us, ds, plot='full'):
    lags, ccor, ccov = calc_corr(us, ds, plot)
    stats = calc_stats(lags, ccor)
    fig, axes = plt.subplots(nrows=2)
    #time = us.index
    axes[0].plot(us)
    axes[0].plot(ds)
    axes[1] = plot_corr(lags, ccor, ax=axes[1])
    arrays = (lags, ccor, ccov)
    return fig, axes, arrays, stats

def new_xcorr(us, ds, plot='full'):
    if plot == 'full':
        select = slice(None)
    if plot == 'half':
        select = slice(len(us)-1, -1)
    ccov = signal.correlate(ds-np.mean(ds), us-np.mean(us), mode='full', method='fft')[select]
    ccor = ccov / (max([len(ds), len(us)]) * np.nanstd(ds) * np.nanstd(us))
    lags = signal.correlation_lags(len(ds), len(us), mode='full')[select]
    time = us.index

    fig, axes = plt.subplots(2, 1)
    fig.tight_layout()
    axes[0].plot(time, ds)
    axes[0].plot(time, us)
    #axes[1].plot(lags, ccov)
    axes[1].plot(lags, ccor)

    axes[0].axhline(0.5, ls=':')

    ## Calc Stats
    indx = lambda x: np.where(lags==x)[0][0]
    # Find zero crossings. Crossings occur between elements. index & lags are before zero crossing.
    zeros_indx =  np.where(np.diff(np.sign(ccor)))[0]
    zeros_lags = lags[zeros_indx]
    # Get index of first lag in zeros_lags that is greater than zero, get lag
    first_zeros_lag = zeros_lags[np.where(zeros_lags > 0)[0][0]]  # lag of the first zero crossing
    #for zero_lag in stats['zeros_lags']:  # plot every zero crossing
    #    axes[1].axvline(zero_lag)
    axes[1].plot(first_zeros_lag, 0, 'rx', label="First zero crossing")

    stats = {
        'ccor_max_indx': ccor.argmax(),
        'ccor_max_lag': lags[ccor.argmax()],
        'ccor_max': ccor.max(),
        'lag_0_indx': indx(0),
        'lag_0_ccor':ccor[indx(0)],
        'lag_11_indx': indx(0),
        'lag_11_ccor':ccor[indx(0)],
        'zeros_indx': zeros_indx,
        'zeros_lags': zeros_lags,
        '1st_0_lag': first_zeros_lag,
    }  

    arrays = (lags, ccor, ccov)
    return fig, axes, arrays, stats

def periodogram2(data, sample_period='15 minutes', x_scale='log', y_scale='log', x_lim='auto', y_lim='auto', labels=None, title="Periodogram", ax=None):
    """Calculates and plots the spectral density of an array: a filter, a timeseries, an autocorrelation, a cross-correlation...

    A periodogram is a graph of a FFT of an autocorrelation.

    Parameters
    ----------
    data: Pandas dataframe 
        A time series of values.


    To Do
    -----
    - Accept multiple series.
    - 
    
    """
    period = pd.Timedelta(sample_period).total_seconds()
    fs = 1/period

    freqs = {}
    psds = {}

    # If user supplies an axis to draw to, then work with that ax and return it and an empty fig.
    fig=None
    if ax is None:
        fig, ax = plt.subplots()

    try:
        data = data.to_frame()
    except AttributeError:
        pass # if data is a series, convert it to a dataframe. If it is a dataframe already, do nothing.
    
    for i, col in enumerate(data.columns):
        if labels is None:
            label = col
        elif len(data.columns) != len(labels):
            raise  ValueError(f"The labels parameter  (len = {len(labels)}) is not the same length as the number of data series to be plotted (len = {len(data.columns)}).")
        else:
            label = labels[i]
        
        freqs[col], psds[col] = signal.welch(data[col], fs, window='blackman', nperseg=1024)

        ax.plot(freqs[col], psds[col], label=label)

    # Time units in seconds
    second = 1
    minute = 60 * second
    hour = 60 * minute
    day = 24 * hour
    week = 7 * day
    year = 365.25 * day
    month = year/12

    # Add reference lines
    ax.axvline(1/hour, color='red', linestyle=':')
    ax.text(1/hour, 0.8, 'hour', rotation=90, transform=ax.get_xaxis_text1_transform(0)[0])
    ax.axvline(1/day, color='red', linestyle=':')
    ax.text(1/day, 0.8, 'day', rotation=90, transform=ax.get_xaxis_text1_transform(0)[0])
    ax.axvline(1/week, color='red', linestyle=':')
    ax.text(1/week, 0.4, 'week', rotation=90, transform=ax.get_xaxis_text1_transform(0)[0])
    ax.axvline(4/year, color='red', linestyle=':')
    ax.text(4/year, 0.4, '3 months', rotation=90, transform=ax.get_xaxis_text1_transform(0)[0])

    if x_lim == 'auto':
        ax.autoscale('x')
    elif isinstance(x_lim, list):
        ax.set_xlim(x_lim)
    else:
        ax.set_xlim([0.0000001, 0.00056])

    if y_lim == 'auto':
        ax.autoscale('y')
    elif isinstance(y_lim, list):
        ax.set_xlim(y_lim)
    else:
        ax.set_ylim([0.001, 50000])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.set_title(title)
    ax.legend()
    
    if fig is None:
        # If user supplied an ax to draw to, then only return the ax.
        return ax
    # If user didn't supply an ax, then create a whole figure and return fig & ax.
    return fig, ax

def spectrogram2(input_series, sample_period='15 minutes'):
    """ Plot a spectrogram for a data series.
    """
    period = pd.Timedelta(sample_period).total_seconds()
    fs = 1/period
    
    # Original signal
    #fs, N = 200, 1001  # 200 Hz sampling rate for 5 s signal
    #t_z = np.arange(N) / fs  # time indexes for signal
    #z = np.exp(2j*np.pi*70 * (t_z - 0.2*t_z**2))  # complex-valued chirp

    N = len(input_series)
    #fs = 1/(15 * minute) # Sample rate in Hz is 1 sample every 900 seconds, or 0.00111
    t_z = np.arange(N) / fs  # time indexes for signal

    #nperseg, noverlap = 50, 40
    nperseg, noverlap = 4024, 4000

    win = ('gaussian', 1e-2 * fs)  # Gaussian with 0.01 s standard dev.
    # Legacy STFT:
    #f0_u, t0, Sz0_u = stft(z, fs, win, nperseg, noverlap,
    #                       return_onesided=False, scaling='spectrum')
    #f0, Sz0 = fftshift(f0_u), fftshift(Sz0_u, axes=0)

    # New STFT:
    SFT = signal.ShortTimeFFT.from_window(win, fs, nperseg, noverlap, fft_mode='centered',
                               scale_to='magnitude', phase_shift=None)
    Sz1 = SFT.stft(input_series)
    # Plot results:
    fig, ax = plt.subplots(1, 1, sharex='all', sharey='all',
                         figsize=(6., 4.))  # enlarge figure a bit
    t_lo, t_hi, f_lo, f_hi = SFT.extent(N, center_bins=True)
    #axx[0].set_title(r"Legacy stft() produces $%d\times%d$ points" % Sz0.T.shape)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(f_lo, f_hi)
    ax.set_title(r"ShortTimeFFT produces $%d\times%d$ points" % Sz1.T.shape)

    second = 1
    minute = 60 * second
    hour = 60 * minute
    
    ax.set_xlabel(rf"Time $t$ in seconds ($\Delta t= {SFT.delta_t:g}\,$s {(SFT.delta_t)/(hour):.1f} hours)")
    # Calculate extent of plot with centered bins since
    # imshow does not interpolate by default:
    dt2 = (nperseg-noverlap) / fs / 2  # equals SFT.delta_t / 2
    df2 = fs / nperseg / 2  # equals SFT.delta_f / 2
    #extent0 = (-dt2, t0[-1] + dt2, f0[0] - df2, f0[-1] - df2)
    extent1 = SFT.extent(N, center_bins=True)
    kw = dict(origin='lower', aspect='auto', cmap='viridis')
    #im1a = axx[0].imshow(abs(Sz0), extent=extent0, **kw)
    im1b = ax.imshow(abs(Sz1), extent=extent1, **kw)
    fig.colorbar(im1b, ax=ax, label="Magnitude $|S_z(t, f)|$")
    _ = fig.supylabel(r"Frequency $f$ in Hertz ($\Delta f = %g\,$Hz)" %
                   SFT.delta_f, x=0.08, y=0.5, fontsize='medium')
    #plt.show()
    
    return fig, ax