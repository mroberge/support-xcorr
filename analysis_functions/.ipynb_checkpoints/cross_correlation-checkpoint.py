"""
analysis_funtions.cross_correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains functions that calculate cross-correlation.

-----
"""
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset as pandas_to_offset
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as mpl_tools_inset

import scipy.signal as signal
from scipy.signal import butter, bessel
from scipy.signal import sosfiltfilt
from scipy.signal import detrend as sig_detrend
import gzip

import hydrofunctions as hf

def xcorr_calc(us, ds, plot='full'):
    """Calculate the correlation matrix for an upstream and downstream series.

    Parameters:
    ----------
    us, ds : list or pandas.DataFrame or numpy.array
        The two matricies to correlate. For autocorrelation, enter the same series for both us and ds.
        For cross-correlation, 'us' should be the initial 'uspstream' signal and 'ds' should be the delayed 'downstream' signal.

    plot : str {'full', 'half'}
        How much of the correlation matrix should be returned. Can be either 'full' or 'half'. 
        'full': returns a matrix that is as long as len(us) + len(ds).
        'half': returns the positive side of the matrix, from zero to the end.

    Returns:
    -------
    lags : np.array
        A matrix containing the lag number for each position in the ccor and ccov matricies. The center
        value is zero, with every value to the left being negative, and everything to the right being positive.
        If the 'plot' parameter = 'half', then only the positive and the zero lags will be returned.

    ccor, ccov : np.array
        The correlation and covariance matricies. These  will be the same length as the `lags` matrix.


    Notes:
    -----
    After some testing, this is the fastest implementation of the cross-correlation algorithm because it takes advantage of
    the Fast Fourier Transform, which cuts calculation times down considerably.
    """
    if plot == 'full':
        select = slice(None)
    elif plot == 'half':
        # Only return the positive side of the matrix, from matrix[0, end]
        select = slice(len(us)-1, -1)
    else:
        raise ValueError(f"plot should be 'half' or 'full'. You entered {plot}.")
    
    # Calculate the cross-variance from the normalized series: series - mean(series)
    ccov = signal.correlate(ds-np.mean(ds), us-np.mean(us), mode='full', method='fft')[select]
    ccor = ccov / (max([len(ds), len(us)]) * np.nanstd(ds) * np.nanstd(us))
    lags = signal.correlation_lags(len(ds), len(us), mode='full')[select]
    return lags, ccor, ccov


def xcorr_stats(lags, ccor):
    """Calculate a variety of statistics from the correlation matrix.

    Returns:
    -------
    stats : dict
        A dictionary with a variety of statistics, including:

        'ccor_max_indx': The position within the ccor matrix that has the highest value.
        'ccor_max_lag': The lag with the highest correlation. For autocorrelation, this will be zero. For
            cross-correlation, this will hopefully correspond to the delay in the signal.
        'ccor_max': the maximum correlation. Will be 1.0 for autocorrelation.
        'lag_0_ccor': the correlation at zero lags.
        'lag_11_ccor': the correlation at eleven lags. This could be a measure of how quickly autocorrelation
            decreases from the peak. Lag 11 was chosen because that was the expected time delay for Marty's test
            data.
        'zeros_lags': The lags for every zero crossing, both positive and negative.
        '1st_0_lag': The lag of the first positive zero crossing. This is a measure of how much autocorrelation
            is present in the signals. Smaller values indicate less autocorrelation.

    Notes:
    -----
    If 'ccor_max_lag' is negative, it means that the downstream signal from 'ds' actually preceeds the signal from 'us'.
    This may mean that you have mixed up us & ds, or that something that you weren't expecting happened...

    Zero crossings occur between elements in the ccor matrix. 'zeros_lags' and '1st_0_lag' return the lag *BEFORE* the
    crossing.
    """
    # A function for looking up the index value when given a lag.
    indx = lambda x: np.where(lags==x)[0][0]
    # Find zero crossings. Crossings occur between elements. index & lags are before zero crossing.
    zeros_indx =  np.where(np.diff(np.sign(ccor)))[0]
    zeros_lags = lags[zeros_indx]
    # Get index of first lag in zeros_lags that is greater than zero, get lag
    try:
        first_zeros_lag = zeros_lags[np.where(zeros_lags > 0)[0][0]]  # lag of the first positive zero crossing
    except IndexError as e:
        first_zeros_lag = np.nan

    stats = {
        'ccor_max_indx': ccor.argmax(),
        'ccor_max_lag': lags[ccor.argmax()],
        'ccor_max': ccor.max(),
        'lag_0_indx': indx(0),
        'lag_0_ccor':ccor[indx(0)],
        'lag_11_indx': indx(11),
        'lag_11_ccor':ccor[indx(11)],
        'zeros_indx': zeros_indx,
        'zeros_lags': zeros_lags,
        '1st_0_lag': first_zeros_lag,
    } 
    return stats

def xcorr_plot(lags, ccor, plot='full', ax=None):
    """Plot a correlation matrix.

    Parameters:
    ----------
    lags, ccor : numpy.matrix
        Two matricies corresponding to the lag number and correlation matrix.

    plot : str {'full' or 'half'} or tuple
        How much of the correlation matrix to plot. Default is 'full', which plots the entire matrix.
        Other options include 'half', which plots all of the positive lags from zero to the end of the matrix.
        If a tuple is sent, then the first value is the lag to start plotting from, and the second value is
        the lag where the plotting should stop. For example, plot=(0, 1000) will plot from lag zero to 1,000.

     ax : omatplotlib.axes, optional
         The function will plot to this ax if it is provided. If it is not provided (default), this function will 
         create a matplotlib figure and draw to the associated ax.
    """
    if plot == 'full':
        select = slice(None)
    elif plot == 'half':
        # Only return the positive side of the matrix, from matrix[0, end]
        select = slice(len(us)-1, -1)
    else:
        # Create a function for looking up the index value when given a lag.
        indx = lambda x: np.where(lags==x)[0][0]
        try:
            # if plot=(lag_a, lag_b), then we will plot from lag_a to lag_b.
            start_lag = plot[0]
            stop_lag = plot[1]
            start_indx = indx(start_lag)
            stop_indx = indx(stop_lag)
            select = slice(start_indx, stop_indx)
        except TypeError as e:
            raise ValueError(f"The plot parameter should be 'half', 'full', or a tuple of lags, like `plot=(4, 23)`. You entered {plot}.") from e
    
    # If user supplies an axis to draw to, then work with that ax and return it and an empty fig.
    fig=None
    if ax is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    ax.plot(lags[select], ccor[select])
    #ax.axhline(0.5, ls=':')
    
    if fig is None:
        # If user supplied an ax to draw to, then only return the ax.
        return ax
    else:
        # If user didn't supply an ax, then create a whole figure and return fig & ax.
        return fig, ax
    
def xcorr2(us, ds, plot='full'):
    """Calculate a cross-correlation on two series, calcuate statistics, and plot.

    Parameters:
    ----------
    us, ds : list or pandas.DataFrame or numpy.array
        The two matricies to correlate. For autocorrelation, enter the same series for both us and ds.
        For cross-correlation, 'us' should be the initial 'uspstream' signal and 'ds' should be the delayed 'downstream' signal.
    
    plot : str {'full' or 'half'} or tuple
        How much of the correlation matrix to plot. Default is 'full', which plots the entire matrix.
        Other options include 'half', which plots all of the positive lags from zero to the end of the matrix.
        If a tuple is sent, then the first value is the lag to start plotting from, and the second value is
        the lag where the plotting should stop. For example, plot=(0, 1000) will plot from lag zero to 1,000. 

    Returns:
    -------
    fig : matplotlib figure
    axs : matplotlib axes
    arrays : tuple of arrays
        lags
        ccor
        ccov
    stats : dict of statistics
    """
    lags, ccor, ccov = xcorr_calc(us, ds, plot)
    stats = xcorr_stats(lags, ccor)
    fig, axs = plt.subplots(nrows=2)
    #time = us.index
    axs[0].plot(us)
    axs[0].plot(ds)
    axs[1] = xcorr_plot(lags, ccor, ax=axes[1])
    arrays = (lags, ccor, ccov)
    return fig, axs, arrays, stats

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
        x.freq = pd.to_timedelta(pandas_to_offset(pd.infer_freq(x)))
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
    
        ax_xc_inset = mpl_tools_inset.inset_axes(ax_xc, width="40%",height="50%",loc=1)
        ax_xc_inset.set_xlim(-100, 100)
        ax_xc_inset.set_ylim(0, 1.1)
        ax_xc_inset.plot([0, 0], [-1, 1], color='r', linestyle='-', linewidth=1)
        ax_xc_inset.plot(lags, ccor)
        ax_xc_inset.plot(maxlag, maxccor,'r+', markersize=12, markeredgewidth=2)
        annotation = f'lag: {maxlag}  xcorr: {maxccor:.2f}'
        ax_xc_inset.annotate(annotation, xy=(maxlag, maxccor), xytext=(3, 9), textcoords='offset points')
        mpl_tools_inset.mark_inset(ax_xc, ax_xc_inset, 2, 4)
    
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
        data.index.freq = pd.to_timedelta(pandas_to_offset(pd.infer_freq(data.index)))
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
                axins = mpl_tools_inset.inset_axes(ax1, width="40%", height="50%", loc=1)
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
                mpl_tools_inset.mark_inset(ax1, axins, 2, 4)
            plt.show()
            plt.close()
    
    
    return results

