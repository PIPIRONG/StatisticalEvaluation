import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stat 

from lmfit.models import LognormalModel, GaussianModel
from scipy.stats import gstd # to set param hint in lmfit
from scipy.stats import gmean # to set param hint in lmfit


def get_pkl_paths():
    data_path = code_path + "\\data_input\\"
    pkl_paths = [data_path + f for f in os.listdir(data_path) if f.endswith(".pkl")]
    return pkl_paths


def show_data_example(anypath):
    data = pd.read_pickle(anypath)
    return data


def apply_evaluation_parameters(one_pkl_path:dict, grainproperty:str):

    # read pkl dictionary
    my_dict = pd.read_pickle(one_pkl_path)
    
    # read parameters which determine which grains need to be removed
    scores = np.asarray(my_dict["scores"])
    distances = np.asarray(my_dict["mask_distance_to_image_border"]) 
    
    # hold grains whose distance/scores are above both distance/scores threshold
    is_relevant_data = (distances > dth) & (scores > sth)
    
    # iterate throught dictionary
    for key, value in my_dict.items():
        if key.startswith(grainproperty):
            if not key == 'polygon_dihedral_angle':
                data = np.asarray(value, dtype=float)
                data = data[is_relevant_data]
                data = data[~np.isnan(data)]
    
                if key.endswith('area'):
                    data = data *(nm_px**2)
                if not key.startswith('polygon'):
                    if not key.endswith('area'): 
                        data = data*nm_px
            else:
                # Some arrays from the data are ragged 
                # (i.e. non-uniform number of columns per row).
                # With `dtype=object` this is no problem.
                data = np.asarray(value,  dtype=object)
                data = data[is_relevant_data]
                # special treatment of key 'polygon_dihedral_angle',
                # since it contains multiple angles per instance/row
                # (ragged array needs to be unfold, so that 1 row contains 1 value)
                global dihedral
                dihedral = [] # empty list
                for angles in data: 
                    # continue iteration if one row contains NoneType object
                    if angles is None:
                        continue
                    # unfold the ragged array to a list with one value per row
                    dihedral+= angles # type object
                    
                # convert the class'list' to a class 'numpy.ndarray''    
                dihedral= np.array(dihedral, dtype=float)
                data = dihedral
                             
            mean = np.mean(data)

    return data, mean



def calculate_statistics(single_image_means, x=None, y=None):
    
    # statistic for field measurements or
    # statistic for individual grain measurements
    
    # number of observations nobs
    nobs = len(single_image_means)
    
    minimum = min(single_image_means)
    maximum = max(single_image_means)
    
    # complete range = largest value - smallest value
    cran = maximum-minimum
    
    if nobs > 1:
        # calculate the mean  
        mean = np.mean(single_image_means)
        
        # calculate s-std using Bessel's function (sample population with n-1 )
        sample_standart_deviation = np.std(single_image_means, ddof=1)
            
        # calculate the confidence interval ci level with the t-distribution
        number_of_images = len(single_image_means)
        t_value = pd.read_excel(code_path + '\\t-value.xlsx', index_col=0)
        
        if not nobs > 4428: 
            t = t_value.loc[number_of_images]['t']
        else:
            t = 1.96
        
        # calculate 95% confidence interval limit
        ci = (sample_standart_deviation / (np.sqrt(number_of_images))) * t
        
        # calculate relative accuracy ra
        ra = ci*100/mean
        
        # statistic beyond ASTM procedure
        median = np.median(single_image_means)
        # interquartile percentile
        q75,q50, q25  = np.percentile(single_image_means, [75,50,25])
        iqr = q75-q25
        
        try:
            mode = (x[y.argmax()])
        except:
            mode =None
        skewness = stat.skew(single_image_means, bias=False)
        kurtosis = stat.kurtosis(single_image_means, bias=False) 
        
        
        statistics = [nobs, 
                      mean, median, mode,
                      sample_standart_deviation, minimum, maximum, cran,
                      q25, q75, iqr,
                      skewness,kurtosis,
                      ci, ra]
    
    else:        
        statistics = ['No Statistic for 1 image' for n in range(15)]
      
    return statistics

  



def iterate_grain_property(pkl_paths:list, grainproperty:str):
    
    data = []
    
    single_image_means = []
    
    for n in range(len(pkl_paths)):
         
        # apply the thresholds to data and calculate mean of single-images
        data_and_means = apply_evaluation_parameters(pkl_paths[n], grainproperty)
                                                
        # append single-image data to a list containing arrays in each row
        data.append(data_and_means[0])
        
        # append single-image mean to a list containg floats in each row
        single_image_means.append(data_and_means[1])
        

    # create one array containing all values of multiple SEM-Images in one array
    multiple_images_data = np.concatenate(data) 
  
    return single_image_means, multiple_images_data, data
 


   

def iterate_composed_grain_properties(pkl_paths:list, 
                                      grainproperty1:str,  
                                      grainproperty2:str):
    list_of_means = []
    
    for n in range(len(pkl_paths)):
         
        # apply the thresholds to data and calculate mean
        data_and_mean_1 = apply_evaluation_parameters(pkl_paths[n],
                                                grainproperty1)
        data_and_mean_2 = apply_evaluation_parameters(pkl_paths[n],
                                                grainproperty2)                                        
        
        # calculate composite mean of single image data
        aspect_ratio = data_and_mean_2[0]/data_and_mean_1[0]
        aspect_ratio_mean = np.mean(aspect_ratio)
        
        # creata a list containing arrays of all 
        list_of_means.append(aspect_ratio_mean)
        
    
    # calculate the total mean of the single-image means and statistics according to ASTM
    if len(list_of_means) > 1:
            ASTM_statistics = calculate_statistics(list_of_means)
    else:
        print("ASTM calculation is NoneType, because only one SEM-image (which equals one pkl-file)"
              "is analyzed. To get ASTM statistics insert more pkl files into data folder \n ")
        ASTM_statistics = None
            
    return ASTM_statistics



def figure_customization(lw:float = 1.5, ms:float = 5.5,
                        top:float  = 0.85 , bot: float = 0.2,
                        lt:float = 0.25, rt:float = 0.95,
                        ws:float = 0.4, hs:float = 0.4):
    '''
    Parameters
    ----------
    lw : float, optional
        Linewidth. The default is 0.5.
    ms : float, optional
        markersize. The default is 1.5.
    top : float, optional
        edge of top axis is fractional units. The default is 0.9
    bot : float, optional
        edge of bottom axis is fractional units. The default is 0.1
    lt : float, optional
        edge of left axis is fractional units. The default is 0.15
    rt : float, optional
        edge of right axis is fractional units. The default is 0.85

    Returns
    -------
    None.

    '''
    
    # Customize linewidth of scale lwsc
    lwsc = 1.5
    
    ## Customizing LINES and LEGEND in plt.plot
    plt.rc('lines', linewidth=lw, markersize=ms)
    # Customie standart marker:
    plt.rcParams['scatter.marker']='x'
    
    # typical use
    # linewithd 2 marker 1 |linewidth 0.5 marker 1.5
    plt.rc('legend', frameon=False)

    ## Customizing TICKS of x-axis in plt.plot
    plt.rc('xtick', top=True, direction='in')
    plt.rcParams['xtick.labelsize'] = 'xx-large'  # Größe der Zahlen der x-Achsen
    plt.rcParams['xtick.major.size'] = 10  # Länge des Striches 6
    plt.rcParams['xtick.major.width'] = lwsc  # Breite des Striches 1
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.minor.size'] = 5  # Länge des Striches 3
    plt.rcParams['xtick.minor.width'] = lwsc  # Breite des Striches 1

    ## Customizing TICKS of y-axis in plt.plot
    plt.rc('ytick', right=True, direction='in')
    plt.rcParams['ytick.labelsize'] = 'xx-large'  # Größe der Zahlen der x-Achsen
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = lwsc
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = lwsc
    
    # Customize thickness of axis spines top, bottom, left, right
    plt.rcParams['axes.linewidth']=lwsc
     
    # Customize size of label
    plt.rcParams['legend.fontsize']='xx-large'
    
    # Customize ????
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    ## Customizing dots per inch (auflösung)
    plt.rcParams['figure.dpi'] = 600

    ## Customizing FONTSIZE of x and y labels in plt.plot
    plt.rcParams['axes.labelsize'] = 'xx-large'

    ## Customizing PAPERSIZE
    plt.rcParams['ps.papersize'] = 'letter'

    ## Customizing FONT of Text to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    ## Customizing where edges of axes start and end in fractional units
    plt.rcParams['figure.subplot.left']= lt
    plt.rcParams['figure.subplot.right']= rt
    plt.rcParams['figure.subplot.top']= top
    plt.rcParams['figure.subplot.bottom']= bot

    ## Customizing space(width or height) between axes as fraction of axis dimension
    plt.rcParams['figure.subplot.wspace']= ws
    plt.rcParams['figure.subplot.hspace']= hs

    return



def bin_the_data(grainprop, binwidth):
    # for comparability: select a specific binwidth over all data 
    # this function generates easily readable and uniform bins in the plot
    
    # find minimum & maximum value of grainprop
    minimum = int(min(grainprop))
    maximum = int(max(grainprop))
    
    # generate easily readable and uniform bin in the plot
    if binwidth != 1:
        
        # get order of magnitude of minimum value
        if minimum != 0:
            ord_of_mag = math.floor(math.log(minimum,10))
        else:
            ord_of_mag = 0
            
        # get the first digit of the number
        first_digit = int(minimum / pow(10, ord_of_mag))
        
        # generate uniform bins
        bin_start = first_digit*10**ord_of_mag
        # start all bin at 0
        bin_start = 0
        bin_stop = maximum + binwidth
        bins = list(range(bin_start, bin_stop, binwidth))
    else:
        bin_start = minimum - binwidth
        bin_stop = maximum + binwidth
        bins = [x+0.5 for x in range(bin_start, bin_stop, binwidth)]
    
    return bins


def get_binned_data_points(binval):
    # y axis: get the value of the number density of the bins 
    y = binval[0]
    # x axis: calculate the middle of the bin from the bin edges
    x = [(binval[1][n]+binval[1][n+1])/2 for n in range(len(binval[1])-1)]
    x = np.array(x)
    return x,y


def plot_grain_property(grainprop, binwidth, unit, axislabels, color, name, key):

    # bin the data by deciding the bin edges
    bins = bin_the_data(grainprop, binwidth)

    # plot a histogram of the data    
    fig, ax = plt.subplots()
    binval = ax.hist(grainprop, bins = bins, density='True',
                      edgecolor='k', color=color)
    
    # Customize style of y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    
    
    # get the x and y values of the bins in the histogram
    x,y = get_binned_data_points(binval)
       
       
    # customize the x-axis and y-axis labels
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1]) 
    
    # get the number of individuals within the grainproperty
    num_of_indiv = len(grainprop)
    
    #Area plots have a very long right skewed tail
    if key.endswith('area'):
        ax.set_xlim(left =-100000 ,right=3000000)
        
    
    # customize the text/label within the figures
    if not key == 'polygon_dihedral_angle':
        xtxt = 0.95 
        ytxt = 0.85
    else:
        xtxt = 0.45 
        ytxt = 0.90
    
    lastbin = str(bins[-1])
    ax.text(xtxt, ytxt, 
                  r"$n_{img} =$" + str(num_of_sem_img) + "\n" 
                  r"$n_{obs} = $"+ str(num_of_indiv) + "\n" 
                  r"$bw=$" + str(binwidth) + ' ' + str(unit) + "\n" 
                  r"$bins \in$[" + str(bins[0]) + ',' + lastbin + "]\n"
                  r"$thr_{sc}>$ " + str(sth) + "\n"
                  r"$thr_{dist}>$" + str(dth) + ' px',
                  ha='right', va='top',
                  transform=ax.transAxes,
                  fontsize='large',)

    return x,y


def customize_grain_characteristic(key):
    
    if key == 'mask_feret_diameter_max':
        color = 'red'
        name = 'feret_max'
        xlabel = r"Maximum Feret diameter $d_{Fmax}$ $\left[\mathrm{nm}\right]$"
           
    if key == 'mask_feret_diameter_min':
        color = 'orange'
        name = 'feret_min'
        xlabel = r"Minimum Feret diameter $d_{Fmax}$ $\left[\mathrm{nm}\right]$"
     
    if key == 'mask_axis_length_max':
        color = 'b'
        name = 'ellipse_max'
        xlabel = r"Major diameter of ellipse $d_{Emax}$ $\left[\mathrm{nm}\right]$"
    
    if key == 'mask_axis_length_min':
        color = 'dodgerblue'
        name = 'ellipse_min'
        xlabel = r'Minor diameter of ellipse  $d_{Emin}$ $\left[\mathrm{nm}\right]$'
    
    ylabel = r"Probability density $f \left(d\right)$ $\left[\frac{1}{\mathrm{nm}}\right]$"
    unit ='nm'   
    
    if key == 'mask_perimeter':
        color = 'gold'
        name = 'perimeter'
        xlabel =  r'Circumference $C$ $\left[\mathrm{nm}\right]$'
        ylabel = r"Probability density $f \left(C\right)$ $\left[\frac{1}{\mathrm{nm}}\right]$"
        unit ='nm' 
        
    if key == 'mask_area':
        color = 'limegreen'
        name = 'area'
        xlabel = r'Area $A$ $\left[\mathrm{nm^2}\right]$'
        ylabel = r'Probability density $f \left(A\right)$ $\left[\frac{1}{\mathrm{nm^2}}\right]$'
        unit = r'nm$^2$'
    
    if key == 'polygon_area':
        color ='green'
        name = 'area_polygon'
        xlabel = r'Polygonarea $A_{P}$ $\left[\mathrm{nm^2}\right]$'
        ylabel = r'Probability density $f \left(A\right)$ $\left[\frac{1}{\mathrm{nm^2}}\right]$'
        unit = r'nm$^2$'

    if key == 'polygon_coordination_number':
        color = 'indigo'
        name = 'coordination number'
        xlabel = r'Coordination number $N_C$'
        ylabel = r'Probability density $f \left(N_C\right)$' 
        unit = ' '
        
    if key == 'polygon_dihedral_angle':
        color = 'violet'
        name = 'dihedral angle'
        xlabel = r'Dihedral angle $\phi$ $\left[\mathrm{^°}\right]$'
        ylabel = r'Probability density $f \left(\phi\right)$ $\left[\frac{1}{\mathrm{^°}}\right]$'
        unit = '°'
    
    axislabels = [xlabel,ylabel] 
    return unit, axislabels, color, name
 
       
def calculate_statistics_from_numeric_values(data,x,y):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    statistics : list
        [0] mode
        [1] median,
        [2] skewness,
        [3] kurtosis
        
    '''
   
    median = np.median(data)
    # take the most frequently value  
    mode = (x[y.argmax()])
    skewness = stat.skew(data, bias=False)
    kurtosis = stat.kurtosis(data, bias=False)  

    statistics = [median,mode,skewness,kurtosis]
    # geometric_mean = stat.gmean(data)
    # geometric_std = stat.gstd(data)
        
    return statistics


def save_table(name, dictionary):
     dictionary.style.format(precision=2).to_latex(name + '.tex')
     dictionary.to_excel(name + '.xlsx') 
     return print('Table'+ name +' is saved.')



def fit_lognormal_pdf_to_binned_data(grainprop, x,y, name):

    # load a lognormal model for curve-fitting
    log_mod = LognormalModel()
    
    # parameters: suggest inital parameters
    log_mod.set_param_hint('amplitude', value= 1, vary = False)
    log_mod.set_param_hint('center', value=np.log(gmean(grainprop)))
    log_mod.set_param_hint('sigma', value=np.log(gstd(grainprop)))
    
    # parameters: construct intial parameters from the suggested ones
    log_initparams = log_mod.make_params()
    
    # fitting: fit the binned data points
    log_results = log_mod.fit(y, log_initparams, x=x)
    
    # results: save the fitting report
    report = log_results.fit_report() #min_correl=0.25
    write_report = open('fit_report_' + str(name) + '.txt', 'w')
    write_report.write(report)
    write_report.close()

    # display markers at the used binned datapoints
    plt.scatter(x, y, c='k')
    
    max_bin = max(grainprop)
    # create a new axis in order to show a smooth fitting function
    x_eval = np.linspace(0.0001, int(max_bin + binwidth), int(max_bin*2))
    
    # evaluate the model with best parameter
    y_eval = log_mod.eval(log_results.params, x=x_eval)
    
    # plot the final/ best parametral function
    plt.plot(x_eval, y_eval, '--', c= 'k',  lw=3,
              label='Lognormal fit') 

    plt.legend(loc='upper right', fontsize=14) 
    return log_results

def statistics_from_lognormal_model(log_results):

     # results: get the calculated parameters
     calc_params = log_results.params.valuesdict()     
     sigma = calc_params['sigma']
     center = calc_params['center']
     amplitude = calc_params['amplitude']

     # results: calculations of lognormal PDF 
     mean = np.exp(center+sigma**2/2)
     median = np.exp(center)
     mode = np.exp(center-sigma**2)
     variance = (np.exp(sigma**2)-1)*np.exp(2*center+sigma**2)
     skew = (np.exp(sigma**2)+2)*np.sqrt((np.exp(sigma**2)-1))
     kurt = np.exp(4*sigma**2) + 2*np.exp(3*sigma**2) + 3*np.exp(2*sigma**2)-6
     
     # results: get the error of the center and sigma 
     center_stderr = log_results.params['center'].stderr
     sigma_stderr = log_results.params['sigma'].stderr
     
     # results: propagation of error  
     mean_err = np.sqrt((np.exp(center+sigma**2/2)*center_stderr)**2
                        +(sigma*np.exp(center*sigma**2/2))**2)
     median_err = np.exp(center)*center_stderr
     mode_err = np.sqrt((np.exp(center-sigma**2)*center_stderr)**2
                        +(-2*np.exp(center-sigma**2)*sigma_stderr)**2)
     skew_err = 3*np.exp(2*sigma**2)*sigma/np.sqrt(np.exp(sigma**1)-1)*sigma_stderr
     kurt_err = 4*sigma*np.exp(2*sigma**2)*(2*np.exp(2*sigma**2)
                                           +3*np.exp(sigma**2)
                                           +3) * sigma_stderr
     
     statistics = [center,
                   center_stderr, 
                   sigma,
                   sigma_stderr,
                   mean,
                   median,
                   mode,
                   skew,
                   kurt,
                   mean_err,
                   median_err,
                   mode_err,
                   skew_err,
                   kurt_err ]
     
     
     return statistics

  
#***************************************************************************
# main Loop - start
#***************************************************************************
def main_loop(key, bw:int = 200):
    
  
    global pkl_paths, num_of_sem_img
    # creata a list containing the paths of all pkl files
    pkl_paths = get_pkl_paths()
    # show the number of pkl_path which equals the number of SEM images
    num_of_sem_img = len(pkl_paths)
    # decide the values of the thresholds and scale
    # sth, dth, nm_px = decide_evaluation_parameters()

    os.chdir(data_output)
    global example_data
    # dictionary of the specified pkl-path as example
    example_data = show_data_example(pkl_paths[0])
    
    # apply the thresholds and scale to data. Choose a key of the dictionary
    imagewise, grainwise, listdata = iterate_grain_property(pkl_paths, key)
    
    print(key, ': Thresholds are applied')
    
    #*********************************************************************
    # PLOTTING OF HISTOGRAMM AND FITTING IT
    #*********************************************************************
   
    # design general figure appearance    
    figure_customization()
    
    # design grain specialized figure appearance
    unit, axislabels, color, name = customize_grain_characteristic(key)
    
    global binwidth
    binwidth = bw
    
    # plot histogram and return the midpoints from the binned data
    x,y = plot_grain_property(grainwise, binwidth, 
                              unit, axislabels, color, name, key)
    print(key, ': Histogramm is plotted')
    
    #fitting of lognormal
    if not key =="polygon_coordination_number" and not key =="polygon_dihedral_angle":
            parameter = fit_lognormal_pdf_to_binned_data(grainwise, x, y, name)
            print(key, ': Fitting of lognormal function succeded \n')
            statistics_model = statistics_from_lognormal_model(parameter)
    else:
        print(key, ': No Fitting is supposed for this distribution \n')
        statistics_model = None
        
    # saving figure    
    plt.savefig('figure_' + name + '.png')
    plt.savefig('figure_' + name + '.pdf')
    
    #*********************************************************************
    # STATISTICS
    #*********************************************************************

    # statistic image-wise according to ASTM
    statistic_imagewise  = calculate_statistics(imagewise) 
    
    # statistic grain-wise on individual grain
    statistic_grainwise = calculate_statistics(grainwise,x,y)  
    #highorder_stat = calculate_statistics_from_numeric_values(grainwise, x, y)
    #statistic_grainwise_plus = np.concatenate([statistic_grainwise,highorder_stat])
   
    
    return name, grainwise, statistic_imagewise, statistic_grainwise ,statistics_model, imagewise 

#***************************************************************************
# main Loop - end
#***************************************************************************





  
#***************************************************************************
# MAIN PROGRAM - start
#***************************************************************************
def main():
    
    # get data from folder
    global code_path, data_input, data_output
    code_path = os.path.abspath(os.getcwd()) 
    data_input = code_path + '\\data_input\\' 
    data_output = code_path + '\\data_output\\' 
    
    # create directory if it does not exist
    if not os.path.exists(data_input):
        os.makedirs('data_input')
    if not os.path.exists(data_output):
        os.makedirs('data_output')
    
    
    # decide upon threshold, scale and binwidths
    global sth, dth, nm_px
    # score threshold
    sth = 0.7
    # distance threshold
    dth = 10
    # scale nanometer per pixel
    nm_px = 1000/160
    # decide binwidth for various diamaters
    bw = 200
    # decide binwidth for various areas
    bw_area = 100000
    # decide binwidth for perimeter or circumference
    bw_peri = 500
    # decide binwidth for coordination number (cn)
    bw_cn = 1
    # decide binwidth for dihedral angle (dh)
    bw_dh = 10
    
    # get data, remove grain with low score and at image border
    feret_max = main_loop('mask_feret_diameter_max', bw=bw)
    feret_min = main_loop('mask_feret_diameter_min',bw=bw)
    ellipse_max = main_loop('mask_axis_length_max',bw=bw)
    ellipse_min = main_loop('mask_axis_length_min',bw=bw)
    perimeter = main_loop('mask_perimeter', bw= bw_peri)
    area = main_loop('mask_area', bw = bw_area)
    area_polygon = main_loop('polygon_area', bw = bw_area)
    coordination_number = main_loop('polygon_coordination_number', bw=bw_cn)
    dih_ang = main_loop('polygon_dihedral_angle',bw=bw_dh)
    
    # put data in a list
    all_data = []
    
    # check if variable exists and fill in a list if it does
    try:
        all_data.append(feret_max)
        all_data.append(feret_min) 
        all_data.append(ellipse_max)
        all_data.append(ellipse_min)
        all_data.append(perimeter)
        all_data.append(area)
        all_data.append(area_polygon)
        all_data.append(coordination_number)
        all_data.append(dih_ang)
    except:
        print('Some grain size characteristic (of 9) is missing your evaluation.\n'
              'Look inside the folder "data_output" for the results\n')
    else: 
        print('All 9 grain size characteristics are evaluated.\n'
               'Look inside the folder "data_output" for the results\n')
         
    # get names raw data and statistics of all grain characteristics
    names = [all_data[n][0] for n in range(len(all_data))]
    grainmeasures = [all_data[n][1] for n in range(len(all_data))]
    statistics_imagewise = [all_data[n][2] for n in range(len(all_data))]
    statistics_grainwise = [all_data[n][3] for n in range(len(all_data))]
    statistics_modeled = [all_data[n][4] for n in range(len(all_data))]
    
    
    index_statistics = ['nobs', 
              r'Mean $\bar{x}$', 
              r'Median $x_{50}$', 
              r'Mode $\hat{x}$',
              r'Sample Std. Dev. $s$', 
              r'Minimum $Min(x_i)$', 
              r'Maximum $Max(x_i)$',
              r'Range $R$',
              r'First Quartile $Q_{25}$', 
              r'Third Quartile $Q_{75}$',
              r'Interquartile range $IQR$',
              r'Coeff. of Skewness $G_1$',
              r'Coeff. of Kurtosis $G_2$',
              r'95\% CI $\Delta \bar{x}$',
              r'\% Rel. Accuracy $\%RA$']
    
    
    # Save IMAGE-WISE Statistics
    summary1 = dict(zip(names, statistics_imagewise))  
    summary_statistic_imagewise = pd.DataFrame(data = summary1,
                               index=index_statistics)
    
    # Save GRAIN-WISE Statistics
    summary2 = dict(zip(names, statistics_grainwise)) 
    summary_statistic_grainwise = pd.DataFrame(data = summary2,
                                                index = index_statistics)
    
    # Save RawData in one column for each grain size measure
    summary3 = dict(zip(names, grainmeasures))
    # allow to create a Daraframe with arrays of different length
    summary3_allow_arr_diff_len = dict([ (k,pd.Series(v)) for k,v in summary3.items() ])
    summary_rawdata_concatenation = pd.DataFrame(data = summary3_allow_arr_diff_len)
    summary_rawdata_concatenation.index = np.arange(1, len(summary_rawdata_concatenation) + 1)    
    

    # Save MODELED Statistics
    index_modeled = [ r'Center &$\mu$',
                      r'Error of Center &$\Delta\mu$', 
                      r'Width &$\sigma$',
                      r'Error of Width &$\Delta\sigma$',
                      r'Mean &$\bar{x}$', 
                      r'Median &$x_{50}$',
                      r'Mode &$\hat{x}$',
                      r'Skewness &$G_1$',
                      r'Excess Kurtosis &$G_2$',
                      r'Error of Mean &$\Delta\bar{x}$',
                      r'Error of Median &$\Delta x_{50}$',
                      r'Error of Mode &$\Delta \hat{x}$',
                      r'Error of Skewness &$\Delta G_1$',
                      r'Error of Excess Kurtosis &$\Delta G_2$',
                      ]
    
    summary4 = dict(zip(names, statistics_modeled))
    summary_statistic_modeled= pd.DataFrame(data = summary4,
                                                index = index_modeled)
    
    
    # save txt file containing all used inputs
    with open('00_Sinter-Evaluation-Report.txt', 'w') as f:
        f.writelines('score-threshold: ' + str(sth) +'\n')
        f.writelines('distance_threshold: ' + str(dth)+'\n')
        f.writelines('scale_nanometers_per_pixel: ' + str(nm_px) +'\n')
        f.writelines('number of SEM images: ' + str(num_of_sem_img) +'\n')
        
        f.writelines('\nbinwidth for diamaters: ' + str(bw) +'\n')
        f.writelines('binwidth for area: ' + str(bw_area) +'\n')
        f.writelines('binwidth for perimeter: ' + str(bw_peri) +'\n')
        f.writelines('binwidth for coordination number: ' + str(bw_cn) +'\n')
        f.writelines('binwidth for dihedral angle: ' + str(bw_dh) +'\n')
        
        
        f.writelines('\nEvaluated grain size characteristic: \n')
        f.write('\n'.join(names) + '\n' )
        f.writelines('\npkl-paths used: \n')
        f.write('\n'.join(pkl_paths) + '\n' )
        
    # save tables in the summaries folder  
    save_table('summary_statistic_imagewise', summary_statistic_imagewise) 
    save_table('summary_statistic_grainwise', summary_statistic_grainwise)             
    save_table('summary_rawdata_concatenation', summary_rawdata_concatenation) 
    save_table('summary_statistic_modeled', summary_statistic_modeled)
    
    # TODO 
    # plot polygons visualization inside SEM-image  

    return summary_statistic_imagewise, summary_statistic_grainwise, summary_rawdata_concatenation, summary_statistic_modeled
#***************************************************************************
# MAIN PROGRAM - start
#***************************************************************************
sum_imagegwise,sum_grainwise,sum_rawdata,sum_fit =  main()




