import matplotlib.pyplot as plt    
import numpy as np
from pdb import set_trace
import iris
import matplotlib as mpl

def sqidge_01s(X, logXmin):
    if logXmin is not None: X = logXmin + X*(1 - 2*logXmin)
    return(X)

def Bayes_benchmark(filename_out, fig_dir, Y, X, lmask, logXmin = None, logYmin = None, ax = None):
    X = X.data.flatten()[lmask]
    prob = Y[1].data.flatten()[lmask]
    Y = np.array([Y[0][i].data.flatten()[lmask] for i in range(Y[0].shape[0])]).T
    pos = np.mean(Y < X[:, np.newaxis], axis=1)
    
    def NME(x, y, step1_only = False):  
        def nme(xi, yi): return np.sum(np.abs(xi - yi))/np.sum(np.abs(xi - np.mean(xi)))
    
        step1 = nme(x, y)
        if step1_only: return step1
        x = x - np.mean(x)
        y = y - np.mean(y)
        step2 = nme(x, y)
    
        x = x /np.mean(np.abs(x))
        y = y /np.mean(np.abs(y))
        step3 = nme(x, y)

        return step1, step2, step3

    def run_NME_over_subset(Xi, Yi, percentiles = None):
        if percentiles is not None:
            lower = np.percentile(Xi[Xi>0], percentiles[0])
            upper = np.percentile(Xi[Xi>0], percentiles[1])
            #if percentiles[0] == 0: lower = np.min(Xi)
            test = (Xi >= lower) & (Xi <= upper)
            X = Xi[test]
            Y = Yi[test, :]
        else:
            X = Xi
            Y = Yi
        M = Y.shape[1]
        nme_results = np.zeros([M, 3])
        for i in range(M): nme_results[i, :] = NME(X, Y[:, i])
        
        nme_median = NME(X, np.median(X), step1_only = True)
        
        XR = X.copy()
        nme_random = np.zeros(100)
        for i in range(100): 
            np.random.shuffle(XR)
            nme_random[i] = NME(X, XR, True)
    
        min_val = np.min(nme_results)
        max_val = np.max(nme_results)
        bins = np.linspace(min_val, max_val, 50)

        for i in range(3): plt.hist(nme_results[:, i], bins=bins, alpha=0.5, label=f'NME step {i+1}') 
        plt.hist(nme_random, bins=bins, color = 'black', alpha=0.5, label='RR Null Model') 
        
        if percentiles is None:
            median_lab = 'Median Null Model'
            mean_lab =  'Mean Null Model'
        else:         
            median_lab = ''
            mean_lab = ''
        plt.gca().axvline(1, linestyle='dotted', color='red', label=mean_lab)
        plt.gca().axvline(nme_median, linestyle='dotted', color='blue', label=median_lab)
        for i in range(3):
            nme10 = np.percentile(nme_results[:,i], [10])[0]
            nme90 = np.percentile(nme_results[:,i], [90])[0]
            if percentiles is None:
                plt.text(0.05, 0.95 - 0.1 * i, 
                        f'NME{i} 10-90%: {nme10:.2f}-{nme90:.2f}', 
                        transform=plt.gca().transAxes)
            else:
                plt.text(0.05, 0.95 - 0.1 * i, 
                        f'{nme10:.2f}-{nme90:.2f}', transform=plt.gca().transAxes)

        
    
    plt.figure(figsize=(8, 12))  # Set the figure size
    plt.subplot(3, 1, 1)  # Create the density plot in the top subplot
    run_NME_over_subset(X, Y, None)

    
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    percentiles = [0, 1, 5, 10, 25, 75, 90, 95, 99, 100]

    for i in range(len(percentiles) - 1):
        plt.subplot(6, 3, i + 10)
        run_NME_over_subset(X, Y, percentiles[i:(i+2)])
    
    plt.savefig(fig_dir + filename_out + '-NME_scores.png') 
    plt.clf()
    plt.close()

    pos = pos[X > 0]
    X = X[X>0]
    scatter_metric_overall_and_percentiles(X, pos)
    plt.savefig(fig_dir + filename_out + '-posterior-position.png')
    plt.clf()
    plt.close()

def scatter_metric_overall_and_percentiles(X, pos, xlabel = 'Burnt Area', 
                                           ylabel = 'Posterior Position', 
                                           y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           cmap = 'afmhot_r', norm =mpl.colors.LogNorm(),
                                           percentiles = [0, 1, 5, 10, 25, 75, 
                                                          90, 95, 99, 100]):
    plt.figure(figsize=(8, 12))  # Set the figure size
    plt.subplot(3, 1, 1)  # Create the density plot in the top subplot

    Xf = np.log10(X)
    posf = pos
    plot_id = plt.gca().hist2d(Xf, posf, bins = 100, cmap = cmap, norm = norm)
    plt.gcf().colorbar(plot_id[3], ax=plt.gca())
    at = np.unique(np.round(np.arange(np.min(Xf), np.max(Xf))))
    plt.xticks(at, 10**at)
    labels = np.array(y_ticks)
    plt.yticks(labels, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
   
    mean_y = np.mean(pos)
    plt.text(0.05, 0.85, f'Mean Y: {mean_y:.2f}', transform=plt.gca().transAxes)
        
    

    for i in range(len(percentiles) - 1):
        plot0 = 10 #if i > 7 else 11
        plt.subplot(6, 3, i + plot0)  # Create subplots in the bottom row
        lower = np.percentile(X, percentiles[i])
        upper = np.percentile(X, percentiles[i + 1])
        y_subset = pos[(X >= lower) & (X <= upper)]
        plt.hist(y_subset, bins=30, alpha=0.7)
    
        mean_y = np.mean(y_subset)
        plt.text(0.05, 0.7, f'Mean Y: {mean_y:.2f}', transform=plt.gca().transAxes)
        
        #plt.xlabel(f'Y ({percentiles[i]}-{percentiles[i+1]}% X)')
        plt.text(0.05, 0.85, f'Obs ({percentiles[i]}-{percentiles[i+1]}%):', 
                 transform=plt.gca().transAxes)
    
        #if i == 0 or i == 5: plt.ylabel('Frequency')

    plt.gcf().text(0.5, 0.04, ylabel, ha='center', fontsize=12)
    plt.gcf().text(0.04, 0.25, 'Frequency', ha='center', fontsize=12, rotation=90)
    

def BayesScatter(X, Y, lmask = None, logXmin = None, logYmin = None, ax = None):
    pc_size = 10
    percentiles = np.arange(pc_size, 100, pc_size) 
    percentiles = np.array([1, 2, 5, 10, 25, 75, 90, 95, 99])
    #import iris.plot as iplt
    #import iris.quickplot as qplt 
    #set_trace()  
    if lmask is  None: 
        Y = np.percentile(Y, q = percentiles, axis = 0).T       
    else:
        X = X.data.flatten()[lmask]
        Y = Y.collapsed('realization', iris.analysis.PERCENTILE, percent = percentiles)
        Y = np.array([Y[i].data.flatten()[lmask] for i in range(Y.shape[0])]).T
    
    if len(X) > 5000:
        select = np.random.choice(len(X), 1000)
        X = X[select]
        Y = Y[select, :]
    
    X = sqidge_01s(X, logXmin)
    Y = sqidge_01s(Y, logYmin)
    if ax is None: fig, ax = plt.subplots()
    
    ncols = int(Y.shape[1]/2)
    line_widths = np.linspace(0.2, 2, ncols)
    
    alpha = 1.0/(Y.shape[1])

    for i in range(ncols):        
        ax.vlines(X, ymin = Y[:,i], ymax = Y[:, -i-1],
                  linewidth = line_widths[i],
                  alpha = alpha, color = 'black')
    
    vals = [min(np.min(X), np.min(Y)),  max(np.max(X), np.max(Y))]
    ax.plot(vals, vals, color = 'blue', linestyle = '--')
    #ax.scatter(X, Y[:, ncols + 1], s = 0.2, color = 'red')
    if logYmin is not None: ax.set_yscale('logit')
    if logXmin is not None: ax.set_xscale('logit')

    labels = np.array([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5, 1-10**-1])
    plt.yticks(labels)
    plt.xticks(labels)

    plt.xlabel("Observation")
    plt.ylabel("Simulation")
