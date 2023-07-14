import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import pickle as pkl
from collections import OrderedDict
import pdb

kB = 1.3806503E-23   # Boltzmann constant, [J/K]
NA = 6.022E23   # Avogadro's number, number of particles in one mole
hbar = 1.055E-34   # reduced Plancks constant, [J s]
planck = hbar*2*np.pi   # planck's constant , [J s]
G0 = np.pi**2*kB**2*0.170/(3*planck)*1E12   # an inherent G at 170 mK; pW/K
bolos=np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])   # this is just always true
mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

# bolotest geometry 
L = 220   # bolotest TES leg length, um
A_U = 7*420E-3; A_W = 5*400E-3; A_I = 7*400E-3  # Area of film on one leg, um^2
wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
AoL_bolo = A_bolo/L   # A/L for bolotest devices


##### Supporting Functions for G_layer analysis ##########

### scale G wrt temp
def scale_G(T, GTc, Tc, n):
    return GTc * T**(n-1)/Tc**(n-1)

def sigma_GscaledT(T, GTc, Tc, n, sigma_GTc, sigma_Tc, sigma_n):   
    Gterm = sigma_GTc * T**(n-1)/(Tc**(n-1))
    Tcterm = sigma_Tc * GTc * (1-n) * T**(n-1)/(Tc**(n))   # this is very tiny
    nterm = sigma_n * GTc * (T/Tc)**(n-1) * np.log(T/Tc)
    return np.sqrt( Gterm**2 + Tcterm**2 + nterm**2)   # quadratic sum of sigma G(Tc), sigma Tc, and sigma_n terms

### G - kappa conversion
def GtoKappa(G, A, L):   # thermal conductance, area, length
    return G*L/A

def KappatoG(kappa, A, L):   # thermal conductance in pW/K/um, area in um^2, length in um
    return kappa*A/L

def Gfromkappas(kappas, dsub, lw, ll, layer='total', fab='legacy'):   # thickness of substrate, leg width, and leg length in um
    # predicts G from kappa measurements and bolo geometry
    # thickness of wiring layers is independent of geometry
    # assumes all four legs have a wiring stack

    kappaU, kappaW, kappaI = kappas   # pW/K/um, our measured values

    # bolotest W thinner from process sheet
    if fab=='legacy': dW1 = .190; dI1 = .350; dW2 = .400; dI2 = .400   # film thicknesses, um
    elif fab=='bolotest': dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
    else: print('Invalid fab type, choose "legacy" or "bolotest."')    
    
    w1w, w2w = wlw(lw, fab='legacy')
    GU_pred = KappatoG(kappaU, lw*dsub, ll)   # predicted G from substrate layer
    GW_pred = KappatoG(kappaW, w1w*dW1, ll) + KappatoG(kappaW, w2w*dW2, ll)   # predicted G from Nb layers
    GI_pred = KappatoG(kappaI, lw*dI1, ll) + KappatoG(kappaI, lw*dI2, ll)   # predicted G from nitride layers
    if layer=='total': return 4*(GU_pred+GW_pred+GI_pred)
    elif layer=='wiring': return 4*(GW_pred+GI_pred)
    elif layer=='U': return 4*(GU_pred)
    else: print('Invalid layer type.'); return

### G from alpha model
def wlw(lw, fab='bolotest', layer='wiring'):
    # calculate W layer widths for bolotest or legacy data
    # INPUT lw in um
    # W2 width = W1 width + 2 um = leg width - 4um, and any W layer !< 1um (W2 !< 3 um for microstrip, !< 1um for single W1 layer)
    if fab=='legacy':
        w1w=8*np.ones_like(lw); w2w=5*np.ones_like(lw)
        if np.ndim(lw)!=0: # accommodate leg widths smaller than default W1 width
            smallw = np.where(lw<=8)[0]   
            w1w[smallw]=5; w2w[smallw]=3   
        elif lw<=8:
            w1w=5; w2w=3

    elif fab=='bolotest':
        w1w=5*np.ones_like(lw); w2w=3*np.ones_like(lw)   # W layer widths for leg widths => 7 um 
        maxw1w = 5; minw1w = 1 if layer=='W1' else 2   # single W layer can span 1-5 um
        if np.ndim(lw)!=0:
            w1w0 = lw-1   # naive W1 width estimate
            naninds = np.where(w1w0<minw1w)[0]   # W1 width  !< min width, return nans
            scaleinds = np.where((minw1w<=w1w0)&(w1w0<maxw1w))[0]   # layer is < normalized width, scale 
            if len(naninds)!=0: w1w[naninds] = np.nan; w2w[naninds] = np.nan
            if len(scaleinds)!=0: w1w[scaleinds] = w1w0[scaleinds]; w2w[scaleinds] = w1w0[scaleinds]-2   # um
        elif lw-2<minw1w:   # handle single leg widths
            w1w = np.nan; w2w = np.nan
        elif lw-2<maxw1w:
            w1w = lw-2; w2w = lw-4   # um 

    return w1w, w2w

def G_layer(fit, d, layer='U'):
    # fit = [fit parameters, fit errors (if calc_sigma=True)], thickness d is in um
    # RETURNS prediction (and error if 'fit' is 2D)

    # numdevs = 1 if np.isscalar(d) else len(d)   # number of devices
    # d = np.array([d]) if np.isscalar(d) else np.array(d)   # handle thickness scalars and arrays
    d = d if np.isscalar(d) else np.array(d)   # handle thickness scalars and arrays

    if layer=='U': linds=np.array([0,3]); d0=.420   # substrate layer parameter indexes and default thickness in um
    elif layer=='W': linds=np.array([1,4]); d0=.400   # Nb layer parameter indexes and default thickness in um
    elif layer=='I': linds=np.array([2,5]); d0=.400   # insulating layer parameter indexes and default thickness in um

    # if np.isscalar(d):
    if len(fit.shape)==2:   # user passed fit parameters and errors
        G0, alpha = fit[0][linds]; sig_G0, sig_alpha = fit[1][linds]   # parameters for layer x
        Glayer = G0 * (d/d0)**(alpha+1) 
        sig_Glayer = sig_G0**2 * (((d/d0)**(alpha+1)))**2 + (G0*(alpha+1)*d**alpha)**2 * sig_alpha**2
        # return np.array([Glayer, sig_Glayer]).reshape((2, numdevs))
        return np.array([Glayer, sig_Glayer])
    elif len(fit.shape)==1:   # user passed only fit parameters 
        G0, alpha = fit[linds]
        Glayer = G0 * (d/d0)**(alpha+1) 
        # return np.array([Glayer]).reshape((1, numdevs))
        return Glayer
    else: 
        print('Fit array has too many dimensions, either pass 1D array of parameters or 2D array of parameters and param errors.')
        return

# def Gbolos(params):   
#     GU, GW, GI, aU, aW, aI = params
#     wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    
#     bolo1b = 4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4)
#     bolo24 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5) + GI*((350/400)**(1 + aI) + 1)
#     bolo23 = GU*(2 + 2*((220+120)/420)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2)
#     bolo22 = GU*(3 + ((220+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3)
#     bolo21 = GU*(1 + 3*((280+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI))  
#     bolo20 = 4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI)
#     bolo7 = GU*(3 + ((280+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) 
#     bolo13 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI)
    
#     return np.array([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13])

def Gbolotest(fit, layer='total'):
    # returns G_TES for bolotest data set given fit parameters
    # assumes bolotest geometry
    # can return full substrate + microstrip, just substrate, just microstrip, or an individual W / I layer

    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20

    # options to isolate U, W, I, and microstrip layers
    if layer=='total':   
        include_U = 1; include_W = 1; include_I = 1
    elif layer=='U':
        include_U = 1; include_W = 0; include_I = 0
    elif layer=='wiring':
        include_U = 0; include_W = 1; include_I = 1
    elif layer=='W':
        include_U = 0; include_W = 1; include_I = 0
    elif layer=='I':
        include_U = 0; include_W = 0; include_I = 1
    else:
        print('Unknown layer'+layer+'. Options include "total", "wiring", "U", "W", and "I".')

    G_legA = G_layer(fit, 0.420, layer='U')*include_U + G_layer(fit, 0.160, layer='W')*include_W + G_layer(fit, 0.350, layer='I')*include_I + G_layer(fit, 0.340, layer='W')*3/5*include_W + G_layer(fit, 0.400, layer='I')*include_I # S-W1-I1-W2-I2
    G_legB = G_layer(fit, 0.420, layer='U')*include_U + G_layer(fit, 0.160, layer='W')*include_W + G_layer(fit, 0.350, layer='I')*include_I + G_layer(fit, 0.285, layer='W')*3/5*include_W   # S-W1-I1-W2
    G_legC = G_layer(fit, 0.400, layer='U')*include_U + G_layer(fit, 0.350, layer='I') + G_layer(fit, 0.340, layer='W')*3/5*include_W + G_layer(fit, 0.400, layer='I')*include_I   # S-I1-W2-I2
    G_legD = G_layer(fit, 0.420, layer='U')*include_U + G_layer(fit, 0.160, layer='W')*include_W + G_layer(fit, 0.270 + 0.400, layer='I')*include_I   # S-W1-I1-I2 (I stack)
    G_legE = G_layer(fit, 0.420, layer='U')*include_U + G_layer(fit, 0.100 + 0.285, layer='W')*wstack_width/5*include_W   # S-W1-W2 (W stack)
    G_legF = G_layer(fit, 0.400, layer='U')*include_U + G_layer(fit, 0.400 + 0.270, layer='I')*include_I   # S-I1-I2 (I stack)
    G_legG = G_layer(fit, 0.340, layer='U')*include_U   # bare S 

    G_1b = 4*G_legA
    G_24 = 1*G_legA + 3*G_legG
    G_23 = 2*G_legA + 2*G_legG
    G_22 = 3*G_legA + 1*G_legG
    G_21 = 1*G_legB + 3*G_legF
    G_20 = 1*G_legB + 3*G_legE
    G_7 = 2*G_legA + 1*G_legC + 1*G_legD
    G_13 = 1*G_legB + 3*G_legG

    if len(fit.shape)==2:   # return values and errors
        Gbolos = np.array([G_1b[0], G_24[0], G_23[0], G_22[0], G_21[0], G_20[0], G_7[0], G_13[0]]); sigma_Gbolos = np.array([G_1b[1], G_24[1], G_23[1], G_22[1], G_21[1], G_20[1], G_7[1], G_13[1]])
        return Gbolos, sigma_Gbolos
    else:
        return np.array([G_1b, G_24, G_23, G_22, G_21, G_20, G_7, G_13])   # return values
    

def Gbolos_six(params):   
    # returns G for alpha six-layer model 

    GSiO, GSiN, GW, GI, aSiN, aW, aI = params    
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    
    bolo1b = 4*GSiO + 4*GSiN + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4)
    bolo24 = 4*GSiO + GSiN*(1 + 3*((220+120)/420)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5*3/5) + GI*((350/400)**(1 + aI) + 1)
    bolo23 = 4*GSiO + GSiN*(2 + 2*((220+120)/420)**(1 + aSiN)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2)
    bolo22 = 4*GSiO + GSiN*(3 + ((220+120)/420)**(1 + aSiN)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3)
    bolo21 = 4*GSiO + GSiN*(1 + 3*((280+120)/420)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI))  
    bolo20 = 4*GSiO + 4*GSiN + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI)  
    bolo7 = 4*GSiO + GSiN*(3 + ((280+120)/420)**(1 + aSiN)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) 
    bolo13 = 4*GSiO + GSiN*(1 + 3*((220+120)/420)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) 
    
    return np.array([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13])

### fitting free parameters of model
def chisq_val(params, data, model='default'):   # calculates chi-squared value
    ydata, sigma = data
    if model=='default':
        Gbolos_model = Gbolotest(params)   # predicted G of each bolo
    elif model=='six_layers':   # model SiOx as it's own layer
        Gbolos_model = Gbolos_six(params)
    chisq_vals = (Gbolos_model-ydata)**2/sigma**2

    return np.sum(chisq_vals)

def calc_func_grid(params, data):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = chisq_val(params_rc, data)
    return func_grid

def runsim_chisq(num_its, p0, data, bounds, plot_dir, show_yplots=False, save_figs=False, fn_comments='', save_sim=False, sim_file=None, model='default'):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (most likely pW/K)

    print('\n'); print('Running Minimization MC Simulation'); print('\n')
    ydata, sigma = data

    pfits_sim = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    Gwires = np.empty((num_its, 1))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        it_result = minimize(chisq_val, p0, args=[y_its[ii], sigma], bounds=bounds)
        pfits_sim[ii] = it_result['x']
        Gwires[ii] = Gfrommodel(pfits_sim[ii], 0.420, 7, 220, layer='wiring', fab='bolotest')[0,0]/4   # function outputs G for four legs worth of microstrip

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            n, bins, patches = plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 0.9*max(n)))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata' + fn_comments + '.png', dpi=300) 
        
    # sort & print results    
    sim_params = np.mean(pfits_sim, axis=0); sim_std = np.std(pfits_sim, axis=0)
    U_sim, W_sim, I_sim, aU_sim, aW_sim, aI_sim = sim_params   # parameter fits from Monte Carlo Function Minimization
    Uerr_sim, Werr_sim, Ierr_sim, aUerr_sim, aWerr_sim, aIerr_sim = sim_std   # parameter errors from Monte Carlo Function Minimization
    Gwire = np.mean(Gwires); Gwire_std = np.std(Gwires)

    print('Results from Monte Carlo Sim - chisq Min')
    print('G_U(420 nm) = ', round(U_sim, 2), ' +/- ', round(Uerr_sim, 2), 'pW/K')
    print('G_W(400 nm) = ', round(W_sim, 2), ' +/- ', round(Werr_sim, 2), 'pW/K')
    print('G_I(400 nm) = ', round(I_sim, 2), ' +/- ', round(Ierr_sim, 2), 'pW/K')
    print('alpha_U = ', round(aU_sim, 2), ' +/- ', round(aUerr_sim, 2))
    print('alpha_W = ', round(aW_sim, 2), ' +/- ', round(aWerr_sim, 2))
    print('alpha_I = ', round(aI_sim, 2), ' +/- ', round(aIerr_sim, 2))
    print('G_microstrip = ', round(Gwire, 2), ' +/- ', round(Gwire_std, 2), 'pW/K')
    print('')

    if save_sim:
        results_dict = {}
        results_dict['sim'] = pfits_sim   # add simulation 
        results_dict['Gwires'] = Gwires   
        results_dict['sim_params'] = {}
        results_dict['sim_params']['num_its'] = num_its   
        results_dict['sim_params']['p0'] = p0  
        results_dict['fit'] = {}
        results_dict['fit']['fit_params'] = sim_params   
        results_dict['fit']['fit_std'] = sim_std  
        results_dict['fit']['Gwire'] = Gwire  
        results_dict['fit']['sigma_Gwire'] = Gwire_std  
        results_dict['fit']['Gwire'] = Gwire   # pW/K
        results_dict['fit']['Gwire_std'] = Gwire_std   # pW/K
        print('Saving simulation to ', sim_file); print('\n')
        with open(sim_file, 'wb') as outfile:   # save simulation pkl
            pkl.dump(results_dict, outfile)

    return sim_params, sim_std


### visualize and evaluate quality of fit
def qualityplots(data, sim_dict, plot_dir='./', save_figs=False, fn_comments='', vmax=500, figsize=(17,5.75), title='', print_results=True, calc='mean', spinds=[], plot=True):
    ### plot chisq values in 2D parameter space (alpha_x vs G_x) overlayed with resulting parameters from simulation for all three layers
    # params can be either the mean or median of the simulation values
    # spinds are indexes of a certain subpopulation to plot. if the length of this is 0, it will analyze the entire population. 

    layers = np.array(['U', 'W', 'I'])
    simdata_temp = sim_dict['sim']
    if len(spinds)==0: spinds = np.arange(np.shape(simdata_temp)[1])   # allow for analyzing only subsections of simulation data 
    sim_data = simdata_temp[spinds]
    Gwires = sim_dict['Gwires'][spinds]

    # calculate the fit params as either the mean or median of the simulation values
    if calc == 'mean':
        fit_params, fit_errs = [np.mean(sim_data, axis=0), np.std(sim_data, axis=0)]   # take mean values
        Gwire = np.mean(Gwires); sigma_Gwire = np.std(Gwires)
    if calc == 'median':
        fit_params, fit_errs = [np.median(sim_data, axis=0), np.std(sim_data, axis=0)]   # take median values to avoid outliers
        Gwire = np.median(Gwires); sigma_Gwire = np.std(Gwires)

    if plot:
        xgridlim=[0,2]; ygridlim=[0,2]   # alpha_layer vs G_layer 
        xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:150j, ygridlim[0]:ygridlim[1]:150j]   # make 2D grid for plotter
        wspace = 0.25

        fig = plt.figure(figsize=figsize)   # initialize figure
        fig.subplots_adjust(wspace=wspace, left=0.065)

        for ll, layer in enumerate(layers):
            xlab = '\\textbf{G}$_\\textbf{'+layer+'}$'
            ylab = '$\\boldsymbol{\\alpha_\\textbf{'+layer+'}}$'
            if layer=='U': 
                Gind=0; aind=3   # G and alpha indexes in parameters array
                gridparams = np.array([xgrid, fit_params[1]*np.ones_like(xgrid), fit_params[2]*np.ones_like(xgrid), ygrid, fit_params[4]*np.ones_like(ygrid), fit_params[5]*np.ones_like(ygrid)]).T
                splot_ID = '\\textbf{i.}'
            elif layer=='W': 
                Gind=1; aind=4   # G and alpha indexes in parameters array
                gridparams = np.array([fit_params[0]*np.ones_like(xgrid), xgrid, fit_params[2]*np.ones_like(xgrid), fit_params[3]*np.ones_like(ygrid), ygrid, fit_params[5]*np.ones_like(ygrid)]).T
                splot_ID = '\\textbf{ii.}'
            elif layer=='I': 
                Gind=2; aind=5   # G and alpha indexes in parameters array
                gridparams = np.array([fit_params[0]*np.ones_like(xgrid), fit_params[1]*np.ones_like(xgrid), xgrid, fit_params[3]*np.ones_like(ygrid), fit_params[4]*np.ones_like(ygrid), ygrid]).T
                splot_ID = '\\textbf{iii.}'

            funcgrid = calc_func_grid(gridparams, data)   # calculate chisq values for points in the grid
            ax = fig.add_subplot(1,3,ll+1)   # select subplot
            im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=100, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.6)   # quality plot
            plt.errorbar(fit_params[Gind], fit_params[aind], xerr=fit_errs[Gind], yerr=fit_errs[aind], color='black', label='\\textbf{Model Fit}', capsize=2, linestyle='None')   # fit results
            plt.xlabel(xlab); plt.ylabel(ylab)
            plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
            plt.annotate(splot_ID, (0.1, 1.825), bbox=dict(boxstyle="square,pad=0.3", fc='w', ec='k', lw=1))
            if ll==2: 
                axpos = ax.get_position()
                cax = fig.add_axes([axpos.x1+0.02, axpos.y0+0.04, 0.01, axpos.y1-axpos.y0-0.08], label='\\textbf{Chi-Sq Value}')
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('$\\boldsymbol{\\chi^2\\textbf{ Value}}$', rotation=270, labelpad=20)
                # ax.legend(loc='lower left')
                # ax.legend(loc=(0.075,0.75))
                ax.legend(loc=(0.1,0.15))
        plt.suptitle(title, fontsize=20, y=0.86)
        if save_figs: plt.savefig(plot_dir + 'qualityplots' + fn_comments + '.png', dpi=300)   # save figure

    if print_results:
        ### print results
        GmeasU, GmeasW, GmeasI, alphaU, alphaW, alphaI = fit_params; sigGU, sigGW, sigGI, sigalphaU, sigalphaW, sigalphaI = fit_errs
        print ('\n\nResults taking '+ calc +' values of fit parameters:')
        print('G_U(420 nm) = ', round(GmeasU, 2), ' +/- ', round(sigGU, 2), 'pW/K')
        print('G_W(400 nm) = ', round(GmeasW, 2), ' +/- ', round(sigGW, 2), 'pW/K')
        print('G_I(400 nm) = ', round(GmeasI, 2), ' +/- ', round(sigGI, 2), 'pW/K')
        print('alpha_U = ', round(alphaU, 2), ' +/- ', round(sigalphaU, 2))
        print('alpha_W = ', round(alphaW, 2), ' +/- ', round(sigalphaW, 2))
        print('alpha_I = ', round(alphaI, 2), ' +/- ', round(sigalphaI, 2))
        print('')
        kappaU = GtoKappa(GmeasU, A_U, L); sigkappaU = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
        kappaW = GtoKappa(GmeasW, A_W, L); sigkappaW = GtoKappa(sigGW, A_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
        kappaI = GtoKappa(GmeasI, A_I, L); sigkappaI = GtoKappa(sigGI, A_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
        print('Kappa_U: ', round(kappaU, 2), ' +/- ', round(sigkappaU, 2), ' pW/K/um')
        print('Kappa_W: ', round(kappaW, 2), ' +/- ', round(sigkappaW, 2), ' pW/K/um')
        print('Kappa_I: ', round(kappaI, 2), ' +/- ', round(sigkappaI, 2), ' pW/K/um')
        print('G_wire = ', round(Gwire, 2), ' +/- ', round(sigma_Gwire, 2), 'pW/K')

        chisq_fit = chisq_val(fit_params, data)
        print('Chi-squared value for the fit: ', round(chisq_fit, 3)) 

    return fit_params, fit_errs, [kappaU, kappaW, kappaI], [sigkappaU, sigkappaW, sigkappaI], Gwire, sigma_Gwire, chisq_fit

def pairwise(sim_data, labels, title='', plot_dir='./', fn_comments='', save_figs=False, indstp=[], indsop=[], oplotlabel=''):
    # make pairwise correlation plots with histograms on the diagonal 
    # sim_data needs to be transposed so that it's 6 x number of iterations
    # indstp = index of solutions to plot, default is all
    # indsop = index of subset of solutions to overplot on all solutions

    if len(indstp)==0: indstp = np.arange(np.shape(sim_data)[1])   # allow for plotting subsections of simulation data 
    nsolns = len(indsop) if len(indsop)!=0 else len(indstp)   # count number of solutions, if overplotting count number of overplotted solutions
    ndim = len(sim_data)   # number of dimensions 

    limpad = np.array([max(sim_data[pp][indstp])-min(sim_data[pp][indstp]) for pp in np.arange(len(sim_data))])*0.10   # axis padding for each parameter
    limits = np.array([[min(sim_data[pp][indstp])-limpad[pp], max(sim_data[pp][indstp])+limpad[pp]] for pp in np.arange(len(sim_data))])   # axis limits for each parameter
    histlim = [1,1E4]

    pairfig = plt.figure()
    for ii in np.arange(ndim):   # row
        for jj in range(ndim):   # column
            spind = ii*ndim+jj+1   # subplot index, starts at 1
            ax = pairfig.add_subplot(ndim, ndim, spind)
            if ii == jj:   # histograms on the diagonal
                hist, bins, patches = ax.hist(sim_data[ii][indstp], bins=30, color='C1', label='All')
                ax.hist(sim_data[ii][indsop], bins=bins, color='C2', histtype='stepfilled', alpha=0.5, label=oplotlabel)  # highlight subset of solutions
                ax.set_yscale('log')
                ax.yaxis.set_ticks([1E1, 1E2, 1E3, 1E4])
                ax.set_xlim(limits[ii]); ax.set_ylim(histlim)
                ax.set_xlabel(labels[jj]); ax.set_ylabel(labels[ii])
            else:           
            # elif jj<=ii:   # scatter plots on off-diagonal 
                ax.scatter(sim_data[jj][indstp], sim_data[ii][indstp], marker='.', alpha=0.3)   # row shares the y axis, column shares the x axis
                ax.scatter(sim_data[jj][indsop], sim_data[ii][indsop], marker='.', alpha=0.3, color='C2')   # highlight subset of solutions
                ax.set_xlim(limits[jj]); ax.set_ylim(limits[ii])
                ax.set_xlabel(labels[jj]); ax.set_ylabel(labels[ii])
        
    axes = pairfig.get_axes()
    for ax in axes:   # only label bottom and left side
        ax.label_outer()
    if len(indsop)!=0: 
        # pairfig.legend()
        ax = axes[0]
        handles, labels = ax.get_legend_handles_labels()
        # handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc=(-0.5, 0.3))
        # ax.legend()
    plt.suptitle(title+'\\textbf{ (N='+str(nsolns)+')}', fontsize=20, y=0.93)

    if save_figs: plt.savefig(plot_dir + 'pairwiseplots' + fn_comments + '.png', dpi=300)   # save figure

    return pairfig

def plot_modelvdata(fit, data, title=''):
    # plot bolotest data vs model fit
    # fit = [params, sigma_params]
    # data = [Gbolos, sigma_Gbolos]

    # bolotest geometry
    L = 220   # bolotest leg length, um
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
    AoL_bolo = A_bolo/L   # A/L for bolotest devices

    # predictions from model
    Gpred, sigmaGpred = Gbolotest(fit)   # predictions and error from model [pW/K]
    Gpred_wire, sigmaGpred_wire = Gbolotest(fit, layer='wiring')   # predictions and error from model [pW/K]
    Gpred_U, sigmaGpred_U = Gbolotest(fit, layer='U')   # predictions and error from model [pW/K]

    plt.figure()
    plt.errorbar(AoL_bolo, data[0], yerr=data[1], marker='o', markersize=5, color='g', capsize=2, linestyle='None')
    plt.errorbar(AoL_bolo, Gpred, yerr=sigmaGpred, color='k', marker='*', label=r"G$_\text{TES}$", capsize=2, linestyle='None')
    plt.errorbar(AoL_bolo, Gpred_wire, yerr=sigmaGpred_wire, color='mediumpurple', marker='x', label=r"G$_\text{micro}$", linestyle='None')
    plt.errorbar(AoL_bolo, Gpred_U, yerr=sigmaGpred_U, markersize=5, color='blue', marker='+', label=r"G$_\text{sub}$", linestyle='None')
    plt.ylabel('G(170mK) [pW/K]')
    plt.title(title)
    plt.legend()
    plt.xlabel('Leg A/L [$\mu$m]')
    plt.ylim(0.70, 18.4)

    return 

def Gfrommodel(fit, dsub, lw, ll, layer='total', fab='legacy'):   # model params, thickness of substrate, leg width, and leg length in um
    # predicts G_TES and error from our model and arbitrary bolo geometry, assumes microstrip on all four legs
    # thickness of wiring layers is independent of geometry
    # RETURNS [G prediction, prediction error]

    arrayconv = 1 if np.isscalar(lw) else np.ones(len(lw))   # convert geometry terms to arrays if number of devices > 1

    if fab=='legacy': 
        dW1 = .190*arrayconv; dI1 = .350*arrayconv; dW2 = .400*arrayconv; dI2 = .400*arrayconv   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='legacy')
    elif fab=='bolotest': 
        dW1 = .160*arrayconv; dI1 = .350*arrayconv; dW2 = .340*arrayconv; dI2 = .400*arrayconv   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='bolotest', layer=layer)
    else: print('Invalid fab type, choose "legacy" or "bolotest."')
    dW = .200**arrayconv

    GU = G_layer(fit, dsub, layer='U')*lw/7 *220/ll   # G prediction and error on substrate layer for one leg
    GW = (G_layer(fit, dW1, layer='W')*w1w/5 + G_layer(fit, dW2, layer='W')*w2w/5) *220/ll  # G prediction and error from Nb layers for one leg
    GW1 = G_layer(fit, dW, layer='W')*w1w/5 *220/ll  # G prediction and error from 200 nm Nb layer one leg
    GI = (G_layer(fit, dI1, layer='I') + G_layer(fit, dI2, layer='I')) *lw/7 *220/ll   # G prediction and error on insulating layers for one leg
    Gwire = GW + GI # G error for microstrip on one leg, summing error works because error is never negative here

    if layer=='total': return 4*(GU+Gwire)   # value and error, microstrip + substrate on four legs
    elif layer=='wiring': return 4*(Gwire)   # value and error, microstrip (W1+I1+W2+I2) on four legs
    elif layer=='U': return 4*(GU)   # value and error, substrate on four legs
    elif layer=='W': return 4*(GW)   # value and error, W1+W2 on four legs
    elif layer=='W1': return 4*(GW1)   # value and error, W1 on four legs
    elif layer=='I': return 4*(GI)   # value and error, I1+I2 on four legs
    else: print('Invalid layer type.'); return

def predict_Glegacy(fit, data1b=[], save_figs=False, estimator='model', kappas=np.array([[54.6, 67.1, 100.9], [6.7, 21.3, 4.7]]), title='', plot_comments='', fs=(7,6), plot_dir='/Users/angi/NIS/Bolotest_Analysis/plots/layer_extraction_analysis/'):
    # predicts G for legacy TES data using alpha model, then plots prediction vs measurements (scaled to 170 mK)
    # legacy geometry and measurements are from Shannon's spreadsheet, then plots 

    dW1 = .190; dI1 = .350; dW2 = .400; dI2 = .400   # general film thicknesses (use for legacy), um
    legacyGs_all = np.array([1296.659705, 276.1, 229.3, 88.3, 44, 76.5, 22.6, 644, 676, 550, 125, 103, 583, 603, 498, 328, 84, 77, 19, 12.2, 10.5, 11.7, 13.1, 9.98, 16.4, 8.766, 9.18, 8.29, 9.57, 7.14, 81.73229733, 103.2593154, 106.535245, 96.57474779, 90.04141806, 108.616653, 116.2369491, 136.2558345, 128.6066776, 180.7454359, 172.273248, 172.4456603, 192.5852409, 12.8, 623, 600, 620, 547, 636, 600.3, 645, 568, 538.7, 491.3, 623, 541.2, 661.4, 563.3, 377.3, 597.4, 395.3, 415.3, 575, 544.8, 237.8, 331.3, 193.25, 331.8, 335.613, 512.562, 513.889, 316.88, 319.756, 484.476, 478.2, 118.818, 117.644, 210.535, 136.383, 130.912, 229.002, 236.02, 101.9, 129.387, 230.783, 230.917, 130.829, 127.191, 232.006, 231.056])  
    legacy_ll = np.array([250, 61, 61, 219.8, 500, 500, 1000, 50, 50, 50, 100, 300, 50, 50, 50, 100, 100, 300, 500, 1000, 1000, 1000, 1000, 1250, 500, 1000, 1000, 1000, 1000, 1250, 640, 510, 510, 510, 510, 510, 730, 610, 490, 370, 370, 370, 300, 500, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    legacy_lw = np.array([25, 14.4, 12.1, 10, 10, 15, 10, 41.5, 41.5, 34.5, 13, 16.5, 41.5, 41.5, 34.5, 29, 13, 16.5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 7, 7, 7, 7, 7, 10, 10, 8, 8, 8, 8, 8, 6, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 34.5, 34.5, 41.5, 41.5, 34.5, 34.5, 23.6, 37.5, 23.6, 23.6, 37.5, 37.5, 13.5, 21.6, 13.5, 21.6, 23.6, 37.5, 37.5, 23.6, 23.6, 37.5, 37.5, 11.3, 11.3, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5])
    legacy_dsub = 0.450 + np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    legacy_Tcs = np.array([557, 178.9, 178.5, 173.4, 170.5, 172.9, 164.7, 163, 162, 163, 164, 164, 168, 168, 167, 167, 165, 166, 156, 158, 146, 146, 149, 144, 155, 158, 146, 141, 147, 141, 485.4587986, 481.037173, 484.9293596, 478.3771521, 475.3010335, 483.4209782, 484.0258522, 477.436482, 483.5417917, 485.8804622, 479.8911157, 487.785816, 481.0323883, 262, 193, 188, 188.8, 188.2, 190.2, 188.1, 186.5, 184.5, 187.5, 185.5, 185.8, 185.6, 185.7, 183.3, 167.3, 167, 172.9, 172.8, 166.61, 162.33, 172.87, 161.65, 163.06, 166.44, 177.920926, 178.955154, 178.839062, 177.514658, 177.126927, 178.196297, 177.53632, 169.704602, 169.641018, 173.026393, 177.895192, 177.966456, 178.934122, 180.143125, 177.16833, 178.328865, 179.420334, 179.696264, 172.724501, 172.479515, 177.385267, 177.492689])*1E-3
    legacy_ns = np.array([3.5, 3.4, 3.4, 3, 2.8, 3, 2.7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.707252717, 2.742876666, 2.741499631, 2.783995279, 2.75259088, 2.796872814, 2.747211811, 2.782265754, 2.804876038, 2.879595447, 2.871133545, 2.889243695, 2.870571891, 2.6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    legacyGs170_all = scale_G(.170, legacyGs_all, legacy_Tcs, legacy_ns)   # scale G's to Tc of 170 mK
    legacy_w1w, legacy_w2w = wlw(legacy_lw, fab='legacy')   # calculate W layer widths for legacy data
    legacy_A = 4*(legacy_dsub + dI1 + dI2)*legacy_lw + 4*dW1*legacy_w1w + 4*dW2*legacy_w2w   # um^2, area for four legs, thickness is substrate + wiring stack
    legacyAoLs_all = legacy_A/legacy_ll  # um

    lTcinds = np.where(legacy_Tcs<0.200)[0]   # Tc < 200 mK bolometers from Shannon's data, ignore higher Tc data

    # only look at low Tc data
    dsub = legacy_dsub[lTcinds]; lw = legacy_lw[lTcinds]; ll = legacy_ll[lTcinds]
    legacy_AoLs = legacyAoLs_all[lTcinds]; legacy_Gs = legacyGs170_all[lTcinds] 

    ### predictions 
    if estimator=='model':   # make predictions using our model
        # fit_params, sigma_params = fit
        Gpred, sigma_Gpred = Gfrommodel(fit, dsub, lw, ll)   # predictions and error from model [pW/K]
        GpredW, sigma_GpredW = Gfrommodel(fit, dsub, lw, ll, layer='wiring')
        GpredU, sigma_GpredU = Gfrommodel(fit, dsub, lw, ll, layer='U')
    elif estimator=='kappa':   # make predictions using measured kappas
        kappa, sigma_kappa = kappas
        Gpred = Gfromkappas(kappa, dsub, lw, ll); sigma_Gpred = Gfromkappas(sigma_kappa, dsub, lw, ll)   # predictions and error from model [pW/K]
        GpredW = Gfromkappas(kappa, dsub, lw, ll, layer='wiring'); GpredU = Gfromkappas(kappa, dsub, lw, ll, layer='U')   # break down into wiring and substrate contributions
    else: print('Invalid estimator - choose "model" or "kappa"')
    normres = (legacy_Gs - Gpred)/legacy_Gs*100   # normalized residuals [% data]
    resylim = -600 if 'a01' in plot_comments else -100   # different lower limits on residuals depending on model

    plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])   # model vs data
    plt.scatter(legacy_AoLs, legacy_Gs, color='g', alpha=.7, label=r"Data")
    plt.errorbar(legacy_AoLs, Gpred, yerr=sigma_Gpred, color='k', marker='*', label=r"G$_\text{TES}$", capsize=2, linestyle='None')
    plt.errorbar(legacy_AoLs, GpredU, yerr=sigma_GpredU, color='blue', marker='+', label=r"G$_\text{sub}$", linestyle='None')
    plt.errorbar(legacy_AoLs, GpredW, yerr=sigma_GpredW, color='mediumpurple', marker='x', label=r"G$_\text{micro}$", linestyle='None')      
    plt.ylabel('G(170mK) [pW/K]')
    plt.title(title)
    plt.tick_params(axis="y", which="both", right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.ylim(1,1E3)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off

    if len(data1b)==2:   # plot bolotest 1b data and prediction
        lorder = [0, 3, 1, 2, 4]   # legend label order
        plt.errorbar(AoL_bolo[0], data1b[0], yerr=data1b[1], marker='o', markersize=5, color='purple', label='Bolo 1', capsize=2, linestyle='None')
        if estimator=='model':
            Gpred1b_U = Gfrommodel(fit, .420, 7, 220, layer='U', fab='bolotest')
            Gpred1b_wire = Gfrommodel(fit, .420, 7, 220, layer='wiring', fab='bolotest')
        elif estimator=='kappa':
            Gpred1b_U = Gfromkappas(kappa, .420, 7, 220, layer='U', fab='bolotest')
            Gpred1b_wire = Gfromkappas(kappa, .420, 7, 220, layer='wiring', fab='bolotest')  
        if len(Gpred1b_U)==2:   # includes errors on prediction
            plt.errorbar(AoL_bolo[0], Gpred1b_wire[0]+Gpred1b_U[0], yerr=Gpred1b_wire[1]+Gpred1b_U[1], marker='*', color='purple')
            plt.errorbar(AoL_bolo[0], Gpred1b_U[0], yerr=Gpred1b_U[1], marker='+', color='purple')
            plt.errorbar(AoL_bolo[0], Gpred1b_wire[0], yerr=Gpred1b_wire[1], marker='x', markersize=5, color='purple')
        else:                   
            plt.scatter(AoL_bolo[0], Gpred1b_wire+Gpred1b_U, marker='*', color='purple')
            plt.scatter(AoL_bolo[0], Gpred1b_U, marker='+', color='purple')
            plt.scatter(AoL_bolo[0], Gpred1b_wire, marker='x', s=20, color='purple')
    else:
        lorder = [0, 3, 1, 2]   # legend label order

    # plt.gca().yaxis.set_ticks([1E1, 1E2, 1E3])
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder], loc=2)   # 2 is upper left, 4 is lower right

    ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
    plt.axhline(0, color='k', ls='--')
    plt.scatter(legacy_AoLs, normres, color='r')
    if len(data1b)==3: plt.scatter(data1b[0], (data1b[1]-Gpred1b_wire[0]-Gpred1b_U[0])/data1b[1]*100, color='purple')
    plt.ylabel("Res'ls [$\%$G]")
    plt.xlabel('Leg A/L [$\mu$m]')
    plt.ylim(resylim,100)
    plt.tick_params(axis="y", which="both", right=True)
    plt.gca().yaxis.set_ticks([-100, -50, 50, 100])
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
    if save_figs: plt.savefig(plot_dir + 'Gpredfrom' + estimator + plot_comments + '.png', dpi=300) 

    return Gpred, sigma_Gpred, normres


### analyze results and compare with literature values
def phonon_wlength(vs, T, domcoeff=2.82):   # returns dominant phonon wavelength in vel units * s (probably um)
    # dominant coefficient takes different values in the literature
    # 2.82 is from Ziman (Bourgeois et al J. Appl. Phys 101, 016104 (2007)), Pohl uses domcoeff = 4.25
    
    return planck*vs/(domcoeff*kB*T)   # [v_s * s]

def Cv(T, TD, Tc, volmol, gamma=7.8E-3, carrier=''):   # volumetric heat capacity for superconducting Nb, pJ/K/um^3 = J/K/cm^3
    # INPUT: bath temp, Debye temperature, critical temperature, gamma, molar volume in 1/um^3
    # calculates specific heat for bulk supercondcuting Nb or SiN
    # electron gamma is for Nb [J/mol/K^2] (gamma doesn't matter for SiN), LEUPOLD & BOORSE 1964
    a = 8.21; b=1.52

    if carrier=='electron': C_v = (gamma*Tc*a*np.exp(-b*Tc/T))/volmol  # electron specific heat, electron from LEUPOLD & BOORSE 1964, pJ/K/um^3 (=J/K/cm^3)
    elif carrier=='phonon': C_v = ((12*np.pi**4*NA*kB)/5 * (T/TD)**3)/volmol  # phonon specific heat from low temp limit of Debye model, pJ/K/um^3 (=J/K/cm^3)
    else: print("Invalid carrier, options are 'phonon' or 'electron'")
    return C_v
    
def kappa_permfp(T, material=''):   # Leopold and Boorse 1964, Nb
    # calculates theoretical thermal conductivity via specific heat for bulk SC Nb and SiN  
    # INPUT: bath temp, Debye temperature, critical temperature, carrier velocity (Fermi vel for electrons, sound speed for phonons) in um/s
    # RETURNS: thermal conductivity per mean free path in pW / K / um^2

    if material=='Nb':
        TD_Nb = 275   # K, Nb Debye temperature
        Tc_Nb = 9.2   # Nb, K
        vF_Nb = 1.37E6   # Nb Fermi velocity (electron velocity), m/s
        vs_Nb = 3480   # phonon velocity is the speed of sound in Nb, m/s
        volmol_Nb = 10.84   # cm^3 per mole for Nb = 1E12 um^3
        Cv_ph = Cv(T, TD_Nb, Tc_Nb, volmol_Nb, carrier='phonon')  # vol heat capacity of carrier, electron from LEUPOLD & BOORSE 1964, J/K/um^3
        Cv_el = Cv(T, TD_Nb, Tc_Nb, volmol_Nb, carrier='electron')  # vol heat capacity of carrier, electron from LEUPOLD & BOORSE 1964, J/K/um^3
        kappapmfp = 1/3*(Cv_ph*vs_Nb + Cv_el*vF_Nb)*1E6 # thermal conductivity via phenomenological gas kinetic theory / mfp, pW / K / um^2

    elif material=='SiN':
        vs_SiN = 6986   # m/s; Wang et al
        TD_Si = 645   # K, silicon, low temp limit
        volmol_SiN = 40.78   # cm^3 per mole for SiN = 1E12 um^3
        Cv_ph = Cv(T, TD_Si, np.nan, volmol_SiN, carrier='phonon')  # vol heat capacity of phonons from Debye model, J/K/um^3
        kappapmfp = 1/3*(Cv_ph*vs_SiN)*1E6 # thermal conductivity via phenomenological gas kinetic theory / mfp, pW / K / um^2

    else: 
        print('Invalid material. Options are Nb and SiN')
        return

    return kappapmfp   

def I_mfp(x):
    return x/2 * np.arcsinh(x) + 1/6*((1+x**2)**(1/2) * (x**2-2) + (2-x**3))

def sumfunc_J(n, f, J):   # function to J sum over
    if J==0:   # handle divide by 0 error
        return f*(1-f)**J * ( n**3 * ((J+1)**3 * I_mfp(1/(n*(J+1))) - J**3 * 0 ) + 1/2*(2-1) * (I_mfp(n*(J+1)) - 2*I_mfp(n*J) + I_mfp(n*(J-1))) )
        # return f*(1-f)**J * ( n**3 * ((J+1)**3 * I_mfp(1/(n*(J+1))) - J**3 * 0 ) + 1/2*(2-1) * (I_mfp(n*(J+1)) - 2*I_mfp(n*J) ) )
        # return f*(1-f)**J * ( n**3 * ((J+1)**3 * I_mfp(1/(n*(J+1))) - 0) + 0 )
    elif J>0:
        return f*(1-f)**J * ( n**3 * ((J+1)**3 * I_mfp(1/(n*(J+1))) - J**3 * I_mfp(1/(n*J))) + 1/2*(2-0) * (I_mfp(n*(J+1)) - 2*I_mfp(n*J) + I_mfp(n*(J-1))) )
    else:
        print('J cannot be <0. Returning NaN')
        return np.nan

def l_eff(w, d, f, sumlim=1E3):   # boundary-limited phonon mfp including spectral scattering from Wybourne84
    # inputs are leg width and thickness in the same units (probably um); w>d
    # f is fraction of diffusive reflection (1-f is fraction of spectral reflection)
    # J is the number of times a phonon impinges on the surface before being diffusively scattered
    n = w/d   # aspect ratio
    # print(n)
    Jfunc_vals = [sumfunc_J(n, f, J) for J in np.arange(sumlim)]   # should be an infinite sum but converges pretty quickly
    mfp = 3*d/(2*n) * np.sum(Jfunc_vals, axis=0)  # same units as d, probably um
    return mfp

def l_bl(w, d):   # boundary-limited phonon mfp from Wybourne84
    # inputs are leg width and thickness in the same units (probably um); w>d
    n = w/d   # aspect ratio
    mfp = 3*d/(2*n) * (n**3 * I_mfp(1/n)+I_mfp(n))   # same units as w and d, probably um
    return mfp

def l_Casimir(w, d):   # mfp in Casimir limit assuming d ~ w (not true for our devices)
    return 2*np.sqrt(w*d)/np.sqrt(np.pi)

def G_Holmes(A, T, xi=1):
    # G in crystalline dielectric thin films (d=um-mm) in Holmes98
    # xi gives reduction in G from diffuse scattering, = 1 in purely spectral scattering
    # x-sectional area A should be um^2
    sigmaSF = 15.7*10  # pW/K^4/um^2
    return 4*A*sigmaSF*T**3*xi   # pW/K

def l_PLT02(vs):   # mfp at 170 mK for amorphous solids from dominant phonon wavelength ratio
    # inverse Q is internal friction
    invQ = 3E-4   # at 170 mK
    return phonon_wlength(vs, 0.170)/(2*np.pi*invQ)

def TFNEP(T, G):   # calculate thermal fluctuation noise equivalent power as a function of G and T
    return np.sqrt(4*kB*G)*T

def sigma_NEP(T, G, sigma_G):   # error on NEP estimation
    sigma_nepsq = kB/G*T**2 * sigma_G**2
    return np.sqrt(sigma_nepsq)

def GandPsatfromNEP(NEP, Tc, Tb, gamma=1):   # calculate G(Tc) and Psat(Tc) given thermal fluctuation NEP, Tc in K, and Tbath in K
    G_Tc = (NEP/Tc)**2 / (4*kB*gamma)   # W/K
    P_Tc = G_Tc*(Tc-Tb)   # W
    return np.array([G_Tc, P_Tc])