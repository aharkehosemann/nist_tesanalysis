import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle as pkl

kB = 1.3806503E-23   # Boltzmann constant, [J/K]
NA = 6.022E23   # Avogadro's number, number of particles in one mole
hbar = 1.055E-34   # J s
bolos=np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])   # this is just always true
mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

### Supporting Functions
# def G_layer(Gnorm, alpha, coeff, ratio):   # G of a single layer, to be summed over all three layers for one bolometer
#     return Gnorm * np.nansum(coeff * ratio**(1+alpha))

# def G_bolo(geom_terms, U, W, I, aU, aW, aI):   # G_total of a bolometer, sum of G_layer's
#     coeffs, ratios = geom_terms
#     return sum([G_layer(U, aU, coeffs[0], ratios[0]), G_layer(W, aW, coeffs[1], ratios[1]), G_layer(I, aI, coeffs[2], ratios[2])])

# def chi_sq(ydata, ymodel, sigma):   # UPDATE: this is weighted least squares, not the chi-squared statistic
#     return np.nansum([((ydata[ii] - ymodel[ii])**2/ sigma[ii]**2) for ii in np.arange(len(ydata))])

def GtoKappa(G, A, L):   # thermal conductance, area, length
    return G*L/A

def KappatoG(kappa, A, L):   # thermal conductance in pW/K/um, area in um^2, length in um
    return kappa*A/L

def G_legacy(params, legw, dsub):  # predict G of legacy geometries using our model, all four legs have full wire stacks
    Gwire = params[1]*(5/5*(340/400)**(params[4]+1) + 8/5*(160/400)**(params[4]+1)) + legw/7*params[2]*((350/400)**(params[5]+1) + 1)
    Gsub = legw/7*params[0]*(dsub/420)**(params[3]+1)
    return 4*Gsub + 4*Gwire

def Gfromkappas(dsub, lw, ll, layer='total'):   # thickness of substrate, leg width, and leg length in um
    # predicts G from kappa measurements and bolo geometry
    # thickness of wiring layers is independent of geometry
    # assumes all four legs have a wiring stack
    # kappaW_meas = 42.22; kappaU_meas = 84.13; kappaI_meas = 107.39   # pW/K/um
    kappaU_meas = 75.08; kappaW_meas = 66.31; kappaI_meas = 100.00   # pW/K/um
    dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
    w1w=8*np.ones_like(lw); w2w=5*np.ones_like(lw)
    if hasattr(lw, "__len__"):
        smallw = np.where(lw<8)[0]   # accommodate leg widths smaller than default W1 width
        w1w[smallw]=lw[smallw]-2; w2w[smallw]=lw[smallw]-4   # 1 um of clearance on each side
    elif lw<8:
        w1w=lw-2; w2w=lw-4

    GU_pred = KappatoG(kappaU_meas, lw*dsub, ll)   # predicted G from substrate layer
    GW_pred = KappatoG(kappaW_meas, w1w*dW1, ll) + KappatoG(kappaW_meas, w2w*dW2, ll)   # predicted G from Nb layers
    GI_pred = KappatoG(kappaI_meas, lw*dI1, ll) + KappatoG(kappaI_meas, lw*dI2, ll)   # predicted G from nitride layers
    if layer=='total': return 4*(GU_pred+GW_pred+GI_pred)
    elif layer=='wiring': return 4*(GW_pred+GI_pred)
    elif layer=='U': return 4*(GU_pred)
    else: print('Invalid layer type.'); return

def Gfrommodel(params, dsub, lw, ll, layer='total'):   # model params, thickness of substrate, leg width, and leg length in um
    # predicts G_TES from our model and bolo geometry
    # thickness of wiring layers is independent of geometry
    # assumes all four legs have a wiring stack

    dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
    w1w=8*np.ones_like(lw); w2w=5*np.ones_like(lw)
    if hasattr(lw, "__len__"):
        smallw = np.where(lw<=8)[0]   # accommodate leg widths smaller than default W1 width
        w1w[smallw]=5; w2w[smallw]=3   # 1 um of clearance on each side
    elif lw<=8:
        w1w=5; w2w=3

    GU_pred = params[0]*(dsub/.420)**(params[3]+1)*lw/7*220/ll   # predicted G from substrate layer for one leg
    GW_pred = params[1]*(w1w/5*(dW1/.400)**(params[4]+1) + w2w/3*(dW2/.400)**(params[4]+1))*220/ll   # predicted G from substrate layer for one leg
    GI_pred = params[2]*((dI1/.400)**(params[5]+1) + (dI2/.400)**(params[5]+1))*lw/7*220/ll   # predicted G from insulating layers for one leg
    Gwire_pred = GW_pred + GI_pred

    if layer=='total': return 4*(GU_pred+Gwire_pred)   # one leg
    elif layer=='wiring': return 4*(Gwire_pred)   # one leg
    elif layer=='U': return 4*(GU_pred)   # one leg
    elif layer=='W': return 4*(GW_pred)   # one leg
    elif layer=='I': return 4*(GI_pred)   # one leg
    else: print('Invalid layer type.'); return

def G_wirestack(params):   # calculate G of full wiring stack for one leg of bolotest data, params = GU, GW, GI, alpha_U, alpha_W, alpha_I
    return params[1]*(3/5*(340/400)**(params[4]+1) + (160/400)**(params[4]+1)) + params[2]*((350/400)**(params[5]+1) + 1)

def scale_G(T, GTc, Tc, n):
    return GTc * T**(n-1)/Tc**(n-1)

def sigma_GscaledT(T, GTc, Tc, n, sigma_GTc, sigma_Tc, sigma_n):
    Gterm = sigma_GTc * T**(n-1)/(Tc**(n-1))
    Tcterm = sigma_Tc * GTc * (1-n) * T**(n-1)/(Tc**(n-1))   # this is very tiny
    nterm = sigma_n * GTc * T**(n-1)/(Tc**(n-1)) * (np.log(T)-np.log(Tc))
    return np.sqrt( Gterm**2 + Tcterm**2 + nterm**2)   # quadratic sum of sigma G(Tc), sigma Tc, and sigma_n terms

def calc_chisq(obs, expect):
    return np.sum((obs-expect)**2/expect)

# def calc_chisq(params, data):   # wrapper for chi-squared min
#     U, W, I, aU, aW, aI = params
#     ydata, sigma = data
#     coeffs = xdata[0]; ratios=xdata[1]
#     ymodel = [G_bolo([coeffs[ii], coeffs[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
#     return chi_sq(ydata, ymodel, sigma)

def Gbolos(params):   
    GU, GW, GI, aU, aW, aI = params
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    
    bolo1b = 4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4)
    bolo24 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5) + GI*((350/400)**(1 + aI) + 1)
    bolo23 = GU*(2 + 2*((220+120)/420)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2)
    bolo22 = GU*(3 + ((220+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3)
    bolo21 = GU*(1 + 3*((280+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI))  
    bolo20 = 4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI)  
    bolo7 = GU*(3 + ((280+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) 
    bolo13 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) 
    
    return np.array([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13])


def Gbolos_six(params):   
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


# def Gbolos_singleI(params):   
#     GU, GW, GI, aU, aW, aI = params
#     wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    
#     bolo1b = 4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4)
#     bolo24 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5) + GI*((350/400)**(1 + aI) + 1)
#     bolo23 = GU*(2 + 2*((220+120)/420)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2)
#     bolo22 = GU*(3 + ((220+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3)
#     bolo21 = GU*(1 + 3*((280+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*((670+350)/400)**(1 + aI))  
#     bolo20 = 4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI)  
#     bolo7 = GU*(3 + ((280+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) 
#     bolo13 = GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) 
    
#     return np.array([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13])


def WLS_val(params, data, model='default'):   # calculates error-weighted least squares
    ydata, sigma = data
    if model=='default':
        Gbolos_model = Gbolos(params)   # predicted G of each bolo
    elif model=='six_layers':   # model SiOx as it's own layer
        Gbolos_model = Gbolos_six(params)
    WLS_vals = (Gbolos_model-ydata)**2/sigma**2

    return np.sum(WLS_vals) 

def calc_func_grid(params, data):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = WLS_val(params_rc, data)
    return func_grid


def runsim_WLS(num_its, p0, data, bounds2, plot_dir, show_yplots=False, save_figs=False, fn_comments='', sim_file=None,  save_sim=False, calc_Gwire=False, model='default'):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (probably pW/K)

    print('Running minimization simulation for weighted least-squares minimization.')
    ydata, sigma = data

    pfits_sim = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        sim_result = minimize(WLS_val, p0, args=[y_its[ii], sigma], bounds=bounds2)
        pfits_sim[ii] = sim_result['x']

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

    print('Results from Monte Carlo Sim - WLS Min')
    print('G_U(420 nm) = ', round(U_sim, 2), ' +/- ', round(Uerr_sim, 2), 'pW/K')
    print('G_W(400 nm) = ', round(W_sim, 2), ' +/- ', round(Werr_sim, 2), 'pW/K')
    print('G_I(400 nm) = ', round(I_sim, 2), ' +/- ', round(Ierr_sim, 2), 'pW/K')
    print('alpha_U = ', round(aU_sim, 2), ' +/- ', round(aUerr_sim, 2))
    print('alpha_W = ', round(aW_sim, 2), ' +/- ', round(aWerr_sim, 2))
    print('alpha_I = ', round(aI_sim, 2), ' +/- ', round(aIerr_sim, 2))
    print('')

    if calc_Gwire:
        Gwires = G_wirestack(pfits_sim.T)
        Gwire = np.mean(Gwires); Gwire_std = np.std(Gwires)
        print('G_wirestack = ', round(Gwire, 2), ' +/- ', round(Gwire_std, 2), 'pW/K')

    if save_sim:
        results_dict = {}
        results_dict['sim'] = pfits_sim   # add simulation 
        results_dict['sim_params'] = {}
        results_dict['sim_params']['num_its'] = num_its   
        results_dict['sim_params']['p0'] = p0  
        results_dict['fit'] = {}
        results_dict['fit']['fit_params'] = sim_params   
        results_dict['fit']['fit_std'] = sim_std  
        if calc_Gwire: 
            results_dict['fit']['Gwire'] = Gwire   # pW/K
            results_dict['fit']['Gwire_std'] = Gwire_std   # pW/K
        with open(sim_file, 'wb') as outfile:   # save simulation pkl
            pkl.dump(results_dict, outfile)

    return sim_params, sim_std

def redWLS_val(params, p0, data, param):   # WLS value with reduced number of parameters
    
    GU0, GW0, GI0, aU0, aW0, aI0 = p0

    if param == 'U': params_topass = [params[0], GW0, GI0, params[1], aW0, aI0]
    elif param == 'W': params_topass = [GU0, params[0], GI0, aU0, params[1], aI0]
    elif param == 'I': params_topass = [GU0, GW0, params[0], aU0, aW0, params[1]]

    return WLS_val(params_topass, data)

def runsim_red(num_its, p0, data, bounds, param, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):   ### reduced MC sim of hand-written function minimization, debugging minimization
    # assumes 4 parameters from p0, fits two other parameters

    print('\n'); print('Running Reduced Minimization Simulation'); print('\n')

    ydata, sigma = data
    if param == 'U': p0red = p0[[0,3]]
    elif param == 'W': p0red = p0[[1,4]]
    elif param == 'I': p0red = p0[[2,5]]

    # func_result = minimize(redWLS_val, p0red, args=(p0, data, param), bounds=bounds)
    # if func_result['success']:  
    #     if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
    #         G_func, a_func = func_result['x']
    #     else:
    #         print('Single inimization returned result outside of bounds.')
    #         G_func, a_func = [np.nan, np.nan]
    # else:
    #     print('Reduced Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, 2))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        # func_result = minimize(redWLS_val, p0red, args=(p0, [ydata, sigma], param), bounds=bounds)
        func_result = minimize(redWLS_val, p0red, args=(p0, [y_its[ii], sigma], param), bounds=bounds)
        if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
            pfits_func[ii] = func_result['x']
        else:
            print('Minimization in simulation returned result outside of bounds.')
            pfits_func[ii] = [np.nan, np.nan]
        
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
    func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
    G_MC, a_MC = func_params; Gerr_MC, aerr_MC = func_std    # parameter fits / errors from Monte Carlo Function Minimization

    print('Results from Monte Carlo Sim WLS Min -- 2 parameters')
    print('G', param, ' = ', round(G_MC, 2), ' +/- ', round(Gerr_MC, 2))
    print('alpha_', param, ' = ', round(a_MC, 2), ' +/- ', round(aerr_MC, 2))
    print('')
    return func_params, func_std


def overplot_qp(p0, data, boundsred, n_its, data_J, param, plot_dir, full_res=[], savefigs=False, fn_comments='', vmax=1E3):
    ### plot chi-squared two parameters space, overlayed with results from full fit, two-parameter fit, and joel's results

    if param == 'U' or param=='W' or param=='I':   # two-parameter MC fit, assume other parameters come from full model MC results
        xgridlim=[0,3]; ygridlim=[0,2]   # alpha_layer vs G_layer 
        xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]
        xlab = 'G'+param  
        if param=='U': 
            pinds = [0,3]
            gridparams = np.array([xgrid, full_res[0][1]*np.ones_like(xgrid), full_res[0][2]*np.ones_like(xgrid), ygrid, full_res[0][4]*np.ones_like(ygrid), full_res[0][5]*np.ones_like(ygrid)]).T
            ylab = 'a$_U$'
        elif param=='W': 
            pinds = [1,4]
            gridparams = np.array([full_res[0][0]*np.ones_like(xgrid), xgrid, full_res[0][2]*np.ones_like(xgrid), full_res[0][3]*np.ones_like(ygrid), ygrid, full_res[0][5]*np.ones_like(ygrid)]).T
            ylab = 'a$_W$'
        elif param=='I': 
            pinds = [2,5]
            gridparams = np.array([full_res[0][0]*np.ones_like(xgrid), full_res[0][1]*np.ones_like(xgrid), xgrid, full_res[0][3]*np.ones_like(ygrid), full_res[0][4]*np.ones_like(ygrid), ygrid]).T
            ylab = 'a$_I$'

        # MC_params, MC_std = runsim_red(n_its, p0, data, boundsred, param, plot_dir, fn_comments=fn_comments)   # MC Sim 
        MC_params, MC_std = runsim_red(n_its, full_res[0], data, boundsred, param, plot_dir, fn_comments=fn_comments)   # MC Sim 
        x1_MC, x2_MC = MC_params; x1err_MC, x2err_MC =  MC_std    # parameter fits / errors from Monte Carlo Function Minimization

    else:
        print('Invalid parameter choice. Available choices: U, W, I')

    funcgrid = calc_func_grid(gridparams, data)   # Grid for Quality Plots
    Jres = data_J[0][pinds]; Jerr = data_J[1][pinds]  # Joe's results
        
    plt.figure()   # G vs a parameter space
    im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.6)   # for poster
    # plt.errorbar(x1_MC, x2_MC, xerr=x1err_MC, yerr=x2err_MC, color='forestgreen', label='Two-Param Fit', capsize=1)   # matching data and model colors
    if list(full_res): plt.errorbar(full_res[0][pinds[0]], full_res[0][pinds[1]], xerr=full_res[1][pinds[0]], yerr=full_res[1][pinds[1]], color='black', label='Full Fit', capsize=1)   # matching data and model colors
    # plt.errorbar(Jres[0], Jres[1], xerr=Jerr[0], yerr=Jerr[1], color='darkviolet', label="Joel's Results", capsize=1)   # matching data and model colors
    plt.colorbar(im)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
    plt.title('Five Layer Fit')
    if savefigs: plt.savefig(plot_dir + 'redqualityplot_' + param + fn_comments + '.png', dpi=300)

    return MC_params, MC_std


def TFNEP(T, G):   # calculate thermal fluctuation noise equivalent power as a function of G and T
    return np.sqrt(4*kB*G)*T
  