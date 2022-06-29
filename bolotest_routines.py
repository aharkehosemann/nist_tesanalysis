import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

bolos=np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])   # this is just always true

### Supporting Functions
def G_layer(Gnorm, alpha, coeff, ratio):   # G of a single layer, to be summed over all three layers for one bolometer
    return Gnorm * np.nansum(coeff * ratio**(1+alpha))

def G_bolo(geom_terms, U, W, I, aU, aW, aI):   # G_total of a bolometer, sum of G_layer's
    coeffs, ratios = geom_terms
    return sum([G_layer(U, aU, coeffs[0], ratios[0]), G_layer(W, aW, coeffs[1], ratios[1]), G_layer(I, aI, coeffs[2], ratios[2])])

def chi_sq(ydata, ymodel, sigma):
    return np.nansum([((ydata[ii] - ymodel[ii])**2/ sigma[ii]**2) for ii in np.arange(len(ydata))])

def calc_chisq(params, data):   # wrapper for chi-squared min
    U, W, I, aU, aW, aI = params
    ydata, xdata, sigma = data
    coeffs = xdata[0]; ratios=xdata[1]
    ymodel = [G_bolo([coeffs[ii], coeffs[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
    return chi_sq(ydata, ymodel, sigma)

def func_tomin(params, data):   # error is randomized every time
    GU, GW, GI, aU, aW, aI = params
    ydata, xdata, sigma = data

    ### new W width scaling W2 to be 3/5 W1, U thickness (SiOx + SiNx)
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20 
    bolo1b = (4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4) - ydata[0])**2/sigma[0]**2 
    bolo24 = (GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5*3/5) + GI*((350/400)**(1 + aI) + 1) - ydata[1])**2/sigma[1]**2 
    bolo23 = (GU*(2 + 2*((220+120)/420)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2) - ydata[2])**2/sigma[2]**2
    bolo22 = (GU*(3 + ((220+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3) - ydata[3])**2/sigma[3]**2 
    bolo21 = (GU*(1 + 3*((280+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI)) - ydata[4])**2/sigma[4]**2  
    bolo20 = (4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) - ydata[5])**2/sigma[5]**2  
    bolo7 = (GU*(3 + ((280+120)/420)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) - ydata[6])**2/sigma[6]**2 
    bolo13 = (GU*(1 + 3*((220+120)/420)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) - ydata[7])**2/sigma[7]**2
    
    return np.sum([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13]) 

def functomin_sixlayers(params, data):   # error is randomized every time
    GSiO, GSiN, GW, GI, aSiN, aW, aI = params
    ydata, xdata, sigma = data

    ### new W width scaling W2 to be 3/5 W1, SiOx and SiNx substrate layers are separate. 
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20 
    bolo1b = (4*GSiO + 4*GSiN + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)*3/5) + GI*(4*(350/400)**(1 + aI) + 4) - ydata[0])**2/sigma[0]**2 
    bolo24 = (4*GSiO + GSiN*(1 + 3*((220+120)/300)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)*3/5*3/5) + GI*((350/400)**(1 + aI) + 1) - ydata[1])**2/sigma[1]**2 
    bolo23 = (4*GSiO + GSiN*(2 + 2*((220+120)/300)**(1 + aSiN)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)*3/5) + GI*(2*(350/400)**(1 + aI) + 2) - ydata[2])**2/sigma[2]**2
    bolo22 = (4*GSiO + GSiN*(3 + ((220+120)/300)**(1 + aSiN)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + 3) - ydata[3])**2/sigma[3]**2 
    bolo21 = (4*GSiO + GSiN*(1 + 3*((280+120)/300)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI)) - ydata[4])**2/sigma[4]**2  
    bolo20 = (4*GSiO + 4*GSiN + GW*((160/300)**(1 + aW) + 3*(385/400)**(1 + aW)*(wstack_width/5) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) - ydata[5])**2/sigma[5]**2  
    bolo7 = (4*GSiO + GSiN*(3 + ((280+120)/300)**(1 + aSiN)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)*3/5) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) - ydata[6])**2/sigma[6]**2 
    bolo13 = (4*GSiO + GSiN*(1 + 3*((220+120)/300)**(1 + aSiN)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)*3/5) + GI*(350/400)**(1 + aI) - ydata[7])**2/sigma[7]**2
    
    return np.sum([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo7, bolo13]) 

def calc_chisq_grid(params, data):   # chi-squared parameter space
    ydata, xdata, sigma = data
    coeffs = xdata[0]; ratios=xdata[1]
    chisq_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            U, W, I, aU, aW, aI = col
            ymodel = [G_bolo([coeffs[ii], ratios[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
            chisq_grid[rr, cc] = chi_sq(ydata, ymodel, sigma)
    return chisq_grid

def calc_func_grid(params, data):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = func_tomin(params_rc, data)
    return func_grid

def run_sim(num_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):   ### MC sim of LS & CS minimization

    print('Running minimization simulation over three routines.')

    ### Least Squares Min
    ydata, xdata, sigma = data
    pfit_LS, pcov_LS = curve_fit(G_bolo, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds1)   # non-linear least squares fit
    U_LS, W_LS, I_LS, aU_LS, aW_LS, aI_LS = pfit_LS   # best fit parameters
    perr_LS = np.sqrt(np.diag(pcov_LS)); Uerr_LS = perr_LS[0]; Werr_LS = perr_LS[1]; Ierr_LS = perr_LS[2]; aUerr_LS = perr_LS[3]; aWerr_LS = perr_LS[4]; aIerr_LS = perr_LS[5]   # error of fit

    ### Chi-Squared Min
    chi_result = minimize(calc_chisq, p0, args=[ydata, xdata, sigma], bounds=bounds2)   # fit is unsuccessful and results are nonsense if bounds aren't specified 
    cbounds_met = np.array([bounds2[ii][0]<=chi_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(chi_result['x']))]).all()
    if chi_result['success']:       
        U_CS, W_CS, I_CS, aU_CS, aW_CS, aI_CS = chi_result['x']
    else:
        print('Chi-Squared Min was unsuccessful.')

    ### Hand-Written Function Minimization
    func_result = minimize(func_tomin, p0, args=[ydata, xdata, sigma], bounds=bounds2) 
    if func_result['success']:   
        fresult_temp = func_result['x']
        fbounds_met = np.array([bounds2[ii][0]<=func_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(func_result['x']))]).all()
        if ~fbounds_met:   # check if fit parameters are not within bounds
            print('Some or all fit parameters returned were not within the prescribed bounds. \n Changing these to NaNs. \n')
            fresult_temp[~fbounds_met] = np.nan
        U_func, W_func, I_func, aU_func, aW_func, aI_func = fresult_temp
    else:
        print('Single Function Min was unsuccessful.')

    pfits_LS = np.empty((num_its, len(p0))); pfits_CS = np.empty((num_its, len(p0))); pfits_func = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        # least squares
        y_it = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        # y_it = np.random.uniform(low=0, high=20, size=len(ydata))   # for testing
        y_its[ii] = y_it
        pfit_LS, pcov_LS = curve_fit(G_bolo, xdata, y_it, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds1)
        pfits_LS[ii] = pfit_LS

        # chi-squared
        chi_result = minimize(calc_chisq, p0, args=[y_it, xdata, sigma], bounds=bounds2) 
        if chi_result['success']:   
            pfits_CS[ii] = chi_result['x']
            cbounds_met = np.array([bounds2[ii][0]<=chi_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(chi_result['x']))]).all()
            if ~cbounds_met:   # check if fit parameters are not within bounds
                print('Some or all Chi-Squared fit parameters returned were not within the prescribed bounds. \n Setting to NaNs. \n')
                pfits_CS[ii] = [np.nan]*len(pfits_CS[ii])
        else:
            print('Chi-Squared Min in simulation run was unsuccessful.')

        func_result = minimize(func_tomin, p0, args=[y_it, xdata, sigma], bounds=bounds2)
        if func_result['success']:   
            pfits_func[ii] = func_result['x']
            fbounds_met = np.array([bounds2[ii][0]<=func_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(func_result['x']))]).all()
            if ~fbounds_met:   # check if fit parameters are not within bounds
                print('Some or all function fit parameters returned were not within the prescribed bounds. \n Changing these to NaNs. \n')
                pfits_func[ii] = [np.nan]*len(pfits_func[ii])
        else:
            print('Function Min in simulation run was unsuccessful.')

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            n, bins, patches = plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate('N$_{iterations}$ = %d'%num_its, (min(yit), 0.9*max(n)))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata' + fn_comments + '.png', dpi=300) 
        
    # sort results
    LS_params = np.mean(pfits_LS, axis=0); CS_params = np.mean(pfits_CS, axis=0); func_params = np.mean(pfits_func, axis=0)
    LS_std = np.std(pfits_LS, axis=0); CS_std = np.std(pfits_CS, axis=0); func_std = np.std(pfits_func, axis=0)

    # print results
    U_LSMC, W_LSMC, I_LSMC, aU_LSMC, aW_LSMC, aI_LSMC = LS_params   # parameter fits from Monte Carlo Least Squares Minimization
    Uerr_LSMC, Werr_LSMC, Ierr_LSMC, aUerr_LSMC, aWerr_LSMC, aIerr_LSMC = LS_std   # parameter errors from Monte Carlo Least Squares Minimization
    U_CSMC, W_CSMC, I_CSMC, aU_CSMC, aW_CSMC, aI_CSMC = CS_params   # parameter fits from Monte Carlo Chi-Squared Minimization
    Uerr_CSMC, Werr_CSMC, Ierr_CSMC, aUerr_CSMC, aWerr_CSMC, aIerr_CSMC = CS_std   # parameter errors from Monte Carlo Chi-Squared Minimization
    U_funcMC, W_funcMC, I_funcMC, aU_funcMC, aW_funcMC, aI_funcMC = func_params   # parameter fits from Monte Carlo Function Minimization
    Uerr_funcMC, Werr_funcMC, Ierr_funcMC, aUerr_funcMC, aWerr_funcMC, aIerr_funcMC = func_std   # parameter errors from Monte Carlo Function Minimization

    print('')   # least-squared minimization results
    print('Results from LSM')
    print('G_SiN(420 nm) = ', round(U_LS, 2), ' +/- ', round(Uerr_LS, 2))
    print('G_W(400 nm) = ', round(W_LS, 2), ' +/- ', round(Werr_LS, 2))
    print('G_I(400 nm) = ', round(I_LS, 2), ' +/- ', round(Ierr_LS, 2))
    print('alpha_U = ', round(aU_LS, 2), ' +/- ', round(aUerr_LS, 2))
    print('alpha_W = ', round(aW_LS, 2), ' +/- ', round(aWerr_LS, 2))
    print('alpha_I = ', round(aI_LS, 2), ' +/- ', round(aIerr_LS, 2))
    print('')
    print('Results from Monte Carlo sim - LSM')
    print('G_SiN(420 nm) = ', round(U_LSMC, 2), ' +/- ', round(Uerr_LSMC, 2))
    print('G_W(400 nm) = ', round(W_LSMC, 2), ' +/- ', round(Werr_LSMC, 2))
    print('G_I(400 nm) = ', round(I_LSMC, 2), ' +/- ', round(Ierr_LSMC, 2))
    print('alpha_U = ', round(aU_LSMC, 2), ' +/- ', round(aUerr_LSMC, 2))
    print('alpha_W = ', round(aW_LSMC, 2), ' +/- ', round(aWerr_LSMC, 2))
    print('alpha_I = ', round(aI_LSMC, 2), ' +/- ', round(aIerr_LSMC, 2))
    print('')

    if chi_result['success']:   # chi-squared results
        print('')   
        print('Results from Chi-Squared Min')
        print('G_SiN(420 nm) = ', round(U_CS, 2))
        print('G_W(400 nm) = ', round(W_CS, 2))
        print('G_I(400 nm) = ', round(I_CS, 2))
        print('alpha_U = ', round(aU_CS, 2))
        print('alpha_W = ', round(aW_CS, 2))
        print('alpha_I = ', round(aI_CS, 2))
    else:
        print('Chi-Squared Min (Single Run) was unsuccessful.')
    print('')
    print('Results from Monte Carlo sim - CSM')
    print('G_SiN(420 nm) = ', round(U_CSMC, 2), ' +/- ', round(Uerr_CSMC, 2))
    print('G_W(400 nm) = ', round(W_CSMC, 2), ' +/- ', round(Werr_CSMC, 2))
    print('G_I(400 nm) = ', round(I_CSMC, 2), ' +/- ', round(Ierr_CSMC, 2))
    print('alpha_U = ', round(aU_CSMC, 2), ' +/- ', round(aUerr_CSMC, 2))
    print('alpha_W = ', round(aW_CSMC, 2), ' +/- ', round(aWerr_CSMC, 2))
    print('alpha_I = ', round(aI_CSMC, 2), ' +/- ', round(aIerr_CSMC, 2))
    print('')

    if func_result['success']:       
        print('')   # function results (also chi-squared?)
        print('Results from Hand-Written Function Min')
        print('G_SiN(420 nm) = ', round(U_func, 2))
        print('G_W(400 nm) = ', round(W_func, 2))
        print('G_I(400 nm) = ', round(I_func, 2))
        print('alpha_U = ', round(aU_func, 2))
        print('alpha_W = ', round(aW_func, 2))
        print('alpha_I = ', round(aI_func, 2))
    else:
        print('Function Min (Single Run) was unsuccessful.')
    print('')
    print('Results from Monte Carlo sim - Func Min')
    print('G_SiN(420 nm) = ', round(U_funcMC, 2), ' +/- ', round(Uerr_funcMC, 2))
    print('G_W(400 nm) = ', round(W_funcMC, 2), ' +/- ', round(Werr_funcMC, 2))
    print('G_I(400 nm) = ', round(I_funcMC, 2), ' +/- ', round(Ierr_funcMC, 2))
    print('alpha_U = ', round(aU_funcMC, 2), ' +/- ', round(aUerr_funcMC, 2))
    print('alpha_W = ', round(aW_funcMC, 2), ' +/- ', round(aWerr_funcMC, 2))
    print('alpha_I = ', round(aI_funcMC, 2), ' +/- ', round(aIerr_funcMC, 2))
    print('')

    return LS_params, LS_std, CS_params, CS_std, func_params, func_std

def runsim_func(num_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (probably pW/K)

    print('Running minimization simulation over hand-written chi-squared function.')
    ydata, xdata, sigma = data

    func_result = minimize(func_tomin, p0, args=data, bounds=bounds2) 
    if func_result['success']:       
        U_func, W_func, I_func, aU_func, aW_func, aI_func = func_result['x']
    else:
        print('Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        func_result = minimize(func_tomin, p0, args=[y_its[ii], xdata, sigma], bounds=bounds2)
        pfits_func[ii] = func_result['x']

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            n, bins, patches = plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 0.9*max(n)))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata' + fn_comments + '.png', dpi=300) 
        
    # sory & print results    
    func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
    U_funcMC, W_funcMC, I_funcMC, aU_funcMC, aW_funcMC, aI_funcMC = func_params   # parameter fits from Monte Carlo Function Minimization
    Uerr_funcMC, Werr_funcMC, Ierr_funcMC, aUerr_funcMC, aWerr_funcMC, aIerr_funcMC = func_std   # parameter errors from Monte Carlo Function Minimization

    print('')   # function results 
    print('Results from Function Min')
    print('G_SiN(420 nm) = ', round(U_func, 2))
    print('G_W(400 nm) = ', round(W_func, 2))
    print('G_I(400 nm) = ', round(I_func, 2))
    print('alpha_U = ', round(aU_func, 2))
    print('alpha_W = ', round(aW_func, 2))
    print('alpha_I = ', round(aI_func, 2))
    print('')
    print('Results from Monte Carlo Sim - Function Min')
    print('G_SiN(420 nm) = ', round(U_funcMC, 2), ' +/- ', round(Uerr_funcMC, 2))
    print('G_W(400 nm) = ', round(W_funcMC, 2), ' +/- ', round(Werr_funcMC, 2))
    print('G_I(400 nm) = ', round(I_funcMC, 2), ' +/- ', round(Ierr_funcMC, 2))
    print('alpha_U = ', round(aU_funcMC, 2), ' +/- ', round(aUerr_funcMC, 2))
    print('alpha_W = ', round(aW_funcMC, 2), ' +/- ', round(aWerr_funcMC, 2))
    print('alpha_I = ', round(aI_funcMC, 2), ' +/- ', round(aIerr_funcMC, 2))
    print('')
    return func_params, func_std

def redfunc_tomin(params, p0, data, param):   # reduced function to minimize, debugging minimization
    
    GU0, GW0, GI0, aU0, aW0, aI0 = p0

    if param == 'U': params_topass = [params[0], GW0, GI0, params[1], aW0, aI0]
    elif param == 'W': params_topass = [GU0, params[0], GI0, aU0, params[1], aI0]
    elif param == 'I': params_topass = [GU0, GW0, params[0], aU0, aW0, params[1]]

    return func_tomin(params_topass, data)

def redfunc_six(params, p0, data, param):   # reduced function to minimize, debugging minimization
    
    GSiO0, GSiN0, GW0, GI0, aU0, aW0, aI0 = p0

    if param == 'SiN': params_topass = [GSiO0, params[0], GW0, GI0, params[1], aW0, aI0]
    elif param == 'W': params_topass = [GSiO0, GSiN0, params[0], GI0, aU0, params[1], aI0]
    elif param == 'I': params_topass = [GSiO0, GSiN0, GW0, params[0], aU0, aW0, params[1]]

    return functomin_sixlayers(params_topass, data)

def runsim_red(num_its, p0, data, bounds, param, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):   ### reduced MC sim of hand-written function minimization, debugging minimization
    # assumes 4 parameters from p0, fits two other parameters

    print('\n'); print('Running Reduced Minimization Simulation'); print('\n')

    ydata, xdata, sigma = data
    if param == 'U': p0red = p0[[0,3]]
    elif param == 'W': p0red = p0[[1,4]]
    elif param == 'I': p0red = p0[[2,5]]

    func_result = minimize(redfunc_tomin, p0red, args=(p0, data, param), bounds=bounds)
    if func_result['success']:  
        if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
            G_func, a_func = func_result['x']
        else:
            print('Single inimization returned result outside of bounds.')
            G_func, a_func = [np.nan, np.nan]
    else:
        print('Reduced Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, 2))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        # func_result = minimize(redfunc_tomin, p0red, args=(p0, [ydata, xdata, sigma], param), bounds=bounds)
        func_result = minimize(redfunc_tomin, p0red, args=(p0, [y_its[ii], xdata, sigma], param), bounds=bounds)
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

    print('')   # function results 
    print('Results from Hand-Written Chi-Squared Min')
    print('G', param, ' = ', round(G_func, 2))
    print('alpha', param, ' = ', round(a_func, 2))
    print('')
    print('Results from Monte Carlo sim - H-W func Min')
    print('G', param, ' = ', round(G_MC, 2), ' +/- ', round(Gerr_MC, 2))
    print('alpha_', param, ' = ', round(a_MC, 2), ' +/- ', round(aerr_MC, 2))
    print('')
    return func_params, func_std

def runsimred_six(num_its, p0, data, bounds, param, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):
    # assumes 4 parameters from p0, fits two other parameters

    print('\n'); print('Running Reduced Minimization Simulation'); print('\n')

    ydata, xdata, sigma = data
    if param == 'SiN': p0red = p0[[1,4]]
    elif param == 'W': p0red = p0[[2,5]]
    elif param == 'I': p0red = p0[[3,6]]

    func_result = minimize(redfunc_six, p0red, args=(p0, data, param), bounds=bounds)
    if func_result['success']:  
        if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
            G_func, a_func = func_result['x']
        else:
            print('Single inimization returned result outside of bounds.')
            G_func, a_func = [np.nan, np.nan]
    else:
        print('Reduced Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, 2))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        func_result = minimize(redfunc_six, p0red, args=(p0, [y_its[ii], xdata, sigma], param), bounds=bounds)
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

    print('')   # function results 
    print('Results from Hand-Written Chi-Squared Min')
    print('G', param, ' = ', round(G_func, 2))
    print('alpha', param, ' = ', round(a_func, 2))
    print('')
    print('Results from Monte Carlo sim - H-W func Min')
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
    im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.7)   # for poster
    plt.errorbar(x1_MC, x2_MC, xerr=x1err_MC, yerr=x2err_MC, color='forestgreen', label='Two-Param Fit', capsize=1)   # matching data and model colors
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

def calcfuncgrid_six(params, data):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = functomin_sixlayers(params_rc, data)
    return func_grid

def runsim_sixlayers(num_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=False, save_figs=False, fn_comments=''):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (probably pW/K)

    print('Running minimization simulation over hand-written chi-squared function for six layers.')
    ydata, xdata, sigma = data

    func_result = minimize(functomin_sixlayers, p0, args=data, bounds=bounds2) 
    if func_result['success']:       
        SiO_func, U_func, W_func, I_func, aU_func, aW_func, aI_func = func_result['x']
    else:
        print('Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        func_result = minimize(functomin_sixlayers, p0, args=[y_its[ii], xdata, sigma], bounds=bounds2)
        pfits_func[ii] = func_result['x']

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            n, bins, patches = plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 0.9*max(n)))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata' + fn_comments + '.png', dpi=300) 
        
    # sory & print results    
    func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
    SiO_funcMC, U_funcMC, W_funcMC, I_funcMC, aU_funcMC, aW_funcMC, aI_funcMC = func_params   # parameter fits from Monte Carlo Function Minimization
    SiOerr_funcMC, Uerr_funcMC, Werr_funcMC, Ierr_funcMC, aUerr_funcMC, aWerr_funcMC, aIerr_funcMC = func_std   # parameter errors from Monte Carlo Function Minimization

    print('')   # function results 
    print('Results from Function Min')
    print('G_SiO(120 nm) = ', round(SiO_func, 2))
    print('G_SiN(300 nm) = ', round(U_func, 2))
    print('G_W(400 nm) = ', round(W_func, 2))
    print('G_I(400 nm) = ', round(I_func, 2))
    print('alpha_U = ', round(aU_func, 2))
    print('alpha_W = ', round(aW_func, 2))
    print('alpha_I = ', round(aI_func, 2))
    print('')
    print('Results from Monte Carlo Sim - Function Min')
    print('G_SiO(120 nm) = ', round(SiO_funcMC, 2), ' +/- ', round(SiOerr_funcMC, 2))
    print('G_SiN(300 nm) = ', round(U_funcMC, 2), ' +/- ', round(Uerr_funcMC, 2))
    print('G_W(400 nm) = ', round(W_funcMC, 2), ' +/- ', round(Werr_funcMC, 2))
    print('G_I(400 nm) = ', round(I_funcMC, 2), ' +/- ', round(Ierr_funcMC, 2))
    print('alpha_U = ', round(aU_funcMC, 2), ' +/- ', round(aUerr_funcMC, 2))
    print('alpha_W = ', round(aW_funcMC, 2), ' +/- ', round(aWerr_funcMC, 2))
    print('alpha_I = ', round(aI_funcMC, 2), ' +/- ', round(aIerr_funcMC, 2))
    print('')
    return func_params, func_std

def overplotqp_sixlayers(p0, data, boundsred, n_its, data_compare, param, plot_dir, full_res=[], savefigs=False, fn_comments='', vmax=1E3):
    if param == 'SiN' or param=='W' or param=='I':
        xgridlim=[0,3]; ygridlim=[0,2]   # alpha_layer vs G_layer 
        xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]
        xlab = 'G'+param  
        if param=='SiN': 
            pinds = [1,4]
            gridparams = np.array([full_res[0][0]*np.ones_like(xgrid), xgrid, full_res[0][2]*np.ones_like(xgrid), full_res[0][3]*np.ones_like(xgrid), ygrid, full_res[0][5]*np.ones_like(ygrid), full_res[0][6]*np.ones_like(ygrid)]).T
            ylab = 'a$_SiN$'
        elif param=='W': 
            pinds = [2,5]
            gridparams = np.array([full_res[0][0]*np.ones_like(xgrid), full_res[0][1]*np.ones_like(xgrid), xgrid, full_res[0][3]*np.ones_like(xgrid), full_res[0][4]*np.ones_like(ygrid), ygrid, full_res[0][6]*np.ones_like(ygrid)]).T
            ylab = 'a$_W$'
        elif param=='I': 
            pinds = [3,6]
            gridparams = np.array([full_res[0][0]*np.ones_like(xgrid), full_res[0][1]*np.ones_like(xgrid), full_res[0][2]*np.ones_like(xgrid), xgrid, full_res[0][4]*np.ones_like(ygrid), full_res[0][5]*np.ones_like(ygrid), ygrid]).T
            ylab = 'a$_I$'
        
        # MC_params, MC_std = runsimred_six(n_its, p0, data, boundsred, param, plot_dir, fn_comments=fn_comments)   # MC Sim 
        MC_params, MC_std = runsimred_six(n_its, full_res[0], data, boundsred, param, plot_dir, fn_comments=fn_comments)   # MC Sim 
        x1_MC, x2_MC = MC_params; x1err_MC, x2err_MC =  MC_std    # parameter fits / errors from Monte Carlo Function Minimization

    else:
        print('Invalid parameter choice. Available choices: U, W, I')

    funcgrid = calcfuncgrid_six(gridparams, data)   # Grid for Quality Plots
    res_compare = data_compare[0][pinds]; err_compare = data_compare[1][pinds]  # previous results to compare (probably five-layer results)

    plt.figure()   # G vs a parameter space
    im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.7)
    plt.errorbar(x1_MC, x2_MC, xerr=x1err_MC, yerr=x2err_MC, color='forestgreen', label='Two-Param Fit', capsize=1)   # matching data and model colors
    if list(full_res): plt.errorbar(full_res[0][pinds[0]], full_res[0][pinds[1]], xerr=full_res[1][pinds[0]], yerr=full_res[1][pinds[1]], color='darkviolet', label='Full Fit', capsize=1)   # matching data and model colors
    # plt.errorbar(Jres[0], Jres[1], xerr=Jerr[0], yerr=Jerr[1], color='darkviolet', label="Joel's Results", capsize=1)   # matching data and model colors
    plt.errorbar(res_compare[0], res_compare[1], xerr=err_compare[0], yerr=err_compare[1], color='black', label="Five-Layer Results", capsize=1)   # matching data and model colors
    plt.colorbar(im)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
    plt.title('Six Layer Fit')
    if savefigs: plt.savefig(plot_dir + 'redqualityplot_' + param + fn_comments + '.png', dpi=300)

    return MC_params, MC_std

def GtoKappa(G, A, L):   # thermal conductance, area, length
    return G*L/A
