import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import pickle as pkl

kB = 1.3806503E-23   # Boltzmann constant, [J/K]
NA = 6.022E23   # Avogadro's number, number of particles in one mole
hbar = 1.055E-34   # reduced Plancks constant, [J s]
planck = hbar*2*np.pi   # planck's constant , [J s]
G0 = np.pi**2*kB**2*0.170/(3*planck)*1E12   # an inherent G at 170 mK; pW/K
bolos=np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])   # this is just always true
mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

### Supporting Functions

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


def wlw(lw, fab='bolotest', layer='wiring'):
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
            w1w[naninds] = np.nan; w1w[scaleinds] = w1w0[scaleinds]   # um
            w2w[naninds] = np.nan; w2w[scaleinds] = w1w0[scaleinds]-2   # um
        elif lw-2<minw1w:   # handle single leg widths
            w1w = np.nan; w2w = np.nan
        elif lw-2<maxw1w:
            w1w = lw-2; w2w = lw-4   # um 

    return w1w, w2w

def G_layer(fit, d, layer='U'):
    # fit = [fit parameters, fit errors (if calc_sigma=True)], thickness d is in um
    # RETURNS prediction (and error if 'fit' is 2D)

    d = np.array([d]) if np.isscalar(d) else np.array(d)   # handle thickness scalars and arrays

    if layer=='U': linds=np.array([0,3]); d0=.420   # substrate layer parameter indexes and default thickness in um
    elif layer=='W': linds=np.array([1,4]); d0=.400   # Nb layer parameter indexes and default thickness in um
    elif layer=='I': linds=np.array([2,5]); d0=.400   # insulating layer parameter indexes and default thickness in um

    if len(fit.shape)==2:   # user passed fit parameters and errors
        G0, alpha = fit[0][linds]; sig_G0, sig_alpha = fit[1][linds]   # parameters for layer x
        Glayer = G0 * (d/d0)**(alpha+1) 
        sig_Glayer = sig_G0**2 * (((d/d0)**(alpha+1)))**2 + (G0*(alpha+1)*d**alpha)**2 * sig_alpha**2
        return np.array([Glayer, sig_Glayer]).reshape((2,len(d)))
    elif len(fit.shape)==1:   # user passed only fit parameters 
        G0, alpha = fit[linds]
        Glayer = G0 * (d/d0)**(alpha+1) 
        return np.array([Glayer]).reshape((1,len(d)))
    else: 
        print('Fit array has too many dimensions, either pass 1D array of parameters or 2D array of parameters and param errors.')
        return



def Gfrommodel(fit, dsub, lw, ll, layer='total', fab='legacy'):   # model params, thickness of substrate, leg width, and leg length in um
    # predicts G_TES and error from our model and bolo geometry, assumes microstrip on all four legs
    # thickness of wiring layers is independent of geometry
    # RETURNS [G prediction, prediction error]

    # sf = np.ones_like(lw)   # scale GW if width is < normalized width
    if fab=='legacy': 
        dW1 = .190; dI1 = .350; dW2 = .400; dI2 = .400   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='legacy')
    elif fab=='bolotest': 
        dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='bolotest', layer=layer)
    else: print('Invalid fab type, choose "legacy" or "bolotest."')

    GU = G_layer(fit, dsub, layer='U') *lw/7 *220/ll   # G prediction and error on substrate layer for one leg
    GW = (G_layer(fit, dW1, layer='W')*w1w/5 + G_layer(fit, dW2, layer='W')*w2w/5) *220/ll  # G prediction and error from Nb layers for one leg
    GW1 = G_layer(fit, .200, layer='W') *w1w/5 *220/ll  # G prediction and error from 200 nm Nb layer one leg
    GI =  (G_layer(fit, dI1, layer='I') + G_layer(fit, dI2, layer='I')) *lw/7 *220/ll   # G prediction and error on insulating layers for one leg
    Gwire = GW + GI # G error for microstrip on one leg, summing error works because error is never negative here

    if layer=='total': return 4*(GU+Gwire)   # value and error, microstrip + substrate on four legs
    elif layer=='wiring': return 4*(Gwire)   # value and error, microstrip (W1+I1+W2+I2) on four legs
    elif layer=='U': return 4*(GU)   # value and error, substrate on four legs
    elif layer=='W': return 4*(GW)   # value and error, W1+W2 on four legs
    elif layer=='W1': return 4*(GW1)   # value and error, W1 on four legs
    elif layer=='I': return 4*(GI)   # value and error, I1+I2 on four legs
    else: print('Invalid layer type.'); return

def predict_G(fit, legacy_data, legacy_geom, bolo1b=False, save_figs=False, estimator='model', kappas=np.array([[54.6, 67.1, 100.9], [6.7, 21.3, 4.7]]), title='', plot_comments='', fs=(7,6), plot_dir='/Users/angi/NIS/Bolotest_Analysis/plots/layer_extraction_analysis/'):

    dsub, lw, ll = legacy_geom
    legacy_AoLs, legacy_Gs = legacy_data

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
    resylim = -600 if 'a01' in plot_comments else -200   # different lower limits on residuals depending on model

    plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])   # model vs data
    plt.scatter(legacy_AoLs, legacy_Gs, color='g', alpha=.6, label=r"Data")
    plt.errorbar(legacy_AoLs, Gpred, yerr=sigma_Gpred, color='k', alpha=.7, marker='*', label=r"G$_\text{TES}$", capsize=2, linestyle='None')
    plt.scatter(legacy_AoLs, GpredU, color='blue', alpha=.7, marker='+', label=r"G$_\text{sub}$")
    plt.scatter(legacy_AoLs, GpredW, color='mediumpurple', alpha=.7, marker='x', s=20, label=r"G$_\text{micro}$")      
    lorder = [0, 3, 1, 2]   # change label order
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder], loc=2)   # 2 is upper left, 4 is lower right
    plt.ylabel('G(170mK) [pW/K]')
    plt.title(title)
    plt.tick_params(axis="y", which="both", right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.ylim(1,5E3)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off

    if bolo1b:   # plot bolotest 1b data and prediction
        data1b = np.array([.0182, 13.51632171, 0.396542])   # bolo1b A/L [um], G and sigma_G [pW/K]
        plt.errorbar(data1b[0], data1b[1], yerr=data1b[2], marker='o', markersize=5, color='purple', label='bolo 1b', capsize=2, linestyle='None')
        if estimator=='model':
            Gpred1b_U = Gfrommodel(fit, .420, 7, 220, layer='U', fab='bolotest')
            Gpred1b_wire = Gfrommodel(fit, .420, 7, 220, layer='wiring', fab='bolotest')
        elif estimator=='kappa':
            Gpred1b_U = Gfromkappas(kappa, .420, 7, 220, layer='U', fab='bolotest')
            Gpred1b_wire = Gfromkappas(kappa, .420, 7, 220, layer='wiring', fab='bolotest')            
        plt.scatter(data1b[0], Gpred1b_wire+Gpred1b_U, marker='*', color='purple')
        plt.scatter(data1b[0], Gpred1b_U, marker='+', color='purple')
        plt.scatter(data1b[0], Gpred1b_wire, marker='x', s=20, color='purple')

    ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
    plt.axhline(0, color='k', ls='--')
    plt.scatter(legacy_AoLs, normres, color='r')
    plt.ylabel("Res'ls [$\%$G]")
    plt.xlabel('Leg A/L [$\mu$m]')
    plt.ylim(resylim,100)
    plt.tick_params(axis="y", which="both", right=True)
    plt.subplots_adjust(hspace=.0)   # merge to share one x axis
    if save_figs: plt.savefig(plot_dir + 'Gpredfrom' + estimator + plot_comments + '.png', dpi=300) 

def scale_G(T, GTc, Tc, n):
    return GTc * T**(n-1)/Tc**(n-1)

def sigma_GscaledT(T, GTc, Tc, n, sigma_GTc, sigma_Tc, sigma_n):   
    Gterm = sigma_GTc * T**(n-1)/(Tc**(n-1))
    Tcterm = sigma_Tc * GTc * (1-n) * T**(n-1)/(Tc**(n))   # this is very tiny
    nterm = sigma_n * GTc * (T/Tc)**(n-1) * np.log(T/Tc)
    return np.sqrt( Gterm**2 + Tcterm**2 + nterm**2)   # quadratic sum of sigma G(Tc), sigma Tc, and sigma_n terms

def WLS_val(params, data, model='default'):   # calculates error-weighted least squares
    ydata, sigma = data
    if model=='default':
        Gbolos_model = Gbolos(params)   # predicted G of each bolo
    elif model=='six_layers':   # model SiOx as it's own layer
        Gbolos_model = Gbolos_six(params)
    WLS_vals = (Gbolos_model-ydata)**2/sigma**2

    return np.sum(WLS_vals) 

def calc_chisq(obs, expect):
    return np.sum((obs-expect)**2/expect)

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


def calc_func_grid(params, data):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = WLS_val(params_rc, data)
    return func_grid

def runsim_WLS(num_its, p0, data, bounds, plot_dir, show_yplots=False, save_figs=False, fn_comments='', save_sim=False, sim_file=None, model='default'):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (most likely pW/K)

    print('\n'); print('Running Minimization MC Simulation'); print('\n')
    ydata, sigma = data

    pfits_sim = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    Gwires = np.empty((num_its, 1))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        it_result = minimize(WLS_val, p0, args=[y_its[ii], sigma], bounds=bounds)
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

    print('Results from Monte Carlo Sim - WLS Min')
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


def qualityplots(data, sim_results, plot_dir='./', save_figs=False, fn_comments='', vmax=500, figsize=(17,5.75), title=''):
    ### plot WLS values in 2D parameter space (alpha_x vs G_x) overlayed with resulting parameters from simulation for all three layers

    layers = np.array(['U', 'W', 'I'])
    fit_params = sim_results[0]; fit_errs = sim_results[1]   # fit parameters and errors

    xgridlim=[0,2]; ygridlim=[0,2]   # alpha_layer vs G_layer 
    xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]   # make 2D grid for plotter

    wspace=.25

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

        funcgrid = calc_func_grid(gridparams, data)   # calculate WLS values for points in the grid
        ticks = np.array([0,0.5,1,1.5,2])
        ax = fig.add_subplot(1,3,ll+1)   # select subplot
        # ax = plt.subplot(gs[ll])   # select subplot
        im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.6)   # quality plot
        plt.errorbar(fit_params[Gind], fit_params[aind], xerr=fit_errs[Gind], yerr=fit_errs[aind], color='black', label='\\textbf{Model Fit}', capsize=2, linestyle='None')   # fit results
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
        plt.annotate(splot_ID, (0.1, 1.825), bbox=dict(boxstyle="square,pad=0.3", fc='w', ec='k', lw=1))
        if ll==2: 
            axpos = ax.get_position()
            cax = fig.add_axes([axpos.x1+0.02, axpos.y0+0.04, 0.01, axpos.y1-axpos.y0-0.08], label='\\textbf{WLS Value}')
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label('\\textbf{WLS Value}', rotation=270, labelpad=15)
            # ax.legend(loc='lower left')
            # ax.legend(loc=(0.075,0.75))
            ax.legend(loc=(0.1,0.15))
    plt.suptitle(title, fontsize=20, y=0.94)

    if save_figs: plt.savefig(plot_dir + 'qualityplots' + fn_comments + '.png', dpi=300)   # save figure
    return 

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

def GandPsatfromNEP(NEP, Tc, Tb, gamma=1):   # calculate G(Tc) and Psat(Tc) given thermal fluctuation NEP, Tc in K, and Tbath in K
    G_Tc = (NEP/Tc)**2 / (4*kB*gamma)   # W/K
    P_Tc = G_Tc*(Tc-Tb)   # W
    return np.array([G_Tc, P_Tc])

def sigma_NEP(T, G, sigma_G):   # error on NEP estimation
    sigma_nepsq = kB/G*T**2 * sigma_G**2
    return np.sqrt(sigma_nepsq)