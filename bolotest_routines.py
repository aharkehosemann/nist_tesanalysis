import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import pickle as pkl
from collections import OrderedDict
import pdb
from scipy.optimize import curve_fit
from numpy.random import normal
from scipy.special import zeta


kB = 1.3806503E-23   # Boltzmann constant, [J/K]
NA = 6.022E23   # Avogadro's number, number of particles in one mole
hbar = 1.055E-34   # reduced Plancks constant = PC / 2pi, [J s]
planck = hbar*2*np.pi   # planck's constant, [J s]

vs_SiN = 6986   # [m/s] average sound speed in SiNx, ; Wang et al
vs_Nb = 3480   # [m/s]a verage sound speed in Nb
vst_SiN = 6.2E3; vsl_SiN = 10.3E3   # [m/s] transverse and longitudinal sound speeds in SiN
vst_Nb = 2.092E3; vsl_Nb = 5.068E3   # [m/s] transverse and longitudinal sound speeds in Nb
rhoNb = 8.57*1E6; rhoSiN = 3.17*1E6   # [g/m^3] mass densities
TD_Nb = 275   # K, Nb Debye temperature
TD_Si = 645   # K, silicon, low temp limit
Tc_Nb = 9.2   # Nb, K
vF_Nb = 1.37E6   # Nb Fermi velocity (electron velocity), m/s
volmol_Nb = 10.84   # cm^3 per mole for Nb = 1E12 um^3
G0 = np.pi**2*kB**2*0.170/(3*planck)*1E12   # an inherent G at 170 mK; pW/K

# bolos=np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])   # this is just always true
bolos=np.array(['1', '2', '3', '4', '5', '6', '7', '8'])   # this is just always true
mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

# bolotest geometry 
# L = 220   # bolotest TES leg length, um
# A_U = 7*420E-3; A_W = 5*400E-3; A_I = 7*400E-3  # Area of film on one leg, um^2
# wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
# A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
# AoL_bolo = A_bolo/L   # A/L for bolotest devices


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

### Thermal Fluctuation Noise Equivalent Power
def FLink(Tb, Tc):   
    # convert equilibrium NEP to nonequilibrium conditions of TES operation
    # Mather 1982
    # assumes k(T)~T^beta, and beta=3 (true for superconductors and crystalline dielectrics)
    Tgrad = Tb/Tc
    return 4/5 * (1-Tgrad**5)/(1-Tgrad**4)

def TFNEP(Tc, G, Tb=0.100):   # calculate thermal fluctuation noise equivalent power as a function of G and T_TES
    return np.sqrt(4*kB*G*FLink(Tb, Tc))*Tc 

def sigma_NEP(Tc, G, sigma_G, Tb=0.100):   # error on NEP estimation
    sigma_nepsq = kB*FLink(Tb, Tc)/G * Tc**2 * sigma_G**2
    return np.sqrt(sigma_nepsq)

def sigmaNEP_sq(Tc, G, sigma_G, Tb=0.100):   # error on NEP^2 = error on S = sqrt(4 kB FL) * T * sigma_G
    # sigma_nep = np.sqrt(kB/G*T**2 * sigma_G**2)
    # return 2*sigma_nep*np.sqrt(4*kB*G*T**2)
    return 4*kB*FLink(Tb, Tc) * Tc**2 * sigma_G

def GandPsatfromNEP(NEP, Tc, Tb, gamma=1):   # calculate G(Tc) and Psat(Tc) given thermal fluctuation NEP, Tc in K, and Tbath in K
    G_Tc = (NEP/Tc)**2 / (4*kB*gamma)   # W/K
    P_Tc = G_Tc*(Tc-Tb)   # W
    return np.array([G_Tc, P_Tc])

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

    scalar_d = False
    if np.isscalar(d):   # handle thickness scalars and arrays
        scalar_d = True
        numdevs = 1   # number of devices = number of thicknesses passed
        d = d    
    else:
        numdevs = len(d)
        d = np.array(d)

    if layer=='U': linds=np.array([0,3]); d0=.420   # substrate layer parameter indexes and default thickness in um
    elif layer=='W': linds=np.array([1,4]); d0=.400   # Nb layer parameter indexes and default thickness in um
    elif layer=='I': linds=np.array([2,5]); d0=.400   # insulating layer parameter indexes and default thickness in um

    numrows = fit.shape[0]   # num of rows determines type of fit passed
    if numrows==1 or len(fit.shape)==1:   # user passed one set of fit parameters with no error bars
        G0, alpha = fit[linds]
        Glayer = G0 * (d/d0)**(alpha+1) 
    elif numrows==2:   # user passed fit parameters and errors
        G0, alpha = fit[0][linds]; sig_G0, sig_alpha = fit[1][linds]   # parameters for layer x
        Glayer = G0 * (d/d0)**(alpha+1) 
        sig_Glayer = np.sqrt( ( sig_G0 * (d/d0)**(alpha+1) )**2 + ( sig_alpha * G0*(alpha+1)*(d/d0)**alpha )**2 )   # check this
        if numdevs==1:
            Glayer = np.array([Glayer, sig_Glayer]).reshape(2, 1)   # [[value], [sigma_value]]
        else:
            Glayer = np.array([Glayer, sig_Glayer]).reshape((2, numdevs))   # [[values for each d], [sigma_values for each d]]
    elif numrows>2:   # user passed many sets of fit parameters, likely the results of a simulation 
        G0s = fit[:, linds[0]]   # G0s for layer x
        alphas = fit[:, linds[1]]   # alphas for layer x
        if scalar_d:
            Glayer = G0s * (d/d0)**(alphas+1)
        else:
            cols = np.array([G0s * (dd/d0)**(alphas+1) for dd in d])   # each row corresponds to results for one value of d = each column of desired output
            Glayer = cols.T   # value for each set of fit parameters (rows) x each thickness d (columns)    

    return Glayer
    

def Gfrommodel(fit, dsub, lw, ll, layer='total', fab='legacy', Lscale=1.0, layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):   # model params, thickness of substrate, leg width, and leg length in um
    # predicts G_TES and error from our model and arbitrary bolo geometry, assumes microstrip on all four legs a la bolo 1b
    # thickness of wiring layers is independent of geometry
    # RETURNS [G prediction, prediction error]

    arrayconv = 1 if np.isscalar(lw) else np.ones(len(lw))   # convert geometry terms to arrays if number of devices > 1

    if fab=='legacy':  # fixed thicknesses
        dW1 = .190*arrayconv; dI1 = .350*arrayconv; dW2 = .400*arrayconv; dI2 = .400*arrayconv   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='legacy')
    elif fab=='bolotest': 
        if len(layer_ds)==10:   # original number of unique film thicknesses
            dW1 = layer_ds[3]*arrayconv; dI1 = layer_ds[5]*arrayconv; dW2 = layer_ds[7]*arrayconv; dI2 = layer_ds[9]*arrayconv   # film thicknesses for microstrip, um
        elif len(layer_ds)==11:   # added one more layer thickness after FIB measurements Feb 2024
            dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC = layer_ds
            dW1 = dW1_ABD*arrayconv; dI1 = dI1_ABC*arrayconv; dW2 = dW2_AC*arrayconv; dI2 = dI2_AC*arrayconv   # film thicknesses for microstrip, um
        w1w = 2; w2w = 4   # um
        # w1w, w2w = wlw(lw, fab='bolotest', layer=layer)
    else: print('Invalid fab type, choose "legacy" or "bolotest."')
    
    GU = (G_layer(fit, dsub, layer='U')) * lw/7 * (220/ll)**Lscale   # G prediction and error on substrate layer for one leg
    GW1 = G_layer(fit, dW1, layer='W') * w1w/5 * (220/ll)**Lscale  # G prediction and error from 200 nm Nb layer one leg
    GW = (G_layer(fit, dW1, layer='W') * w1w/5 + G_layer(fit, dW2, layer='W') * w2w/5) * (220/ll)**Lscale  # G prediction and error from Nb layers for one leg
    GI = (G_layer(fit, dI1, layer='I') + G_layer(fit, dI2, layer='I')) * lw/7 * (220/ll)**Lscale   # G prediction and error on insulating layers for one leg
    Gwire = GW + GI # G error for microstrip on one leg, summing error works because error is never negative here

    if layer=='total': return 4*(GU+Gwire)   # value and error, microstrip + substrate on four legs
    elif layer=='wiring': return 4*(Gwire)   # value and error, microstrip (W1+I1+W2+I2) on four legs
    elif layer=='U': return 4*(GU)   # value and error, substrate on four legs
    elif layer=='W': return 4*(GW)   # value and error, W1+W2 on four legs
    elif layer=='W1': return 4*(GW1)   # value and error, W1 on four legs
    elif layer=='I': return 4*(GI)   # value and error, I1+I2 on four legs
    else: print('Invalid layer type.'); return


def Gbolotest(fit, layer='total', layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):
    # returns G_TES for bolotest data set given fit parameters
    # assumes bolotest geometry
    # derr = error on thickness as a fraction of thickness
    # can return full substrate + microstrip, just substrate, just microstrip, or an individual W / I layer

    # wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20

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

    if len(layer_ds)==10:   # original number of unique film thicknesses
        dS_ABDE, dS_CF, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI1_DF, dW2_AC, dW2_BE, dI2_ACDF = layer_ds
        # dS_ABD, dS_CF, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI1_DF, dW2_AC, dW2_B, dI2_ACDF = layer_ds
        dW2_B = dW2_BE; dI2_AC = dI2_ACDF; dS_ABD = dS_ABDE; dS_E = dS_ABDE   # handle renaming 
        dW1_E = dW1_E+dW2_B; dI_DF = dI1_DF +dI2_ACDF   # handle combining W and I stacks
    elif len(layer_ds)==11:   # added one more layer thickness after FIB measurements Feb 2024
        dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC = layer_ds

    G_legA = G_layer(fit, dS_ABD, layer='U')*include_U + G_layer(fit, dW1_ABD, layer='W')*include_W + G_layer(fit, dI1_ABC, layer='I')*include_I + G_layer(fit, dW2_AC, layer='W')*3/5*include_W + G_layer(fit, dI2_AC, layer='I')*include_I # S-W1-I1-W2-I2
    G_legB = G_layer(fit, dS_ABD, layer='U')*include_U + G_layer(fit, dW1_ABD, layer='W')*include_W + G_layer(fit, dI1_ABC, layer='I')*3/7*include_I + G_layer(fit, dW2_B, layer='W')*3/5*include_W   # S-W1-I1-W2
    G_legC = G_layer(fit, dS_CF, layer='U')*include_U + G_layer(fit, dI1_ABC, layer='I') + G_layer(fit, dW2_AC, layer='W')*3/5*include_W + G_layer(fit, dI2_AC, layer='I')*include_I   # S-I1-W2-I2
    G_legD = G_layer(fit, dS_ABD, layer='U')*include_U + G_layer(fit, dW1_ABD, layer='W')*include_W + G_layer(fit, dI_DF, layer='I')*include_I   # S-W1-I1-I2 (I stack)
    G_legE = G_layer(fit, dS_E, layer='U')*include_U + G_layer(fit, dW1_E, layer='W')*3/5*include_W   # S-W1-W2 (W stack)
    G_legF = G_layer(fit, dS_CF, layer='U')*include_U + G_layer(fit, dI_DF, layer='I')*include_I   # S-I1-I2 (I stack)
    G_legG = G_layer(fit, dS_G, layer='U')*include_U   # bare S 

    G_1b = 4*G_legA
    G_24 = 1*G_legA + 3*G_legG
    G_23 = 2*G_legA + 2*G_legG
    G_22 = 3*G_legA + 1*G_legG
    G_21 = 1*G_legB + 3*G_legF
    G_20 = 1*G_legB + 3*G_legE
    G_7 = 2*G_legA + 1*G_legC + 1*G_legD
    G_13 = 1*G_legB + 3*G_legG

    # if len(fit.shape)==2:   # return values and errors
    if len(fit.shape)==1 or np.shape(fit)[0]==1:   # return values and errors
        return np.array([G_1b, G_24, G_23, G_22, G_21, G_20, G_7, G_13])   # return values
    elif np.shape(fit)[0]==2:   # return values and errors
        Gbolos = np.array([G_1b[0], G_24[0], G_23[0], G_22[0], G_21[0], G_20[0], G_7[0], G_13[0]]).T; sigma_Gbolos = np.array([G_1b[1], G_24[1], G_23[1], G_22[1], G_21[1], G_20[1], G_7[1], G_13[1]]).T
        return Gbolos, sigma_Gbolos
    else:   # return values for each row, prob result of simulation
        return np.array([G_1b, G_24, G_23, G_22, G_21, G_20, G_7, G_13]).T        
    

### fitting free parameters of model
def chisq_val(params, args, model='default', layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):   # calculates chi-squared value

    # if len(args)==4:
    #     ydata, sigma, vary_thickness, layer_ds = args
    if len(args)==3:
        ydata, sigma, layer_ds = args
    elif len(args)==2:
        ydata, sigma = args
    Gbolos_model = Gbolotest(params, layer_ds=layer_ds)   # predicted G of each bolo
    chisq_vals = (Gbolos_model-ydata)**2/sigma**2
    
    return np.sum(chisq_vals)


def calc_func_grid(params, data, layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            params_rc = col            
            func_grid[rr, cc] = chisq_val(params_rc, data, layer_ds=layer_ds)
    return func_grid

def runsim_chisq(num_its, p0, data, bounds, plot_dir, show_simGdata=False, save_figs=False, fn_comments='', save_sim=False, sim_file=None,
                  vary_thickness=False, derr=0.0, layer_d0=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400]),
                  derrs = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])):  
    # returns G and alpha fit parameters
    # returned G's have units of ydata (most likely pW/K)

    print('\n'); print('Running MC Simulation'); print('\n')
    ydata, sigma = data
    pfits_sim = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    Gwires = np.empty((num_its, 1))
    layer_ds = np.empty((num_its, len(layer_d0)))
    for ii in np.arange(num_its):   # run simulation

        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error

        if vary_thickness:   # pull thicknesses from normal distribution, assuming error is some % of d
            if len(layer_d0)==10:
                dS_ABDE = normal(layer_d0[0], derr*layer_d0[0]); dS_CF = normal(layer_d0[1], derr*layer_d0[1]); dS_G = normal(layer_d0[2], derr*layer_d0[2])   # thickness of S for leg A, C, and G [um]
                dW1_ABD = normal(layer_d0[3], derr*layer_d0[3]); dW1_E = normal(layer_d0[4], derr*layer_d0[4])   # thickness of W1 for legs A and E [um]  
                dI1_ABC = normal(layer_d0[5], derr*layer_d0[5]); dI1_DF = normal(layer_d0[6], derr*layer_d0[6])   # thickness of I1 for legs A and G [um]  
                dW2_AC = normal(layer_d0[7], derr*layer_d0[7]); dW2_BE = normal(layer_d0[8], derr*layer_d0[8])   # thickness of W2 for legs A and B [um]  
                dI2_ACDF = normal(layer_d0[9], derr*layer_d0[9])   # thickness of I1 for legs A and G [um]  
                layer_ds[ii] = dS_ABDE, dS_CF, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI1_DF, dW2_AC, dW2_BE, dI2_ACDF
            elif len(layer_d0)==11:
                dS_ABD0, dS_CF0, dS_E0, dS_G0, dW1_ABD0, dW1_E0, dI1_ABC0, dI_DF0, dW2_AC0, dW2_BE0, dI2_AC0 = layer_d0
                dSABD_err, dSCF_err, dSE_err, dSG_err, dW1ABD_err,  dW1E_err, dI1ABC_err, dIDF_err, dW2AC_err, dW2BE_err, dI2AC_err= derrs

                dS_ABD = normal(dS_ABD0, dSABD_err); dS_CF = normal(dS_CF0, dSCF_err); dS_E = normal(dS_E0, dSE_err); dS_G = normal(dS_G0, dSG_err)   # thickness of S for leg A, C, and G [um]
                dW1_ABD = normal(dW1_ABD0, dW1ABD_err); dW1_E = normal(dW1_E0, dW1E_err)   # thickness of W1 for legs A and E [um]  
                dI1_ABC = normal(dI1_ABC0, dI1ABC_err); dI_DF = normal(dI_DF0, dIDF_err)   # thickness of I1 for legs A and G [um]  
                dW2_AC = normal(dW2_AC0, dW2AC_err); dW2_BE = normal(dW2_BE0, dW2BE_err)   # thickness of W2 for legs A and B [um]  
                dI2_AC = normal(dI2_AC0, dI2AC_err)   # thickness of I1 for legs A and C [um]  
                layer_ds[ii] = dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_BE, dI2_AC        
        else:
            layer_ds[ii] = layer_d0 
        
        it_result = minimize(chisq_val, p0, args=[y_its[ii], sigma, layer_ds[ii]], bounds=bounds)   # minimize chi-squared function with this iteration's G_TES values and film thicknesses
        pfits_sim[ii] = it_result['x']

        Gwires[ii] = Gfrommodel(pfits_sim[ii], 0.420, 7, 220, layer='wiring', fab='bolotest', layer_ds=layer_ds[ii])/4   # function outputs G for four legs worth of microstrip
        # Gfrommodel(fit, dsub, lw, ll, layer='wiring')
    if show_simGdata:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            # plt.figure()
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
        sim_dict = {}
        sim_dict['sim'] = pfits_sim   # add simulation 
        sim_dict['Gwires'] = Gwires   
        sim_dict['sim_params'] = {}
        sim_dict['sim_params']['num_its'] = num_its   
        sim_dict['sim_params']['p0'] = p0  
        sim_dict['fit'] = {}
        sim_dict['fit']['fit_params'] = sim_params   
        sim_dict['fit']['fit_std'] = sim_std  
        sim_dict['fit']['Gwire'] = Gwire  
        sim_dict['fit']['sigma_Gwire'] = Gwire_std  
        sim_dict['fit']['Gwire'] = Gwire   # pW/K
        sim_dict['fit']['Gwire_std'] = Gwire_std   # pW/K
        print('Saving simulation to ', sim_file); print('\n')
        with open(sim_file, 'wb') as outfile:   # save simulation pkl
            pkl.dump(sim_dict, outfile)

    # return sim_params, sim_std
    return sim_dict


### visualize and evaluate quality of fit
def qualityplots(data, sim_dict, plot_dir='./', save_figs=False, fn_comments='', vmax=2E3, figsize=(17,5.75), title='', print_results=True, calc='Mean', spinds=[], plot=True, qplim=[0,2], layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):
    ### plot chisq values in 2D parameter space (alpha_x vs G_x) overlayed with resulting parameters from simulation for all three layers
    # params can be either the mean or median of the simulation values
    # spinds are indexes of a certain subpopulation to plot. if the length of this is 0, it will analyze the entire population. 

    layers = np.array(['U', 'W', 'I'])
    A_U = 7*0.420; A_W = 5*0.400; A_I = 7*0.400   # um^2
    L = 220   # um
    
    if type(sim_dict)==dict:

        # sim_dataT = sim_dict['sim']; simdata_temp = sim_dataT.T 
        simdata_temp = sim_dict['sim'] 
        if len(spinds)==0: spinds = np.arange(np.shape(simdata_temp)[0])
        sim_data = simdata_temp[spinds,:]
        Gwires = sim_dict['Gwires'][spinds]

        # calculate the fit params as either the mean or median of the simulation values
        if calc == 'Mean':
            fit_params, fit_errs = [np.mean(sim_data, axis=0), np.std(sim_data, axis=0)]   # take mean values
            Gwire = np.mean(Gwires); sigma_Gwire = np.std(Gwires)
        if calc == 'Median':
            fit_params, fit_errs = [np.median(sim_data, axis=0), np.std(sim_data, axis=0)]   # take median values to avoid outliers
            Gwire = np.median(Gwires); sigma_Gwire = np.std(Gwires)
    else:   # option to pass just fit parameters
        fit_params = sim_dict
        fit_errs = np.array([0,0,0,0,0,0])
        Gwire = Gfrommodel(fit_params, 0.420, 7, 220, layer='wiring', fab='bolotest')/4; sigma_Gwire=0
        
    chisq_fit = chisq_val(fit_params, data, layer_ds=layer_ds)

    if plot:
        xgridlim=qplim; ygridlim=qplim   # alpha_layer vs G_layer 
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

            funcgrid = calc_func_grid(gridparams, data, layer_ds=layer_ds)   # calculate chisq values for points in the grid
            ax = fig.add_subplot(1,3,ll+1)   # select subplot
            im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.6)   # quality plot
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
        print ('\n\nResults taking '+ calc +' values:')
        print('G_U(420 nm) = ', round(GmeasU, 2), ' +/- ', round(sigGU, 2), 'pW/K')
        print('G_W(400 nm) = ', round(GmeasW, 2), ' +/- ', round(sigGW, 2), 'pW/K')
        print('G_I(400 nm) = ', round(GmeasI, 2), ' +/- ', round(sigGI, 2), 'pW/K')
        print('alpha_U = ', round(alphaU, 2), ' +/- ', round(sigalphaU, 2))
        print('alpha_W = ', round(alphaW, 2), ' +/- ', round(sigalphaW, 2))
        print('alpha_I = ', round(alphaI, 2), ' +/- ', round(sigalphaI, 2))
        print('')
        kappaU = GtoKappa(GmeasU, A_U, L); sigkappaU = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
        kappaW = GtoKappa(GmeasW, A_W, L); sigkappaW = GtoKappa(sigGW, A_W, L)   # pW / K / um
        kappaI = GtoKappa(GmeasI, A_I, L); sigkappaI = GtoKappa(sigGI, A_I, L)   # pW / K / um
        print('Kappa_U: ', round(kappaU, 2), ' +/- ', round(sigkappaU, 2), ' pW/K/um')
        print('Kappa_W: ', round(kappaW, 2), ' +/- ', round(sigkappaW, 2), ' pW/K/um')
        print('Kappa_I: ', round(kappaI, 2), ' +/- ', round(sigkappaI, 2), ' pW/K/um')
        print('G_wire = ', round(Gwire, 2), ' +/- ', round(sigma_Gwire, 2), 'pW/K')

        print('Chi-squared value: ', round(chisq_fit, 3)) 

    return fit_params, fit_errs, [kappaU, kappaW, kappaI], [sigkappaU, sigkappaW, sigkappaI], Gwire, sigma_Gwire, chisq_fit

def pairwise(sim_data, labels, title='', plot_dir='./', fn_comments='', save_figs=False, indstp=[], indsop=[], oplotlabel='', fs=(10,8)):
    # make pairwise correlation plots with histograms on the diagonal 
    # indstp = index of solutions to plot, default is all
    # indsop = index of subset of solutions to overplot on all solutions

    sim_dataT = sim_data.T   # sim_data needs to be transposed so that it's 6 x number of iterations
    if len(indstp)==0: indstp = np.arange(np.shape(sim_dataT)[1])   # allow for plotting subsections of simulation data 
    nsolns = len(indsop) if len(indsop)!=0 else len(indstp)   # count number of solutions, if overplotting count number of overplotted solutions
    ndim = len(sim_dataT)   # number of dimensions 

    limpad = np.array([max(sim_dataT[pp][indstp])-min(sim_dataT[pp][indstp]) for pp in np.arange(len(sim_dataT))])*0.10   # axis padding for each parameter
    limits = np.array([[min(sim_dataT[pp][indstp])-limpad[pp], max(sim_dataT[pp][indstp])+limpad[pp]] for pp in np.arange(len(sim_dataT))])   # axis limits for each parameter
    histlim = [1,1E4]

    pairfig = plt.figure(figsize=fs)
    for ii in np.arange(ndim):   # row
        for jj in range(ndim):   # column
            spind = ii*ndim+jj+1   # subplot index, starts at 1
            ax = pairfig.add_subplot(ndim, ndim, spind)
            if ii == jj:   # histograms on the diagonal
                hist, bins, patches = ax.hist(sim_dataT[ii][indstp], bins=30, color='C1', label='All')
                ax.hist(sim_dataT[ii][indsop], bins=bins, color='C2', histtype='stepfilled', alpha=0.5, label=oplotlabel)  # highlight subset of solutions
                ax.set_yscale('log')
                ax.yaxis.set_ticks([1E1, 1E2, 1E3, 1E4])
                ax.set_xlim(limits[ii]); ax.set_ylim(histlim)
                ax.set_xlabel(labels[jj]); ax.set_ylabel(labels[ii])
            else:           
            # elif jj<=ii:   # scatter plots on off-diagonal 
                ax.scatter(sim_dataT[jj][indstp], sim_dataT[ii][indstp], marker='.', alpha=0.3)   # row shares the y axis, column shares the x axis
                ax.scatter(sim_dataT[jj][indsop], sim_dataT[ii][indsop], marker='.', alpha=0.3, color='C2')   # highlight subset of solutions
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


def bolotest_AoL(L=220, layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):
    # calculate bolotest leg xsect area over length

    if len(layer_ds)==10:   # original number of unique film thicknesses
        dS_ABDE, dS_CF, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI1_DF, dW2_AC, dW2_BE, dI2_ACDF = layer_ds
        dW2_B = dW2_BE; dI2_AC = dI2_ACDF; dS_ABD = dS_ABDE; dS_E = dS_ABDE   # handle renaming 
        dW1_E = dW1_E+dW2_B; dI_DF = dI1_DF +dI2_ACDF   # handle combining W and I stacks
    elif len(layer_ds)==11:   # added one more layer thickness after FIB measurements Feb 2024
        dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC = layer_ds

    # wstack_width = (5*dW1_E+3*dW2_BE)/(dW1_E+dW2_BE)   # um, effective width of W1 W2 stack on bolo 20
    A_legA = dS_ABD*7 + dW1_ABD*5      + dI1_ABC*7             + dW2_AC*3   + dI2_AC*7     # S-W1-I1-W2-I2
    A_legB = dS_ABD*7 + dW1_ABD*5      + dI1_ABC*3             + dW2_B*3    + 0            # S-W1-I1-W2, I1 width is actually = W2 width here from FIB measurements
    A_legC = dS_CF*7  + 0              + dI1_ABC*7             + dW2_AC*3   + dI2_AC*7     # S-I1-W2-I2
    A_legD = dS_ABD*7 + dW1_ABD*5      + dI_DF*7               + 0          + 0            # S-W1-I1-I2 (I stack)
    A_legE = dS_E*7   + (dW1_E)*3       + 0                     + 0          + 0            # S-W1-W2 (W stack)
    A_legF = dS_CF*7  + 0              + dI_DF*7               + 0          + 0            # S-I1-I2 (I stack)
    A_legG = dS_G*7   + 0              + 0                     + 0          + 0            # bare S 

    ### bolo 1b = 4×A, 24 = 1×A & 3×G, 23 = 2×A & 2×G, 22= 3×A & 1×G, 21 = 1×B & 3×F, 20 = 1×B & 3×E, 7 = 2×A & 1×C & 1×D, 13=1×B & 3×G 
    A_bolo1b = 4*A_legA; A_bolo24 = 1*A_legA + 3*A_legG; A_bolo23 = 2*A_legA + 2*A_legG; A_bolo22 = 3*A_legA + 1*A_legG; 
    A_bolo21 = 1*A_legB + 3*A_legF; A_bolo20 = 1*A_legB + 3*A_legE; A_bolo7 = 2*A_legA + 1*A_legC + 1*A_legD; A_bolo13 = 1*A_legB + 3*A_legG
    A_bolo = np.array([A_bolo1b, A_bolo24, A_bolo23, A_bolo22, A_bolo21, A_bolo20, A_bolo7, A_bolo13])
    AoL_bolo = A_bolo/L   # A/L for bolotest devices

    return AoL_bolo   # 1b, 24, 23, 22, 21, 20, 7, 13



def plot_modelvdata(sim_data, data, title='', vlength_data=np.array([]), plot_bolotest=True, Lscale=1.0, pred_wfit=True, calc='Mean', layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400]), save_figs=False, plot_dir='./', plot_comments='', fs=(8,6)):
    # plot bolotest data vs model fit
    # fit = [params, sigma_params]
    # data = [Gbolos, sigma_Gbolos]
    # plot_bolotest gives the option of turning off bolos 24-20 etc with various film stacks

    AoL_bolo = bolotest_AoL(layer_ds=layer_ds)

    ### predictions 
    # calculate fit parameters
    if calc=='Mean':
        fit = np.array([np.mean(sim_data, axis=0), np.std(sim_data, axis=0)])
    elif calc=='Median':
        fit = np.array([np.median(sim_data, axis=0), np.std(sim_data, axis=0)])
    else: 
        print('Unknown calculation method {calc}, select "mean" or "median".'.format(calc=calc))

    # calculate predictions and error bars either with fit parameters or std of predictions from all simulated fit parameters
    if pred_wfit:   # use error bars on fit parameters to calculate error bars on predicted values
        Gpred, sigmaGpred = Gbolotest(fit, layer_ds=layer_ds)   # predictions and error from model [pW/K]
        Gpred_wire, sigmaGpred_wire = Gbolotest(fit, layer='wiring', layer_ds=layer_ds)   # predictions and error from model [pW/K]
        Gpred_U, sigmaGpred_U = Gbolotest(fit, layer='U', layer_ds=layer_ds)   # predictions and error from model [pW/K]
    else:   # calculate G predictions from each simulated fit, then take mean and std
        Gpreds = Gbolotest(sim_data, layer_ds=layer_ds)   # predictions from each set of fit parameters [pW/K]
        GpredWs = Gbolotest(sim_data, layer_ds=layer_ds, layer='wiring')
        GpredUs = Gbolotest(sim_data, layer_ds=layer_ds, layer='U')

        if calc=='Mean':
            Gpred = np.mean(Gpreds, axis=0); sigmaGpred = np.std(Gpreds, axis=0)   # predictions and error [pW/K]
            Gpred_wire = np.mean(GpredWs, axis=0)   # predictions and error of W layers [pW/K]
            Gpred_U = np.mean(GpredUs, axis=0)   # predictions and error of substrate layers [pW/K]
        elif calc=='Median':
            Gpred = np.median(Gpreds, axis=0); sigmaGpred = np.std(Gpreds, axis=0)   # predictions and error [pW/K]
            Gpred_wire = np.median(GpredWs, axis=0)   # predictions and error of W layers [pW/K]
            Gpred_U = np.median(GpredUs, axis=0)   # predictions and error of substrate layers [pW/K]

    if len(vlength_data)>0:   # show predictions for bolos1a-f; they share the same geometry as bolo 1b, leg length is varied
        ydatavl_all, sigmavl_all, llvl_all = vlength_data   # send leg lengths in um
        ll_vl = np.array(llvl_all[0:6])  # um
        AoL_vL = A_bolo[0]/ll_vl   # um, all share the area of bolo 1b
        dsub = 0.420; lw = 7   # um

        if pred_wfit:
            Gpred_vl, sigmaGpred_vl = Gfrommodel(fit, dsub, lw, ll_vl, layer='total', fab='bolotest', Lscale=Lscale)
            Gpredwire_vl, sigmaGpredwire_vl = Gfrommodel(fit, dsub, lw, ll_vl, layer='wiring', fab='bolotest', Lscale=Lscale)
            GpredU_vl, sigmaGpredU_vl = Gfrommodel(fit, dsub, lw, ll_vl, layer='U', fab='bolotest', Lscale=Lscale)
        else:
            Gpred_vls = Gfrommodel(sim_data, dsub, lw, ll_vl, layer='total', fab='bolotest', Lscale=Lscale)
            Gpredwire_vls = Gfrommodel(sim_data, dsub, lw, ll_vl, layer='wiring', fab='bolotest', Lscale=Lscale)
            GpredU_vls = Gfrommodel(sim_data, dsub, lw, ll_vl, layer='U', fab='bolotest', Lscale=Lscale)

            if calc=='Mean':
                Gpred_vl = np.mean(Gpred_vls, axis=0); sigmaGpred_vl = np.std(Gpred_vls, axis=0)   # predictions and error [pW/K]
                Gpredwire_vl = np.mean(Gpredwire_vls, axis=0)   # predictions and error of W layers [pW/K]
                GpredU_vl = np.mean(GpredU_vls, axis=0)   # predictions and error of substrate layers [pW/K]
            elif calc=='Median':
                Gpred_vl = np.median(Gpred_vls, axis=0); sigmaGpred_vl = np.std(Gpred_vls, axis=0)   # predictions and error [pW/K]
                Gpredwire_vl = np.median(Gpredwire_vls, axis=0)   # predictions and error of W layers [pW/K]
                GpredU_vl = np.median(GpredU_vls, axis=0)   # predictions and error of substrate layers [pW/K]
    chisq_fit = chisq_val(fit[0], data, layer_ds=layer_ds)

    plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax1 = plt.subplot(gs[0])   # model vs data
    if plot_bolotest:
        plt.errorbar(AoL_bolo, data[0], yerr=data[1], marker='o', markersize=5, color='g', capsize=2, linestyle='None')
        plt.plot(AoL_bolo, Gpred_wire, color='mediumpurple', marker='x', label=r"G$_\text{micro}$", linestyle='None')
        plt.plot(AoL_bolo, Gpred_U, markersize=5, color='blue', marker='+', label=r"G$_\text{sub}$", linestyle='None')
        plt.errorbar(AoL_bolo, Gpred, yerr=sigmaGpred, color='k', marker='*', label=r"G$_\text{TES}$", capsize=2, linestyle='None')
        for bb, boloid in enumerate(bolos):
            # plt.annotate(boloid.split(' ')[1], (AoL_bolo[bb]+0.001, data[0][bb]+0.4))
            plt.annotate(boloid, (AoL_bolo[bb]+0.0012, data[0][bb]+0.5))
        plt.annotate('$\\boldsymbol{\\chi^2}$ = '+str(round(chisq_fit)), (min(AoL_bolo)*1.015, max(data[0])*0.68), bbox=dict(boxstyle="square,pad=0.3", fc='w', ec='k', lw=1))
        normres = (data[0] - Gpred)/data[0]
        norm_ressigma = sigmaGpred/data[0]
        plt.legend()
        
    if len(vlength_data)>0:
        # plt.errorbar(A_bolo[0]/llvl_all, ydatavl_all, yerr=sigmavl_all, marker='o', markersize=5, color='g', capsize=2, linestyle='None', label='Bolos 1a-f')
        plt.errorbar(A_bolo[0]/llvl_all, ydatavl_all, yerr=sigmavl_all, marker='o', markersize=5, color='g', capsize=2, linestyle='None')
        # plt.errorbar(AoL_vL, Gpredwire_vl.flatten(), yerr=sigmaGpredwire_vl.flatten(), color='mediumpurple', marker='x', linestyle='None')
        # plt.errorbar(AoL_vL, GpredU_vl.flatten(), yerr=sigmaGpredU_vl.flatten(), markersize=5, color='blue', marker='+', linestyle='None')
        plt.plot(AoL_vL, Gpredwire_vl.flatten(), color='mediumpurple', marker='x', linestyle='None')
        plt.plot(AoL_vL, GpredU_vl.flatten(), markersize=5, color='blue', marker='+', linestyle='None')
        plt.errorbar(AoL_vL, Gpred_vl.flatten(), yerr=sigmaGpred_vl.flatten(), color='k', marker='*', capsize=2, linestyle='None')
    plt.ylabel('$\\textbf{G(170mK) [pW/K]}$')
    plt.title(title)
    # plt.xlabel('Leg A/L [$\mu$m]')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
    plt.ylim(1, 19)

    ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
    plt.axhline(0, color='k', alpha=0.7)
    plt.scatter(AoL_bolo, normres, color='g', s=40, alpha=0.8)
    plt.ylabel("\\textbf{Norm. Res.}")
    plt.xlabel('Leg A/L [$\mu$m]')
    plt.ylim(-0.1, 0.1)
    plt.tick_params(axis="y", which="both", right=True)
    # plt.gca().yaxis.set_ticks([-2, -1, 0, 1])
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
    if save_figs: plt.savefig(plot_dir + 'Gpred_bolotest' + plot_comments + '.png', dpi=300) 
    return 

def plot_Glegacy(data1b=[], save_figs=False, title='', plot_comments='', Lscale=1, lAoLscale=None, fs=(7,5), plot_dir='/Users/angi/NIS/Bolotest_Analysis/plots/layer_extraction_analysis/', layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400])):
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

    # only concerned with low Tc data
    dsub = legacy_dsub[lTcinds]; lw = legacy_lw[lTcinds]; ll = legacy_ll[lTcinds]
    legacy_AoLs = legacyAoLs_all[lTcinds]; legacy_Gs = legacyGs170_all[lTcinds] 
    lAoLinds = np.where(legacy_AoLs<1)[0]   # bolos with A/L < 1 um
    hAoLinds = np.where(legacy_AoLs>1)[0] 
    # pdb.set_trace()

    if lAoLscale:   # are A/L values for A/L<1um actually larger?
        legacy_AoLs[lAoLinds] = legacy_AoLs[lAoLinds]*lAoLscale

    lfit = np.polyfit(np.log10(legacy_AoLs[lAoLinds]), np.log10(legacy_Gs[lAoLinds]), 1)
    hfit = np.polyfit(np.log10(legacy_AoLs[hAoLinds]), np.log10(legacy_Gs[hAoLinds]), 1)
    afit = np.polyfit(np.log10(legacy_AoLs), np.log10(legacy_Gs), 1)

    # aol_low = np.logspace(np.log10(min(legacy_AoLs[lAoLinds])), np.log10(max(legacy_AoLs[lAoLinds]))); aol_high = np.logspace(np.log10(min(legacy_AoLs[hAoLinds])), np.log10(max(legacy_AoLs[hAoLinds])))                                                     
    aol_low = np.linspace(min(legacy_AoLs[lAoLinds]), max(legacy_AoLs[lAoLinds])); aol_high = np.linspace(min(legacy_AoLs[hAoLinds]), max(legacy_AoLs[hAoLinds]) )                                                     
    fline_low = 10**(lfit[0]*np.log10(aol_low) + lfit[1]); fline_high = 10**(hfit[0]*np.log10(aol_high) + hfit[1])   # best fit lines
    fline_all = 10**(afit[0]*np.log10(legacy_AoLs) + afit[1])   
                   
    plt.figure(figsize=fs)
    plt.scatter(legacy_AoLs, legacy_Gs, color='g', alpha=.8, label=r"Legacy Data", s=40)     
    # plt.plot(aol_low, fline_low, color='r', alpha=.8, label='Slope = '+str(round(lfit[0], 2)))     
    # plt.plot(aol_high, fline_high, color='b', alpha=.8, label='Slope = '+str(round(hfit[0], 2)))     
    plt.plot(legacy_AoLs, fline_all, color='darkorange', alpha=.8, label='Slope = '+str(round(afit[0], 2)))     
    if len(data1b)==2:   # plot bolotest 1b data and prediction
        AoL_bolo = bolotest_AoL(layer_ds=layer_ds)
        plt.errorbar(AoL_bolo[0], data1b[0], yerr=data1b[1], marker='o', markersize=5, color='purple', label='Bolo 1b', capsize=2, linestyle='None')
    plt.ylabel('\\textbf{G(170mK) [pW/K]}'); plt.xlabel('Leg A/L [$\mu$m]')
    plt.title(title)
    plt.tick_params(axis="y", which="both", right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.ylim(2,1.8E3); plt.xlim(0.09, 13.7)
    plt.legend(loc=4)

    if save_figs: plt.savefig(plot_dir + 'legacydata' + plot_comments + '.png', dpi=300) 


def predict_Glegacy(sim_data, data1b=[], save_figs=False, title='', calc='Mean', plot_comments='', fs=(8,6), Lscale=1, pred_wfit=True, lAoLscale=None, layer_ds=np.array([0.420, 0.400, 0.340, 0.160, 0.100, 0.350, 0.270, 0.340, 0.285, 0.400]), plot_dir='/Users/angi/NIS/Bolotest_Analysis/plots/layer_extraction_analysis/'):
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

    # only concerned with low Tc data
    dsub = legacy_dsub[lTcinds]; lw = legacy_lw[lTcinds]; ll = legacy_ll[lTcinds]
    legacy_AoLs = legacyAoLs_all[lTcinds]; legacy_Gs = legacyGs170_all[lTcinds] 
    if lAoLscale:   # are A/L values for A/L<1um actually larger?
        lAoLinds = np.where(legacy_AoLs<1)[0]   # bolos with A/L < 1 um
        legacy_AoLs[lAoLinds] = legacy_AoLs[lAoLinds]*lAoLscale

    ### predictions 
    # calculate fit parameters
    if calc=='Mean':
        fit = np.array([np.mean(sim_data, axis=0), np.std(sim_data, axis=0)])
    elif calc=='Median':
        fit = np.array([np.median(sim_data, axis=0), np.std(sim_data, axis=0)])
    else: 
        print('Unknown calculation method {calc}, select "mean" or "median".'.format(calc=calc))

    # calculate predictions and error bars either with fit parameters or std of predictions from all simulated fit parameters
    if pred_wfit:   # use error bars on fit parameters to calculate error bars on predicted values
        Gpred, sigma_Gpred = Gfrommodel(fit, dsub, lw, ll, Lscale=Lscale)   # predictions and error from model [pW/K]
        GpredW, sigma_GpredW = Gfrommodel(fit, dsub, lw, ll, layer='wiring', Lscale=Lscale)
        GpredU, sigma_GpredU = Gfrommodel(fit, dsub, lw, ll, layer='U', Lscale=Lscale)
        Gpred1b_U, sigma_G1bU = Gfrommodel(fit, .420, 7, 220, layer='U', fab='bolotest')  # bolo 1b predictions
        Gpred1b_wire, sigma_G1bwire = Gfrommodel(fit, .420, 7, 220, layer='wiring', fab='bolotest')   # bolo 1b predictions
    else:   # calculate G predictions from each simulated fit, then take mean and std
        Gpreds = Gfrommodel(sim_data, dsub, lw, ll, Lscale=Lscale)   # predictions from each set of fit parameters [pW/K]
        GpredWs = Gfrommodel(sim_data, dsub, lw, ll, layer='wiring', Lscale=Lscale)
        GpredUs = Gfrommodel(sim_data, dsub, lw, ll, layer='U', Lscale=Lscale)
        Gpred1b_Us = Gfrommodel(sim_data, .420, 7, 220, layer='U', fab='bolotest')
        Gpred1b_wires = Gfrommodel(sim_data, .420, 7, 220, layer='wiring', fab='bolotest')

        Gpred = np.mean(Gpreds, axis=0); sigma_Gpred = np.std(Gpreds, axis=0)   # predictions and error [pW/K]
        GpredW = np.mean(GpredWs, axis=0)   # predictions and error of W layers [pW/K]
        GpredU = np.mean(GpredUs, axis=0)   # predictions and error of substrate layers [pW/K]
        Gpred1b_U = np.mean(Gpred1b_Us); sigma_G1bU = np.std(Gpred1b_Us)   # predictions and error [pW/K]
        Gpred1b_wire = np.mean(Gpred1b_wires); sigma_G1bwire = np.std(Gpred1b_wires)   # predictions and error [pW/K]

    normres = (legacy_Gs - Gpred)/legacy_Gs   # normalized residuals [frac of data]
    resylim = -6 if 'a01' in plot_comments else -2   # different lower limits on residuals depending on model

    plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax1 = plt.subplot(gs[0])   # model vs data
    plt.scatter(legacy_AoLs, legacy_Gs, color='g', alpha=.8, label=r"Legacy Data", s=40)
    plt.errorbar(legacy_AoLs, Gpred, yerr=sigma_Gpred, color='k', marker='*', label=r"G$_\text{TES}$", capsize=2, linestyle='None', markersize=7)
    # plt.errorbar(legacy_AoLs, GpredU, yerr=sigma_GpredU, color='blue', marker='+', label=r"G$_\text{sub}$", linestyle='None')
    # plt.errorbar(legacy_AoLs, GpredW, yerr=sigma_GpredW, color='mediumpurple', marker='x', label=r"G$_\text{micro}$", linestyle='None')  
    plt.scatter(legacy_AoLs, GpredU, color='mediumblue', marker='+', label=r"G$_\text{sub}$", linestyle='None', s=60)
    plt.scatter(legacy_AoLs, GpredW, color='darkorchid', marker='x', label=r"G$_\text{micro}$", linestyle='None', s=40)      
    plt.ylabel('\\textbf{G(170mK) [pW/K]}')
    plt.title(title)
    plt.tick_params(axis="y", which="both", right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.ylim(2,2E3)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off

    if len(data1b)==2:   # plot bolotest 1b data and prediction
        AoL_bolo = bolotest_AoL(layer_ds=layer_ds)
        lorder = [0, 3, 1, 2, 4]   # legend label order
        plt.errorbar(AoL_bolo[0], data1b[0], yerr=data1b[1], marker='o', markersize=5, color='purple', label='Bolo 1', capsize=2, linestyle='None')
        plt.errorbar(AoL_bolo[0], Gpred1b_wire+Gpred1b_U, yerr=(sigma_G1bwire+sigma_G1bU), marker='*', color='purple')
        plt.errorbar(AoL_bolo[0], Gpred1b_U, marker='+', color='purple')
        plt.errorbar(AoL_bolo[0], Gpred1b_wire, marker='x', markersize=5, color='purple')
    else:
        lorder = [0, 3, 1, 2]   # legend label order

    plt.gca().yaxis.set_ticks([1E1, 1E2, 1E3])
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder], loc=4)   # 2 is upper left, 4 is lower right

    ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
    plt.axhline(0, color='k', alpha=0.7)
    plt.scatter(legacy_AoLs, normres, color='g', s=40, alpha=0.8)
    if len(data1b)==2: plt.scatter(AoL_bolo[0], (data1b[0]-Gpred1b_wire-Gpred1b_U)/data1b[0], color='purple')
    plt.ylabel("\\textbf{Norm. Res.}")
    plt.xlabel('Leg A/L [$\mu$m]')
    plt.ylim(resylim,1)
    plt.tick_params(axis="y", which="both", right=True)
    plt.gca().yaxis.set_ticks([-2, -1, 0, 1])
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
    if save_figs: plt.savefig(plot_dir + 'Gpred' + plot_comments + '.png', dpi=300) 

    return Gpred, sigma_Gpred, normres

def A_bolotest(lw, layer='wiring'):   # area of bolotest bolos for four legs
    if layer=='wiring':
        dsub = .420; dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
    elif layer=='W1':
        dsub = .420; dW1 = .160; dI1 = 0; dW2 = 0; dI2 = 0   # film thicknesses, um
    elif layer=='bare':
        dsub = .420; dW1 = 0; dI1 = 0; dW2 = 0; dI2 = 0   # film thicknesses, um
    w1w, w2w = wlw(lw, fab='bolotest', layer=layer)
    return (lw*dsub + w1w*dW1 + w2w*dW2 + lw*dI1 +lw*dI2)*4   # area of four legs 
    

def plot_GandTFNEP(fit, lwidths, Tc=0.170, ll=220, dsub=0.420, plot_Gerr=True, plot_NEPerr=False, plot_vAoL=False, save_fig=False, plot_dir='./', Glims=[], NEPlims=[], plotG=True):

    # plots GTES and thermal fluctuation noise equivalent power predictions from the alpha model vs leg width
    # G values in pW/K, leg dimensions in um, temperature in K
    # can turn off/on G_TES errors and NEP errors
    # plots GTES and TFNEP vs leg area/length
    
    # predict G assuming four legs of one type
    G_full, Gerr_full = Gfrommodel(fit, dsub, lwidths, ll, layer='total', fab='bolotest')   #G(S + microstrip), four legs
    # G_U, Gerr_U = Gfrommodel(fit, dsub, lwidths, ll, layer='U', fab='bolotest')/2 + Gfrommodel(fit, dsub, lwidths, ll, layer='total', fab='bolotest')/2
    # G_W1, Gerr_W1 = Gfrommodel(fit, dsub, lwidths, ll, layer='W1', fab='bolotest')/2 + Gfrommodel(fit, dsub, lwidths, ll, layer='total', fab='bolotest')/2
    # G_Nb200 = G_U+G_W1; Gerr_Nb200 = Gerr_U+Gerr_W1
    # G_bare, Gerr_bare = Gfrommodel(fit, .340, lwidths, ll, layer='U', fab='bolotest') + Gfrommodel(fit, dsub, lwidths, ll, layer='total', fab='bolotest')/2   # bare substrate is thinner from etching steps
    # G_U, Gerr_U = Gfrommodel(fit, dsub, lwidths, ll, layer='U', fab='bolotest') 
    G_W1, Gerr_W1 = Gfrommodel(fit, dsub, lwidths, ll, layer='U', fab='bolotest') + Gfrommodel(fit, dsub, lwidths, ll, layer='W1', fab='bolotest')   # G(S+W1), four legs
    G_S, Gerr_S = Gfrommodel(fit, dsub, lwidths, ll, layer='U', fab='bolotest')  # assuming G(TES) = G(substrate) on four legs


    # predicted G vs substrate width
    if plotG:

        NEP_full = TFNEP(Tc, G_full*1E-12)*1E18; NEPerr_full = sigma_NEP(Tc, G_full*1E-12, Gerr_full*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
        NEP_W1 = TFNEP(Tc, G_W1*1E-12)*1E18; NEPerr_W1 = sigma_NEP(Tc, G_W1*1E-12, Gerr_W1*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
        NEP_S = TFNEP(Tc, G_S*1E-12)*1E18; NEPerr_S = sigma_NEP(Tc, G_S*1E-12, Gerr_S*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK

        fig, ax1 = plt.subplots() 
        ax1.plot(lwidths, G_full, color='rebeccapurple', label='G$_\\text{TES}$, Microstrip', alpha=0.8) 
        ax1.plot(lwidths, G_W1, color='green', label='G$_\\text{TES}$, 200nm Nb', alpha=0.8) 
        ax1.plot(lwidths, G_S, color='royalblue', label='G$_\\text{TES}$, Bare S', alpha=0.8)
        if plot_Gerr:
            plt.fill_between(lwidths, G_full-Gerr_full, G_full+Gerr_full, facecolor="mediumpurple", alpha=0.2)   # error
            plt.fill_between(lwidths, G_W1-Gerr_W1, G_W1+Gerr_W1, facecolor="limegreen", alpha=0.2)   # error
            plt.fill_between(lwidths, G_S-Gerr_S, G_S+Gerr_S, facecolor="cornflowerblue", alpha=0.2)   # error
        ax1.set_xlabel('Substrate Width [$\mu$m]') 
        ax1.set_ylabel('G$_\\text{TES}$(170mK) [pW/K]') 
        if len(Glims)>0: ax1.set_ylim(ymin=Glims[0], ymax=Glims[1])   # user specified G y-axis limits

        # TFNEP vs substrate width
        ax2 = ax1.twinx() 
        ax2.plot(lwidths, NEP_full, '--', color='rebeccapurple', label='NEP')   # this varies as G^1/2
        ax2.plot(lwidths, NEP_W1, '--', color='green', label='NEP')   # this varies as G^1/2
        ax2.plot(lwidths, NEP_S, '--', color='royalblue', label='NEP')   # this varies as G^1/2
        if plot_NEPerr: 
            plt.fill_between(lwidths, NEP_full-NEPerr_full, NEP_full+NEPerr_full, facecolor="rebeccapurple", alpha=0.2)   # error
            plt.fill_between(lwidths, NEP_W1-NEPerr_W1, NEP_W1+NEPerr_W1, facecolor="green", alpha=0.2)   # error
            plt.fill_between(lwidths, NEP_S-NEPerr_S, NEP_S+NEPerr_S, facecolor="royalblue", alpha=0.2)   # error
        if len(NEPlims)>0: ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1])   # user specified TFNEP y-axis limits
        ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')     

        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='12', ncol=2)
        plt.tight_layout()
        if save_fig: plt.savefig(plot_dir + 'design_implications.png', dpi=300)


        if plot_vAoL:   # predicted G and NEP vs leg A/L
            A_full = A_bolotest(lwidths, layer='wiring'); A_W1 = A_bolotest(lwidths, layer='W1'); A_S = A_bolotest(lwidths, layer='bare')   # areas of different film stacks

            fig, ax1 = plt.subplots()   # TES thermal conductance
            ax1.plot(A_full/ll, G_full, color='rebeccapurple', label='G$_\\text{TES}$, Microstrip', alpha=0.8) 
            ax1.plot(A_W1/ll, G_W1, color='green', label='G$_\\text{TES}$, 200nm Nb', alpha=0.8) 
            ax1.plot(A_S/ll, G_S, color='royalblue', label='G$_\\text{TES}$, Bare', alpha=0.8)
            if plot_Gerr:
                plt.fill_between(A_full/ll, G_full-Gerr_full, G_full+Gerr_full, facecolor="mediumpurple", alpha=0.2)   # error
                plt.fill_between(A_W1/ll, G_W1-Gerr_W1, G_W1+Gerr_W1, facecolor="limegreen", alpha=0.2)   # error
                plt.fill_between(A_S/ll, G_S-Gerr_S, G_S+Gerr_S, facecolor="cornflowerblue", alpha=0.2)   # error
            ax1.set_xlabel('TES Leg A/L [$\mu$m]') 
            ax1.set_ylabel('G$_\\text{TES}$ [pW/K]') 
            ax1.set_ylim(ymin=Glims[0], ymax=Glims[1]) 

            ax2 = ax1.twinx()   # thermal fluctuation NEP
            ax2.plot(A_full/ll, NEP_full, '--', color='rebeccapurple', label='NEP')   # this varies as G^1/2
            ax2.plot(A_W1/ll, NEP_W1, '--', color='green', label='NEP')   # this varies as G^1/2
            ax2.plot(A_S/ll, NEP_S, '--', color='royalblue', label='NEP')   # this varies as G^1/2
            if plot_NEPerr:
                plt.fill_between(A_full/ll, NEP_full-NEPerr_full, NEP_full+NEPerr_full, facecolor="rebeccapurple", alpha=0.2)   # error
                plt.fill_between(A_W1/ll, NEP_W1-NEPerr_W1, NEP_W1+NEPerr_W1, facecolor="green", alpha=0.2)   # error
                plt.fill_between(A_S/ll, NEP_S-NEPerr_S, NEP_S+NEPerr_S, facecolor="royalblue", alpha=0.2)   # error
            ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1]) 
            ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')     
            ax2.set_xlim(np.nanmin(A_full/ll)-0.1*np.nanmax(A_full/ll), np.nanmax(A_full/ll)*1.1)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='12', ncol=2)
            plt.tight_layout()
        if save_fig: plt.savefig(plot_dir + 'design_implications_AL.png', dpi=300) 

    else:   # plot NEP^2
        
        NEPfull_sq = (TFNEP(Tc, G_full*1E-12))**2*1E36; NEPerrfull_sq = sigmaNEP_sq(Tc, G_full*1E-12, Gerr_full*1E-12)*1E36   # aW^2 / Hz; Kenyan 2006 measured (1E-17)^2 for a TES with comparable G at 170 mK
        NEPW1_sq = (TFNEP(Tc, G_W1*1E-12))**2*1E36; NEPerrW1_sq = sigmaNEP_sq(Tc, G_W1*1E-12, Gerr_W1*1E-12)*1E36   # aW^2 / Hz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
        NEPS_sq = (TFNEP(Tc, G_S*1E-12)**2)*1E36; NEPerrS_sq = sigmaNEP_sq(Tc, G_S*1E-12, Gerr_S*1E-12)*1E36   # aW^2 / Hz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK

        # Compare to SPIDER NEP?
        Psat_sp = 3E-12   # W; SPIDER 280 GHz target Psat; Hubmayr 2019
        # NEP_spider = 17E-18   # W/rt(Hz), detector NEP of 90 and 150 GHz SPIDER bolos, Mauskopf et al 2018
        # NEP_FIR = 1E-20   # W/rt(Hz), NEP necessary to do background limited FIR spectroscopy, Kenyon et al 2006
        NEP_Psat = TFNEP(0.170, Psat_sp/(0.170-0.100), Tb=0.100)*1E18   # NEP for target SPIDER Psat at 100 mK; aW/rt(Hz); G(Tc)~Psat/(Tc-Tb)

        plt.figure(figsize=(5.5,5))
        plt.plot(lwidths, NEPfull_sq, color='rebeccapurple', label='S+Micro')  
        plt.plot(lwidths, NEPW1_sq, color='green', label='S+W1')   
        plt.plot(lwidths, NEPS_sq, color='royalblue', label='S')  
        # plt.axhline((NEP_FIR)**2*1E36, label='FIR Spec', color='black')
        # plt.fill_between([0,16], 0, (NEP_FIR)**2*1E36, facecolor="black", alpha=0.1)   # FIR spectroscopy NEP limits
        # plt.axhline((NEP_sp)**2, label='SPIDER 280 GHz', color='black')
        # plt.fill_between([0,16], 0, (NEP_sp)**2, facecolor="black", alpha=0.1)   # FIR spectroscopy NEP limits
        # plt.axhline((NEP_Psat)**2, label='SPIDER Loading', color='black')
        # plt.fill_between([0,16], 0, (NEP_Psat)**2, facecolor="black", alpha=0.1)   # FIR spectroscopy NEP limits
                
        if plot_NEPerr: 
            plt.fill_between(lwidths, NEPfull_sq-NEPerrfull_sq, NEPfull_sq+NEPerrfull_sq, facecolor="rebeccapurple", alpha=0.3)   # error
            plt.fill_between(lwidths, NEPW1_sq-NEPerrW1_sq, NEPW1_sq+NEPerrW1_sq, facecolor="green", alpha=0.3)   # error
            plt.fill_between(lwidths, NEPS_sq-NEPerrS_sq, NEPS_sq+NEPerrS_sq, facecolor="royalblue", alpha=0.3)   # error
        if len(NEPlims)>0: plt.ylim(ymin=NEPlims[0], ymax=NEPlims[1])   # user specified TFNEP y-axis limits
        plt.xlim(min(lwidths)-max(lwidths)*0.02, max(lwidths)*1.02)
        plt.ylim(0,35)
        plt.ylabel('NEP$_\\text{TF}^2$ [aW$^2$/Hz]', fontsize='18')     
        plt.xlabel('S Width [$\mu$m]', fontsize='18') 
        plt.legend(loc='upper left', fontsize='16')
        plt.tight_layout()
        if save_fig: plt.savefig(plot_dir + 'design_implications.png', dpi=500) 

        return np.array([NEPfull_sq, NEPerrfull_sq])   # aW^2/Hz for bolo with microstrip on all four legs

### analyze results and compare with literature values
def phonon_wlength(vs, T, domcoeff=2.82):   # returns dominant phonon wavelength in vel units * s (probably um)
    # dominant coefficient takes different values in the literature
    # 2.82 is from Ziman (Bourgeois et al J. Appl. Phys 101, 016104 (2007)), Pohl uses domcoeff = 4.25
    
    return planck*vs/(domcoeff*kB*T)   # [v_s * s]

def f_spec(eta, lambda_ave):
    # calculate probability of specular scattering for a given RMS surface roughness and average phonon wavelength
    # eta = RMS surface roughness 
    # lambda_ave = average phonon wavelength

    q = 2*np.pi / lambda_ave
    return np.exp(-4 * np.pi * eta**2 * q**2)

def trans_coeff(rho1, v1, rho2, v2, d2, T=0.170, eta=0, theta1=0):
    # phonon transmission probability from film 1 to film 2
    # thermal resistance across barrier due to accoustic mismatch
    # valid for low-mid freq phonons
    # from Liang et al 2014
    #
    # rho = density (mass density?); 8.57 g/cm^3 for Nb and 3.17 g/cm^3 for SiN
    # v = sound speed
    # d2 = thickness of film 2
    # eta = surface roughness
    # theta1 = angle of incidence from film 1 to 2
    # T = temperature [K]
    # theta = angle to the normal 

    # rhoSiN = 3.17*1E6   # g/m^3
    # rhoNb = 8.57*1E6   # g/m^3
    # trans_coeff(rhoSiN, vs_SiN, rhoNb, vs_Nb, eta=400E-9, T=0.170, theta1=np.pi/4) for 170 mK with RMS roughness of 10 nm, theta = 45 deg

    # check for total internal reflection
    if (v2>v1) and (theta1>np.arcsin(v1/v2)):   # total internal reflection
        theta2 = np.pi/2
    else:
        theta2 = np.arcsin(v2/v1 * np.sin(theta1))   # related by snell's law

    lambda1 = phonon_wlength(v1, T); lambda2 = phonon_wlength(v2, T)   # mean phonon wavelength
    fspec1 = f_spec(eta, lambda1); fspec2 = f_spec(eta, lambda2)   # probability of specular scattering 
    print('Dominant phonon wavelength in Film 1 = {lambda1} nm'.format(lambda1=round(lambda1*1E9,1)))
    print('Dominant phonon wavelength in Film 2 = {lambda2} nm'.format(lambda2=round(lambda2*1E9,1)))
    print('Probability of specular scattering in Film 1 = {fspec1}'.format(fspec1=round(fspec1,4)))
    print('Probability of specular scattering in Film 2 = {fspec2}'.format(fspec2=round(fspec2,4)))
          
    z1 = rho1*v1; z2 = rho2*v2   # acoustic impedences of films 1 and 2
    a0 = (4*z1*z2*np.cos(theta1)*np.cos(theta2)) / (z1*np.cos(theta1) + z2*np.cos(theta2))**2   # transmission coefficient not considering what bounces back from film 2
    print('a0 = {a0}'.format(a0=a0))
    specterm = fspec2*np.exp(-2*d2 / (np.cos(theta2) * lambda2))
    corrterm = (1 - specterm) / (1 - (1 - a0)*specterm)
    tcoff = a0 * corrterm
    # print('Transmission coefficient = {tcoff}'.format(tcoff=round(tcoff,4)))

    return tcoff

def refl_coeff(rho1, v1, rho2, v2, T=0.170):
    # thermal resistance across barrier due to accoustic mismatch
    # valid for low-mid freq phonons
    # from Liang et al 2014
    #
    # rho = density (mass density?); 8.57 g/cm^3 for Nb and 3.17 g/cm^3 for SiN
    # v = sound speed
    # d2 = thickness of film 2
    # eta = surface roughness
    # theta1 = angle of incidence from film 1 to 2
    # T = temperature [K]
    #
    # refl_coeff(rhoSiN, vs_SiN, rhoNb, vs_Nb) for 170 mK 

    # check for total internal reflection
    # if (v2>v1) and (theta1>np.arcsin(v1/v2)):   # total internal reflection
    #     theta2 = np.pi/2
    #     # fspec = 1   
    # else:
    #     theta2 = np.arcsin(v2/v1 * np.sin(theta1))   # related by snell's law

    lambda1 = phonon_wlength(v1, T); lambda2 = phonon_wlength(v2, T)   # mean phonon wavelength
    # fspec1 = f_spec(eta, lambda1); fspec2 = f_spec(eta, lambda2)   # probability of specular scattering 
    print('Dominant phonon wavelength in Film 1 = {lambda1} nm'.format(lambda1=round(lambda1*1E9,1)))
    print('Dominant phonon wavelength in Film 2 = {lambda2} nm'.format(lambda2=round(lambda2*1E9,1)))
    # print('Probability of specular scattering in Film 1 = {fspec1}'.format(fspec1=round(fspec1,4)))
    # print('Probability of specular scattering in Film 2 = {fspec2}'.format(fspec2=round(fspec2,4)))
          
    z1 = rho1*v1; z2 = rho2*v2   # acoustic impedences of films 1 and 2
    rcoff = (z2-z1)**2 / (z2+z1)**2   # transmission coefficient not considering what bounces back from film 2
    print('Reflection Coefficient = {rcoff}'.format(rcoff=rcoff))

    return rcoff


def trans_coeff_simple(rho1, v1, rho2, v2, T=0.170):
    # simple transmission coefficient from wave mechanics
    # assumes normal incidence

    lambda1 = phonon_wlength(v1, T); lambda2 = phonon_wlength(v2, T)   # mean phonon wavelength
    # fspec1 = f_spec(eta, lambda1); fspec2 = f_spec(eta, lambda2)   # probability of specular scattering 
    print('Dominant phonon wavelength in Film 1 = {lambda1} nm'.format(lambda1=round(lambda1*1E9,1)))
    print('Dominant phonon wavelength in Film 2 = {lambda2} nm'.format(lambda2=round(lambda2*1E9,1)))
    # print('Probability of specular scattering in Film 1 = {fspec1}'.format(fspec1=round(fspec1,4)))
    # print('Probability of specular scattering in Film 2 = {fspec2}'.format(fspec2=round(fspec2,4)))
          
    z1 = rho1*v1; z2 = rho2*v2   # acoustic impedences of films 1 and 2
    # a0 = (4*z1*z2*np.cos(theta1)*np.cos(theta2)) / (z1*np.cos(theta1) + z2*np.cos(theta2))**2   # transmission coefficient not considering what bounces back from film 2
    tcoff = 4*z2*z1 / (z2+z1)**2   # transmission coefficient not considering what bounces back from film 2
    print('Transmission Coefficient = {tcoff}'.format(tcoff=tcoff))
    # specterm = fspec2*np.exp(-2*d2 / (np.cos(theta2) * lambda2))
    # corrterm = (1 - specterm) / (1 - (1 - a0)*specterm)
    # tcoff = a0 * corrterm
    # print('Transmission coefficient = {tcoff}'.format(tcoff=round(tcoff,4)))

    return tcoff


def plot_tcoeff():
    theta_test = np.linspace(0, np.pi/2)
    Tcoeffs = trans_coeff(rhoSiN, vs_SiN, rhoNb, vs_Nb, 400E-9, T=0.170, eta=10E-9, theta1=theta_test)
    Tave = np.sum(Tcoeffs)/len(Tcoeffs)

    plt.figure()
    plt.plot(theta_test*180/np.pi, Tcoeffs)
    plt.xlabel('Incident Angle [deg]'); plt.ylabel('Transmission Coefficient')
    plt.title('SiN to Nb, dNb = 400 nm, roughness = 10 nm')
    plt.annotate('$\\langle T \\rangle = {Tave}}$'.format(Tave=round(Tave,3)), (12, 0.4), fontsize='16')     

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
    # return x/2 * np.arcsinh(x) + 1/6*((1+x**2)**(1/2) * (x**2-2) + (2-x**3))
    return x/2 * np.arcsinh(x) + 1/6 * (np.sqrt(1+x**2) * (x**2-2) + (2-x**3))

def xi_Cas(w, d, L):
    # numerical factor in the Casimir (diffusive) limit
    n = w/d   # aspect ratio
    return 3/(2*L) * (3*d**3)/(2*w) * ((n)**3 * I_mfp(1/n) + I_mfp(n))
 
def G_radtheory(d, w, vst, vsl, T=0.170, dim='3D', lim='diff'):
    # Theoretical G in diffusive or ballistic limit, selectable dimensionality
    # Holmes Thesis Section 4.5.1

    if lim=='diff':   # diffusive (Casimir) limit
        xi = xi_Cas(w, d, L)
    elif lim=='ball':   # ballistic limit
        xi = 1

    if dim=='3D':
        vsq_sum = 2/(vst**2) + 1/(vsl**2)   # sum with transverse and longitudinal sound speeds
        sigma = np.pi**5 * kB**4 /(15*planck**3) * vsq_sum
        G0 = 4*sigma*d*w*T**3
    elif dim=='2D':
        P = L + w   # perimeter of 2D surface
        RZ = zeta(3)   # Reimann Zeta Function (returns scalar)
        v_sum = 2/(vst) + 1/(vsl)
        G0 = 3*P*RZ*kB**3 / (6*planck**2) * v_sum * T**2
    elif dim=='1D':
        G0 = 2*np.pi**2 * kB**2 / planck * T
    
    return xi * G0


def firstterm(n, J):
    subJ = 1E-5 if J==0 else J
    return n**3 * ( (J+1)**3 * I_mfp( 1/(n*(J+1)) ) - J**3 * I_mfp( 1/(n*subJ) ) )

def secondterm(n, J):
    kdelta = 1 if J == 0 else 0
    return 1/2 * (2-kdelta) * ( I_mfp(n*(J+1)) - 2*I_mfp(n*J) +  I_mfp(n*(J-1)) )

def sumfunc_J(n, f, J):  
    if J==0:   
        kdelta = 1  # kroniker delta function
        Jsub = 1E-10   # handle divide by 0 error when J = 0; first and second terms converge to a scalar value for J < 1E-4
    elif J > 0:
        kdelta = 0
        Jsub = J
    else:
        print('J cannot be < 0. Returning NaN')
        return np.nan
    
    # how to handle division by zero?
    firstterm = n**3 * ( (J+1)**3 * I_mfp( 1/(n*(J+1)) ) - J**3 * I_mfp( 1/(n*Jsub) ) )
    secondterm = 1/2 * (2-kdelta) * ( I_mfp(n*(J+1)) - 2*I_mfp(n*J) +  I_mfp(n*(J-1)) )
    
    return f*(1-f)**J * (firstterm + secondterm)   # initial interpretation of Wybourne
    # return f**J*(1-f) * (firstterm + secondterm)
    # return f*(1-f) * (firstterm + secondterm)


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

def monopower(x, m, t, b):
    return m * x**t + b  

def fit_power(x, y, p0, sigma=[], absolute_sigma=True):
    # returns power explicitly 
    # idk if this is lazy or actually extra work

    # check for nans
    if np.isnan(np.sum(y)):
        print('Ignoring NaNs while fitting power law')
    sinds = np.where(~np.isnan(y))[0]   # scalar indices

    params, pcov = curve_fit(monopower, x[sinds], y[sinds], p0, sigma=sigma[sinds], absolute_sigma=absolute_sigma)
    perr = np.sqrt(np.diag(pcov)); t = params[1]; sigma_t = perr[1]
    return params, t, sigma_t

def f_BE(f, T):   
    # computes Bose Einstein distribution function for a given frequency and temperature
    # f in Hz, T in K
    # omega = 2pi * f

    return 1 / (np.exp(planck*f / (kB*T)) - 1)
