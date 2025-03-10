import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pkl
import csv
import copy
from scipy.optimize import minimize,  curve_fit, fsolve
from scipy.special import zeta
from collections import OrderedDict
from numpy.random import normal
from datetime import datetime
import pdb

### constants
kB     = 1.3806503E-23   # Boltzmann constant, [J/K]
NA     = 6.022E23   # Avogadro's number, number of particles in one mole
hbar   = 1.055E-34   # reduced Plancks constant = h / 2pi, [J s]
planck = hbar*2*np.pi   # planck's constant, [J s]

### properties of SiNx
vs_SiN = 6986   # [m/s] average sound speed in SiNx; Wang et al
# vs_SiN = 7148   # average taking sum 1/v; for debugging
vt_SiN = 6.2E3; vl_SiN = 10.3E3   # [m/s] transverse and longitudinal sound speeds in SiN
# # TD_Si = 645   # K, silicon, low temp limit

# vs_SiN = 4130   # [m/s] average sound speed in SiNx from ZPH 2006
# vt_SiN = 3670; vl_SiN = 7640   # [m/s] from ZPH 2006
# TD_SiN = 487   # K, silicon, low temp limit from Zink Pietri and Hellman 2006
# volmol_SiN = 21/(2.9*1e6)   # m^3 per mole for SiN fro ZPH06; 2.9 g/cm^3; 21 g/mol

# vs_SiN = 7600   # [m/s] average sound speed in SiNx from Zink and Hellman 2003
# TD_SiN = 985   # K, silicon, low temp limit from Zink and Hellman 2003
volmol_SiN = 40.78*1E-6   # m^3 per mole for SiN
rhoSiN = 3.17*1E6   # [g/m^3] mass density

### properties of Nb
# vs_Nb = 3480   # [m/s] average sound speed in Nb
vs_Nb = 3084  # [m/s] average sound speed in Nb
vt_Nb = 2.092E3; vl_Nb = 5.068E3   # [m/s] transverse and longitudinal sound speeds in Nb
# vl_Nb = np.sqrt(25.422/8.620)   # sqrt(C11/rho)
# vt_Nb = np.sqrt(3.090/8.620)   # sqrt(C44/rho)
rhoNb = 8.57*1E6   # [g/m^3] mass density
TD_Nb = 276   # K, Nb Debye temperature
Tc_Nb = 9.2   # Nb, K
vF_Nb = 1.37E6   # Nb Fermi velocity (electron velocity), m/s
volmol_Nb = 10.84*1E-6   # m^3 per mole for Nb

# thermal conductance quantum @ 170 mK
G0 = np.pi**2*kB**2*0.170/(3*planck)*1E12   # quantum G at 170 mK; pW/K

mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

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

### thermal fluctuation noise equivalent power
def TFNEP(Tc, G, Tb=0.100, gamma=1):   # calculate thermal fluctuation noise equivalent power as a function of G and T_TES
    # return np.sqrt(4*kB*G*FLink(Tb, Tc))*Tc
    return np.sqrt(4*kB*G*gamma)*Tc   # FLink only analytical for Nb

def sigma_NEP(Tc, G, sigma_G, Tb=0.100, gamma=1):   # error on NEP estimation
    # sigma_nepsq = kB*FLink(Tb, Tc)/G * Tc**2 * sigma_G**2
    sigma_nepsq = kB*gamma/G * Tc**2 * sigma_G**2
    return np.sqrt(sigma_nepsq)

def sigmaNEP_sq(Tc, G, sigma_G, Tb=0.100, gamma=1):   # error on NEP^2 = error on S = sqrt(4 kB FL) * T * sigma_G
    # sigma_nep = np.sqrt(kB/G*T**2 * sigma_G**2)
    # return 2*sigma_nep*np.sqrt(4*kB*G*T**2)
    # return 4*kB*FLink(Tb, Tc) * Tc**2 * sigma_G
    return 4*kB*gamma * Tc**2 * sigma_G

def GandPsatfromNEP(NEP, Tc, Tb, gamma=1):   # calculate G(Tc) and Psat(Tc) given thermal fluctuation NEP, Tc in K, and Tbath in K
    G_Tc = (NEP/Tc)**2 / (4*kB*gamma)   # W/K
    P_Tc = G_Tc*(Tc-Tb)   # W
    return np.array([G_Tc, P_Tc])

def Psat_fromG(G, n, Tc, Tb):
    return Tc*G/n * (1-(Tb/Tc)**n)

def G_fromPsat(P, n, Tc, Tb):
    return P*n/Tc / (1 - (Tb/Tc)**n)

### G from alpha model

def wlw(lw, fab='bolotest', maxw1w=5, maxw2w=3, minw2w=1):
    # calculate W layer widths for bolotest or legacy data
    # INPUT lw in um
    # W2 width = W1 width + 2 um = leg width - 4um, and any W layer !< 1um (W2 !< 3 um for microstrip, !< 1um for single W1 layer)
    if fab=='legacy':
        w1w=8*np.ones_like(lw); w2w=5*np.ones_like(lw)
        if np.ndim(lw)!=0: # accommodate leg widths smaller than defaSlt W1 width
            smallw = np.where(lw<=8)[0]
            w1w[smallw]=5; w2w[smallw]=3
        elif lw<=8:
            w1w=5; w2w=3

    elif fab=='bolotest':
        # leg widths >= 7 um: W1 = 5 um and W2 = 3 um
        # leg widths 7 um and >= 5 um: W1 = lw - 2 um and W2 = lw - 4 um
        # leg widths < 5 um: W1 and W2 = NaN
        w1w=maxw1w*np.ones_like(lw); w2w=maxw2w*np.ones_like(lw)   # W layer widths for leg widths => 7 um
        # maxw1w = 5; minw1w = 3 #if layer=='W1' else 2   # single W layer can span 1-5 um
        # maxw1w = 5   # for lw < max w1, w1w = lw
        # minw2w = 1; maxw2w = 3   # single W2 layer can span 1-3 um
        w2w0 = lw-2   # naive W width estimate for lw < w1 min width
        if np.ndim(lw)!=0:
            # w1w0 = lw-2   # naive W1 width estimate = lw - 2 um
            # naninds = np.where(w1w0<minw1w)[0]   # W1 width !< min width, return nans
            # scaleinds = np.where((minw1w<=w1w0)&(w1w0<maxw1w))[0]   # layer is < normalized width, scale
            # if len(naninds)!=0: w1w[naninds] = np.nan; w2w[naninds] = np.nan
            # if len(scaleinds)!=0: w1w[scaleinds] = w1w0[scaleinds]; w2w[scaleinds] = w1w0[scaleinds]-2   # um
            naninds   = np.where(w2w0<minw2w)[0]   # W2 width !< min width, return nans
            scaleinds = np.where((w2w0>=minw2w)&(lw<maxw1w))[0]   # layer is < normalized width, scale
            if len(naninds)!=0:   w1w[naninds] = np.nan;          w2w[naninds] = np.nan
            if len(scaleinds)!=0: w1w[scaleinds] = lw[scaleinds]; w2w[scaleinds] = w2w0[scaleinds]   # um
        elif w2w0 < minw2w:   # handle single leg widths; no W2 < min W2 width
            w1w = np.nan; w2w = np.nan
        elif w2w0 < maxw2w:   # scale W2 and W1 with leg width
            w2w = w2w0; w1w = w2w+2   # um

    return w1w, w2w

def G_layer(fit, d, layer='S', model='Three-Layer', dS0=0.400):
    # fit = [fit parameters, fit errors (if calc_sigma=True)], thickness d is in um
    # RETURNS prediction (and error if 'fit' is 2D)

    scalar_d = False
    if np.isscalar(d) or not d.shape:   # handle thickness scalars and arrays
        scalar_d = True
        numdevs = 1   # number of devices = number of thicknesses passed
        d = d
        if d==0:   # handle potential divide by 0 error
            return 0
    else:
        numdevs = len(d)
        d = np.array(d)
        d[d==0] = 1E-12  # handle potential divide by 0 error
        # if d[0]==0:
        #     return np.zeros(len(d))   # handle potential divide by 0 error

    if model=='Three-Layer':   # treat substrate and insulating nitride as separate layers
        if   layer=='S': linds=np.array([0,3]); d0=dS0   # substrate layer parameter indexes and defaSlt thickness in um
        elif layer=='W': linds=np.array([1,4]); d0=.400   # Nb layer parameter indexes and defaSlt thickness in um
        elif layer=='I': linds=np.array([2,5]); d0=.400   # insulating layer parameter indexes and defaSlt thickness in um
    if model=='Four-Layer':   # treat substrate bi-layer as separate layers
        if   layer=='SiO': linds=np.array([0, 0]); d0=.120   # substrate layer parameter indexes and defaSlt thickness in um
        if   layer=='S': linds=np.array([1,4]); d0=dS0   # substrate layer parameter indexes and defaSlt thickness in um
        elif layer=='W': linds=np.array([2,5]); d0=.400   # Nb layer parameter indexes and defaSlt thickness in um
        elif layer=='I': linds=np.array([3,6]); d0=.400   # insulating layer parameter indexes and defaSlt thickness in um
    elif model=='Two-Layer':   # treat substrate and insulating nitride as the same layer
        if   layer=='S': linds=np.array([0,2]); d0=dS0   # nitride layer parameter indexes and defaSlt thickness in um
        elif layer=='W': linds=np.array([1,3]); d0=.400   # Nb layer parameter indexes and defaSlt thickness in um
        elif layer=='I': print('Only considering S and W layers in two-layer model.')   # insulating layer parameter indexes and defaSlt thickness in um

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
            # cols = np.array([G0s * (dd/d0)**(alphas+1) for dd in d])   # each row corresponds to results for one value of d = each column of desired output ... what did i mean by this???
            # Glayer = cols.T   # value for each set of fit parameters (rows) x each thickness d (columns)
            Glayer = np.stack(np.array([G0s * (dd/d0)**(alphas+1) for dd in d]), axis=1)

    return Glayer

def ascale(ll, La):
    return 1/(1+ll/La)

def acoust_factor(bolo):
    # use acoustic length scaling
    La   = bolo['geometry']['La']
    legl = bolo['geometry']['ll']

    a_factor    = ascale(legl, La) / ascale(220, La)
    return a_factor

def lw_regions(bolo, an_opts):
    # calculate widths of leg regions for each layer stack

    stack_I      = an_opts.get('stack_I')
    stack_N      = an_opts.get('stack_N')
    tall_Istacks = an_opts.get('tall_Istacks')

    w1w = bolo['geometry'].get('w1w'); w2w = bolo['geometry'].get('w2w')
    lw  = bolo['geometry'].get('lw');  ll  = bolo['geometry'].get('ll')

    w_tiss = bolo['geometry'].get('w_tiss')   # widths of taller I stack regions

    [tisw_2s, tisw_2, tisw_1s, tisw_1] = w_tiss if tall_Istacks else [0, 0, 0, 0]

    if np.isscalar(lw):
        # sloped regions occupy edge of W layer
        # w2w = (w2 not sloped) + (w2 sloped)

        # W2 region layer widths
        w2w_ns  = (w2w-tisw_2s) if lw > (w2w-tisw_2s) else lw   # width of W2 region before slope
        w2w_s   = tisw_2s if lw>w2w else np.nanmax(lw-w2w_ns, 0)   # width of W2 sloped region
        w2w_e   = tisw_2 if lw>(w2w+tisw_2) else np.nanmax(lw-w2w, 0)   # width of taller I layer(s) at edge of W2

        # W1 region layer widths
        w1w_ns = (w1w-tisw_1s) if lw > (w1w-tisw_1s) else lw
        w1w_s  = tisw_1s if lw>w1w else np.nanmax(lw-w1w_ns, 0)
        w1w_e = tisw_1 if lw>(w1w+tisw_1) else np.nanmax(lw-w1w, 0)   # width of taller I layer(s) at edge of W2

        # nominal i stacks beyond tall ones after W2
        w_istack_b  = (lw-w1w-tisw_1) if lw>(w1w+tisw_1) else 0

    else:
        #        max value at lw > max value;    width between 0 and max;                            width before region begins = 0
        w2w_ns = (w2w-tisw_2s)*np.ones(len(lw)); w2w_ns[lw<(w2w-tisw_2s)] = lw[lw<(w2w-tisw_2s)]
        w2w_s  = tisw_2s*np.ones(len(lw));       w2w_s[lw<w2w]            = (lw-w2w_ns)[lw<w2w];       w2w_s[lw<(w2w-tisw_2s)] = 0   # if lw>w2w else np.nanmax(lw-w2w_ns, 0)
        w2w_e  = tisw_2*np.ones(len(lw));        w2w_e[lw<(w2w+tisw_2)]   = (lw-w2w)[lw<(w2w+tisw_2)]; w2w_s[lw<(w2w)] = 0           # if lw>w2w+tisw_2 else np.nanmax(lw-w2w, 0)

        w1w_ns = (w1w-tisw_1s)*np.ones(len(lw)); w1w_ns[lw<(w1w-tisw_1s)] = lw[lw<(w1w-tisw_1s)]
        w1w_s  = tisw_1s*np.ones(len(lw));       w1w_s[lw<w1w]            = (lw-w1w_ns)[lw<w1w];       w1w_s[lw<(w1w-tisw_1s)] = 0   # if lw>w1w else np.nanmax(lw-w1w_ns, 0)
        w1w_e  = tisw_1*np.ones(len(lw));        w1w_e[lw<(w1w+tisw_1)]   = (lw-w1w)[lw<(w1w+tisw_1)]; w1w_s[lw<(w1w)] = 0 # if lw>w1w+tisw_1 else np.nanmax(lw-w1w, 0)

        # nominal i stacks beyond tall ones after W1
        w_istack_b  = lw-w1w-tisw_1; w_istack_b[lw<(w1w+tisw_1)] = 0

    w2w_tot = w2w_ns + w2w_s   # total width of W2
    w1w_tot = w1w_ns + w1w_s   # total width of W1

    if stack_I:   # i stacks begin after W2
        w1_istack = w1w_ns - (w2w_ns + w2w_s + w2w_e)
        w_istack = w1_istack + w_istack_b   # total width of nominal I stacks
        w_sstack = 0   # total width of nominal S-I1-I2 stacks
    elif stack_N:
        w1_istack = w1w_ns - (w2w_ns + w2w_s + w2w_e)
        w_istack = w1_istack  # total width of nominal I1-I2 stacks
        w_sstack = w_istack_b   # total width of nominal S-I1-I2 stacks
    else:   # I stacks begin after W1
        w_istack = w_istack_b   # total width of nominal I stacks
        w_sstack = 0   # total width of nominal S-I1-I2 stacks

    return lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, w_istack, w_sstack

def G_leg(fit, an_opts, bolo, dS, dW1, dI1, dW2, dI2, include_S, include_W, include_I,
          supG_width=0., legA=False, legB=False, legC=False, legD=False, legE=False, legF=False, legG=False):

    beta         = 1.   # power law exponent for width scaling
    model        = an_opts.get('model')
    stack_I      = an_opts.get('stack_I')
    stack_N      = an_opts.get('stack_N')
    supG         = an_opts.get('supG', 0.0)

    # w1w = bolo['geometry'].get('w1w'); w2w = bolo['geometry'].get('w2w')
    # lw  = bolo['geometry'].get('lw');  ll  = bolo['geometry'].get('ll')

    # scale with L and d in diffuse / ballistic transition
    if bolo['geometry'].get('acoustic_Lscale'):   # use acoustic scaling
        a_factor = acoust_factor(bolo)
    else:   # use power law length scaling
        pLscale  = copy.copy(bolo['geometry']['pLscale'])
        ll       = bolo['geometry'].get('ll')
        a_factor = (220/ll)**pLscale

    ### geometry nuances, trimming I1 on legs A and C and handling thicker I layers near W layer edges
    I1trim   = bolo['geometry'].get('I1trim', 0)
    # w_tiss    = bolo['geometry'].get('w_tiss')
    # d_tiss    = bolo['geometry'].get('d_tiss')   # extra dI1 in region next to W2 on Legs A and C
    # d_tiss    = bolo['geometry'].get('d_tiss')   # extra dI1 in region next to W2 on Legs A and C

    [deltad_AW2, deltad_AW1, deltad_CW2, deltad_DW1, deltad_EW1] = bolo['geometry']['d_tiss'] if 'd_tiss' in bolo['geometry'] else [0, 0, 0, 0, 0]
    lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, w_istack, w_sstack = lw_regions(bolo, an_opts)

    # I1 is trimmed outside of W2 width on legs A and C
    dI1I2 = dI1 + dI2 - I1trim if (legA or legC) else dI1 + dI2

    # hard code these thickness changes in for now
    if legC:
        # dI1 = dI1 + 0.052
        dI2 = dI2 + 0.034

    if legD:
        dS  = dS  - 0.021
        dW1 = dW1 + 0.009

    if model=='Two-Layer':   # treat S and I layers the same

        if legA:   # S-W1-I1-W2-I2
            # I1-I2 stack beyond W2 width, SiNx-I1-I2 stack beyond W1 width

            G_W1   = G_layer(fit, dW1, layer='W', model=model)     * (w1w/5)      * a_factor * include_W
            G_W2   = G_layer(fit, dW2, layer='W', model=model)     * (w2w_tot/5)      * a_factor * include_W
            G_W    = G_W1 + G_W2

            # stack nitrides and oxides
            G_Sub   = G_layer(fit, dS, layer='S', model=model)   * (w1w/5)      * a_factor * include_S   # 5 um
            G_I1    = G_layer(fit, dI1, layer='S', model=model)     * (w2w_tot/5)      * a_factor * include_S   # 3 um
            G_I2    = G_layer(fit, dI2, layer='S', model=model)     * (w2w_tot/5)      * a_factor * include_S   # 3 um
            G_I1I2  = G_layer(fit, dI1I2, layer='S', model=model) * ((w1w-w2w)/5) * a_factor * include_S   # 3 to 5 um
            G_SI1I2 = G_layer(fit, dS+dI1I2, layer='S', model=model) * ((lw-w1w)/5) * a_factor * include_S   # rest of leg
            G_S     = G_Sub + G_I1 + G_I2 + G_I1I2 + G_SI1I2

        elif legB:   # S-W1-I1-W2
            # I1 is trimmed to W2 width, S and I1 never stack, no I2

            G_W1   = G_layer(fit, dW1, layer='W', model=model) * (w1w/5)          * a_factor * include_W
            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5)          * a_factor * include_W
            G_W    = G_W1 + G_W2

            # stack nitrides
            G_Sub  = G_layer(fit, dS, layer='S', model=model) * (lw/5)       * a_factor * include_S   # whole leg
            G_I1   = G_layer(fit, dI1, layer='S', model=model)   * (w2w_tot/5)      * a_factor * include_S   # I1 is trimmed to W2 width, SiNx and I1 never stack
            G_S    = G_Sub + G_I1

        elif legC:   # S-I1-W2-I2
            # SiNx-I1 stack to W2 width, SiNx-I1-I2 stack beyond W2 width

            G_W2    = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5)          * a_factor * include_W
            G_W     = G_W2

            G_SI1   = G_layer(fit, dS+dI1, layer='S', model=model)     * (w2w_tot/5)      * a_factor * include_S
            G_I2    = G_layer(fit, dI2, layer='S', model=model)           * (w2w_tot/5)      * a_factor * include_S
            G_SI1I2 = G_layer(fit, dS+dI1I2, layer='S', model=model) * ((lw-w2w_tot)/5) * a_factor * include_S
            G_S     = G_SI1 + G_I2 + G_SI1I2

        elif legD:   # S-W1-I1-I2
            # I1-I2 to W1 width, SiNx-I1-I2 stack beyond W1

            G_W1    = G_layer(fit, dW1, layer='W', model=model)     * (w1w/5)      * a_factor * include_W
            G_W     = G_W1

            G_Sub   = G_layer(fit, dS, layer='S', model=model)         * (w1w/5)      * a_factor * include_S
            G_I1I2  = G_layer(fit, dI1I2, layer='S', model=model)       * (w1w/5)      * a_factor * include_S   # I1-I2 stack over whole leg
            G_SI1I2 = G_layer(fit, dS+dI1I2, layer='S', model=model) * ((lw-w1w)/5) * a_factor * include_S
            G_S     = G_Sub + G_I1I2 + G_SI1I2

        elif legE:   # S-W1-W2
            # G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400)     * lw/5      * a_factor * include_S

            G_W1W2 = G_layer(fit, dW1+dW2, layer='W', model=model) * (w2w_tot/5)   * a_factor * include_W   # W1-W2 stack is only 3 um wide
            G_W    = G_W1W2

            G_Sub  = G_layer(fit, dS, layer='S', model=model)   * (lw/5)    * a_factor * include_S
            G_S    = G_Sub

        elif legF:   # S-I1-I2
            # SiNx-I1-I2 layer over whole leg
            # G_S        = G_layer(fit, dOx, layer='S', model=model, dS0=0.400)  * lw/5          * a_factor * include_S

            G_W     = 0

            G_SI1I2 = G_layer(fit, dS+dI1I2, layer='S', model=model) * lw/5 * a_factor * include_S
            G_S     = G_SI1I2

        elif legG:   # S
            # G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400)   * lw/5      * a_factor * include_S   # substrate still treated as separate layer

            G_W    = 0

            G_Sub  = G_layer(fit, dS, layer='S', model=model) * (lw/5)    * a_factor * include_S
            G_S    = G_Sub

        G_I = 0

    elif stack_I:   # allow for I1-I2 stacks, including between 3 and 5 um

        if legA:   # S-W1-I1-W2-I2
            # lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w_istack
            G_S    = G_layer(fit, dS, layer='S', model=model)  * lw/5                * a_factor * include_S   # substrate still treated as separate layer

            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W1   = G_layer(fit, dW1, layer='W', model=model) * (w1w_tot/5) * a_factor * include_W
            G_W    = G_W1 + G_W2

            # I1 and I2 are separate to W2 width; I1-I2 stacked beyond W2 width
            # extra d on slope and edge (becomes I1-I2 stack) of W2
            G_I1       = G_layer(fit, dI1,              layer='I', model=model) * (w2w_tot/5)  * a_factor * include_I
            G_I2_ns    = G_layer(fit, dI2,              layer='I', model=model) * (w2w_ns/5)   * a_factor * include_I
            G_I2_s     = G_layer(fit, dI2  +deltad_AW2, layer='I', model=model) * (w2w_s/5)    * a_factor * include_I
            G_I1I2_w2e = G_layer(fit, dI1I2+deltad_AW2, layer='I', model=model) * (w2w_e/5)    * a_factor * include_I   # taller I stack next to W2

            # extra d on slope and edge of W1
            # G_I1I2_ns  = G_layer(fit, dI1I2,            layer='I', model=model) * (w1w_ns/5)   * a_factor * include_I
            G_I1I2_nom = G_layer(fit, dI1I2,            layer='I', model=model) * (w_istack/5) * a_factor * include_I   # total G of I stacks with nominal thickness, incl. lw beyond W1 edge
            G_I1I2_s   = G_layer(fit, dI1I2+deltad_AW1, layer='I', model=model) * (w1w_s/5)    * a_factor * include_I
            G_I1I2_w1e = G_layer(fit, dI1I2+deltad_AW1, layer='I', model=model) * (w1w_e/5)    * a_factor * include_I

            G_I     = G_I1 + G_I2_ns + G_I2_s + G_I1I2_w2e + G_I1I2_s + G_I1I2_w1e + G_I1I2_nom

        elif legB:   # S-W1-I1-W2

            G_S    = G_layer(fit, dS, layer='S', model=model)  * lw/5        * a_factor * include_S   # substrate still treated as separate layer

            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W1   = G_layer(fit, dW1, layer='W', model=model) * (w1w_tot/5) * a_factor * include_W
            G_W    = G_W1 + G_W2

            # I1 is trimmed to W2 width
            G_I1   = G_layer(fit, dI1, layer='I', model=model) * (w2w_tot/5) * a_factor * include_I
            G_I    = G_I1

        elif legC:   # S-I1-W2-I2
            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5         * a_factor * include_S   # substrate still treated as separate layer

            G_W2   = G_layer(fit, dW2, layer='W', model=model)     * (w2w_tot/5)     * a_factor * include_W
            G_W    = G_W2

            # I1 and I2 are separate to W2 width; I1-I2 stacked beyond W2 width
            # extra d on slope and edge (becomes I1-I2 stack) of W2
            G_I1       = G_layer(fit, dI1,              layer='I', model=model) * (w2w_tot/5) * a_factor * include_I
            G_I2_ns    = G_layer(fit, dI2,              layer='I', model=model) * (w2w_ns/5) * a_factor * include_I
            G_I2_s     = G_layer(fit, dI2+deltad_CW2,   layer='I', model=model) * (w2w_s/5)  * a_factor * include_I
            G_I1I2_w2e = G_layer(fit, dI1I2+deltad_CW2, layer='I', model=model) * (w2w_e/5)  * a_factor * include_I   # taller I stack next to W2

            # nominal I stacks beyond W2 edge
            w_istack_legC = w_istack + w1w_s + w1w_e
            G_I1I2_nom = G_layer(fit, dI1I2,            layer='I', model=model) * (w_istack_legC/5)  * a_factor * include_I

            G_I      = G_I1 + G_I2_ns + G_I2_s + G_I1I2_w2e + G_I1I2_nom

        elif legD:   # S-W1-I1-I2

            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5   * a_factor * include_S   # substrate still treated as separate layer

            G_W1   = G_layer(fit, dW1, layer='W', model=model)     * w1w_tot/5 * a_factor * include_W
            G_W    = G_W1

            # I1-I2 stacked across entire leg, nominal height includes W2 edge and slope
            w_istack_legD = w_istack + w2w_s + w2w_e
            G_I1I2_nom = G_layer(fit, dI1I2,            layer='I', model=model) * (w_istack_legD/5) * a_factor * include_I   # total G of I stacks with nominal thickness, incl. lw beyond W1 edge
            G_I1I2_s   = G_layer(fit, dI1I2+deltad_DW1, layer='I', model=model) * (w1w_s/5)         * a_factor * include_I
            G_I1I2_w1e = G_layer(fit, dI1I2+deltad_DW1, layer='I', model=model) * (w1w_e/5)         * a_factor * include_I

            G_I    = G_I1I2_nom + G_I1I2_s + G_I1I2_w1e

        elif legE:   # S-W1-W2

            # allow for different dS between 3 and 5 um
            G_SW2  = G_layer(fit, dS, layer='S', model=model)            * w2w_tot/5      * a_factor * include_S   # substrate still treated as separate layer
            G_SW1  = G_layer(fit, dS+deltad_EW1, layer='S', model=model) * (lw-w2w_tot)/5 * a_factor * include_S   # substrate still treated as separate layer
            G_S    = G_SW2 + G_SW1
            # G_S    = (G_layer(fit, dS, layer='S', model=model)*3/5 + G_layer(fit, dS+0.018, layer='S', model=model)*(lw-3)/5)    * a_factor * include_S   # substrate still treated as separate layer

            # W1-W2 stack trimmed to W2 width
            G_W1W2 = G_layer(fit, dW1+dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W    = G_W1W2

            G_I    = 0

        elif legF:   # S-I1-I2
            G_S    = G_layer(fit, dS, layer='S', model=model)    * lw/5 * a_factor * include_S   # substrate still treated as separate layer

            G_W    = 0

            G_I1I2 = G_layer(fit, dI1I2, layer='I', model=model) * lw/5 * a_factor * include_I
            G_I    = G_I1I2

        elif legG:   # S
            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5          * a_factor * include_S   # substrate still treated as separate layer

            G_W    = 0

            G_I    = 0

    elif stack_N:   # treat S SiOx as S layer and S SiNx as an I layer, allow for SiNx-I1-I2 stacks

        dOx = bolo['geometry']['dSiOx']*np.ones_like(dS)   # nm - thickness of oxide layer - shared with all film stacks
        dSiNx = dS-dOx

        # handle dsub < d_SiNx
        if np.isscalar(dSiNx):
            if (dSiNx < 0): dOx = dS; dSiNx = 0
        else:
            dOx[dSiNx < 0]   = dS[dSiNx < 0]
            dSiNx[dSiNx < 0] = 0

        if legA:   # S-W1-I1-W2-I2
            # I1-I2 stack beyond W2 width, SiNx-I1-I2 stack beyond W1 width

            G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5 * a_factor * include_S

            G_W1   = G_layer(fit, dW1, layer='W', model=model)   * (w1w_tot/5) * a_factor * include_W
            G_W2   = G_layer(fit, dW2, layer='W', model=model)   * (w2w_tot/5) * a_factor * include_W
            G_W    = G_W1 + G_W2

            # separate nitride layers; I1 and I2 single layers for W2; SiNx separate layer for W1
            G_SiNx     = G_layer(fit, dSiNx, layer='I', model=model)            * (w1w_tot/5) * a_factor * include_I
            G_I1       = G_layer(fit, dI1,   layer='I', model=model)            * (w2w_tot/5) * a_factor * include_I
            G_I2       = (G_layer(fit, dI2,   layer='I', model=model)*w2w_ns/5 + G_layer(fit, dI2+deltad_AW2, layer='I', model=model)*w2w_s/5) * a_factor * include_I
            G_I1I2_w2e = G_layer(fit, dI1I2+deltad_AW2, layer='I', model=model) * (w2w_e/5)   * a_factor * include_I   # taller I stack next to W2

            # stack nitrides; I1 and I2 stacked between W1 and W2; SiNx stacked after W1
            G_I1I2_nom     = G_layer(fit, dI1I2,            layer='I', model=model)       * (w_istack/5) * a_factor * include_I
            G_I1I2_s       = G_layer(fit, dI1I2+deltad_AW1, layer='I', model=model)       * (w1w_s/5)    * a_factor * include_I
            G_SiNxI1I2_w1e = G_layer(fit, dSiNx+dI1I2+deltad_AW1, layer='I', model=model) * (w1w_e/5)    * a_factor * include_I
            G_SiNxI1I2_nom = G_layer(fit, dSiNx+dI1I2, layer='I', model=model)            * (w_sstack/5) * a_factor * include_I   # rest of leg
            G_I    = G_SiNx + G_I1 + G_I2 + G_I1I2_w2e + G_I1I2_nom + G_I1I2_s + G_SiNxI1I2_w1e + G_SiNxI1I2_nom

        elif legB:   # S-W1-I1-W2
            G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5 * a_factor * include_S

            G_W1   = G_layer(fit, dW1, layer='W', model=model) * (w1w_tot/5) * a_factor * include_W
            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W    = G_W1 + G_W2

            # I1 is trimmed to W2, SiNx and I1 never stack
            G_SiNx = G_layer(fit, dSiNx, layer='I', model=model) * (lw/5)      * a_factor * include_I
            G_I1   = G_layer(fit, dI1,   layer='I', model=model) * (w2w_tot/5) * a_factor * include_I
            G_I    = G_SiNx + G_I1

        elif legC:   # S-I1-W2-I2

            G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5       * a_factor * include_S

            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W    = G_W2

            # I2 separate to W2 width; SiNx-I1 stack to W2; SiNx-I1-I2 stack beyond W2 width
            G_SiNxI1   = G_layer(fit, dSiNx+dI1, layer='I', model=model)   * (w2w_tot/5) * a_factor * include_I   # rest of leg
            G_I2       = (G_layer(fit, dI2,   layer='I', model=model)*w2w_ns/5 + G_layer(fit, dI2+deltad_CW2,   layer='I', model=model)*w2w_s/5) * a_factor * include_I
            # G_SiNxI1I2 = G_layer(fit, dSiNx+dI1I2, layer='I', model=model) * ((w_istack + w_sstack)/5) * a_factor * include_I   # rest of leg
            G_SiNxI1I2_w2e = G_layer(fit, dSiNx+dI1I2+deltad_CW2, layer='I', model=model) * (w2w_e/5)  * a_factor * include_I   # taller I stack next to W2

            # nominal I stacks beyond W2 edge
            w_sstack_legC = w_istack + w_sstack + w1w_s + w1w_e
            G_I1I2_nom = G_layer(fit, dI1I2, layer='I', model=model) * (w_sstack_legC/5)  * a_factor * include_I

            G_I        =  G_SiNxI1 + G_I2 + G_SiNxI1I2_w2e + G_I1I2_nom

        elif legD:   # S-W1-I1-I2

            G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5 * a_factor * include_S

            G_W1   = G_layer(fit, dW1, layer='W', model=model)            * w1w_tot/5 * a_factor * include_W
            G_W    = G_W1

            # I1-I2 to W1 width, SiNx-I1-I2 stack beyond W1
            w_sstack_legD = w_istack + w_sstack + w2w_s + w2w_e
            G_SiNx         = G_layer(fit, dSiNx,       layer='I', model=model)            * (w1w_tot/5)       * a_factor * include_I
            G_I1I2         = (G_layer(fit, dI1I2,      layer='I', model=model)*w1w_ns/5 + G_layer(fit, dI1I2+deltad_DW1, layer='I', model=model)*w1w_s/5) * a_factor * include_I

            G_SiNxI1I2_w1e = G_layer(fit, dSiNx+dI1I2+deltad_DW1, layer='I', model=model) * (w1w_e/5)         * a_factor * include_I
            G_SiNxI1I2_nom = G_layer(fit, dSiNx+dI1I2, layer='I', model=model)            * (w_sstack_legD/5) * a_factor * include_I   # rest of leg

            G_I = G_SiNx + G_I1I2 + G_SiNxI1I2_w1e + G_SiNxI1I2_nom

        elif legE:   # S-W1-W2
            G_S     = G_layer(fit, dOx, layer='S', model=model, dS0=0.400)    * lw/5           * a_factor * include_S

            # W1-W2 stack trimmed to W2 width
            G_W1W2  = G_layer(fit, dW1+dW2, layer='W', model=model)           * (w2w_tot/5)    * a_factor * include_W
            G_W = G_W1W2

            # allow for different dSiNx between 3 and 5 um
            G_SiNxW2 = G_layer(fit, dSiNx, layer='S', model=model)            * w2w_tot/5      * a_factor * include_S   # substrate still treated as separate layer
            G_SiNxW1 = G_layer(fit, dSiNx+deltad_EW1, layer='S', model=model) * (lw-w2w_tot)/5 * a_factor * include_S   # substrate still treated as separate layer
            G_I = G_SiNxW2 + G_SiNxW1

        elif legF:   # S-I1-I2
            G_S        = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5       * a_factor * include_S

            G_W        = 0

            # SiNx-I1-I2 layer over whole leg
            G_SiNxI1I2 = G_layer(fit, dSiNx+dI1I2, layer='I', model=model) * lw/5 * a_factor * include_I   # rest of leg
            G_I        = G_SiNxI1I2

        elif legG:   # S
            G_S    = G_layer(fit, dOx, layer='S', model=model, dS0=0.400) * lw/5 * a_factor * include_S   # substrate still treated as separate layer

            G_W    = 0

            G_SiNx = G_layer(fit, dSiNx, layer='I', model=model)          * lw/5 * a_factor * include_I
            G_I    = G_SiNx

    else:   # no layer stacking unless explicit geometry

        if legA:   # S-W1-I1-W2-I2

            G_S    = G_layer(fit, dS, layer='S', model=model)    * lw/5         * a_factor * include_S   # substrate still treated as separate layer

            G_W1   = G_layer(fit, dW1, layer='W', model=model)   * (w1w_tot/5)     * a_factor * include_W
            G_W2   = G_layer(fit, dW2, layer='W', model=model)   * (w2w_tot/5)     * a_factor * include_W
            G_W    = G_W1 + G_W2

            # I1 and I2 separate to W1 width, stacked beyond W1
            G_I1       = G_layer(fit, dI1, layer='I', model=model)            * (w1w_tot/5)     * a_factor * include_I
            G_I2_w1    = (G_layer(fit, dI2, layer='I', model=model)*(w1w_ns-w2w_s-w2w_e)/5 + G_layer(fit, dI2+deltad_AW1, layer='I', model=model)*w1w_s/5) * a_factor * include_I
            G_I2_w2se  = G_layer(fit, dI2+deltad_AW2, layer='I', model=model) * (w2w_s+w2w_e)/5 * a_factor * include_I   # add extra I2 on W2 slope and edge but keep I1 and I2 separate

            G_I1I2_w1e = G_layer(fit, dI1I2+deltad_AW1, layer='I', model=model) * (w1w_e/5)    * a_factor * include_I
            G_I1I2_nom = G_layer(fit, dI1I2, layer='I', model=model) * (w_istack/5) * a_factor * include_I

            G_I        = G_I1 + G_I2_w1 + G_I2_w2se + G_I1I2_w1e + G_I1I2_nom

        elif legB:   # S-W1-I1-W2
            G_S    = G_layer(fit, dS, layer='S', model=model)  * lw/5        * a_factor * include_S   # substrate still treated as separate layer
            # G_S    = G_layer(fit, dS+0.087, layer='S', model=model) * lw/5         * a_factor * include_S   # substrate still treated as separate layer

            G_W2   = G_layer(fit, dW2, layer='W', model=model) * (w2w_tot/5) * a_factor * include_W
            G_W1   = G_layer(fit, dW1, layer='W', model=model) * (w1w_tot/5) * a_factor * include_W
            G_W    = G_W1 + G_W2

            # I1 is trimmed to W2 width, no I2
            G_I1   = G_layer(fit, dI1, layer='I', model=model) * (w2w_tot/5) * a_factor * include_I
            G_I    = G_I1

        elif legC:   # S-I1-W2-I2
            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5         * a_factor * include_S   # substrate still treated as separate layer

            G_W2   = G_layer(fit, dW2, layer='W', model=model)     * (w2w_tot/5)      * a_factor * include_W
            G_W    = G_W2

            # I1 and I2 separate to W1 width, stacked beyond W1
            G_I1       = G_layer(fit, dI1,            layer='I', model=model) * (w1w_tot/5)                 * a_factor * include_I
            G_I2_se    = G_layer(fit, dI2+deltad_CW2, layer='I', model=model) * (w2w_s+w2w_e)/5             * a_factor * include_I   # I2 thicker on W2 slope and edge but still separate from I1
            G_I2_ns    = G_layer(fit, dI2,            layer='I', model=model) * ((w1w_tot-(w2w_s+w2w_e))/5) * a_factor * include_I

            w_istack_legC = w_istack + w1w_e
            G_I1I2     = G_layer(fit, dI1I2, layer='I', model=model) * (w_istack_legC/5)  * a_factor * include_I

            G_I    = G_I1 + G_I2_se + G_I2_ns + G_I1I2


        elif legD:   # S-W1-I1-I2
            G_S    = G_layer(fit, dS, layer='S', model=model)    * lw/5      * a_factor * include_S   # substrate still treated as separate layer

            G_W1   = G_layer(fit, dW1, layer='W', model=model)   * w1w_tot/5 * a_factor * include_W
            G_W    = G_W1

            # I1-I2 stacked across entire leg, nominal height includes W2 edge and slope
            w_istack_legD = w_istack + w1w_ns
            G_I1I2_nom    = G_layer(fit, dI1I2, layer='I', model=model) * (w_istack_legD/5) * a_factor * include_I   # total G of I stacks with nominal thickness, incl. lw beyond W1 edge
            G_I1I2_se     = G_layer(fit, dI1I2+deltad_DW1, layer='I', model=model) * ((w1w_s+w1w_e)/5) * a_factor * include_I
            G_I = G_I1I2_nom + G_I1I2_se

        elif legE:   # S-W1-W2
            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5   * a_factor * include_S   # substrate still treated as separate layer

            # W1-W2 stack trimmed to W2 width
            G_W1W2 = G_layer(fit, dW1+dW2, layer='W', model=model) * w2w_tot/5 * a_factor * include_W   # W1-W2 stack is only 3 um wide
            G_W    = G_W1W2

            G_I    = 0

        elif legF:   # S-I1-I2
            G_S = G_layer(fit, dS, layer='S', model=model)       * lw/5 * a_factor * include_S   # substrate still treated as separate layer

            G_W = 0

            # I1-I2 stack over whole leg
            G_I1I2 = G_layer(fit, dI1I2, layer='I', model=model) * lw/5 * a_factor * include_I
            G_I    = G_I1I2

        elif legG:   # S
            G_S    = G_layer(fit, dS, layer='S', model=model)      * lw/5 * a_factor * include_S   # substrate still treated as separate layer

            G_W    = 0

            G_I    = 0

    return G_S + G_W + G_I

def Gfrommodel(fit, an_opts, bolo, layer='total'):   # model params, thickness of substrate, leg width, and leg length in um
    # predicts G_TES and error from our model and arbitrary bolo geometry, assumes microstrip on all four legs a la bolo 1b
    # thickness of wiring layers is independent of geometry
    # RETURNS [G prediction, prediction error]

    dsub  = bolo['geometry']['dsub']
    dW1   = bolo['geometry']['dW1']; dI1 = bolo['geometry']['dI1']; dW2 = bolo['geometry']['dW2']; dI2 = bolo['geometry']['dI2']
    model = an_opts['model']
    if layer=='W1': dW2 = np.zeros_like(dsub)

    GS  = G_leg(fit, an_opts, bolo, dsub, dW1, dI1, dW2, dI2,    True, False, False, legA=True)
    GW  = G_leg(fit, an_opts, bolo, dsub, dW1, dI1, dW2, dI2,    False, True, False, legA=True)

    if model=='Two-Layer':   GI    = 0
    if model=='Three-Layer': GI    = G_leg(fit, an_opts, bolo, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=True)
    if an_opts['stack_N']:   GSiNx = G_leg(fit, an_opts, bolo, dsub, np.zeros_like(dsub), np.zeros_like(dsub), np.zeros_like(dsub), np.zeros_like(dsub), False, False, True, legA=True)

    Gwire = GW + GI # G error for microstrip on one leg, summing error works becaSse error is never negative here

    if   layer=='total':            return 4*(GS+Gwire)   # value and error, microstrip + substrate on four legs
    elif layer=='wiring':           return 4*(Gwire)   # value and error, microstrip (W1+I1+W2+I2) on four legs
    elif layer=='S':                return 4*(GS)   # value and error, substrate on four legs
    elif layer=='W' or layer=='W1': return 4*(GW)   # value and error, W1+W2 on four legs
    elif layer=='I':                return 4*(GI)   # value and error, I1+I2 on four legs
    elif layer=='SiNx':             return 4*(GSiNx)
    else: print('Invalid layer type.'); return

def G_bolotest(fit, an_opts, bolo, layer='total'):
    # returns G_TES for bolotest data set given fit parameters
    # supG = factor to suppress G of textured substrate on legs B, E, and G
    # can return full substrate + microstrip, just substrate, just microstrip, or an individual W / I layer

    if   layer=='total':
        include_S = 1; include_W = 1; include_I = 1
    elif layer=='S':
        include_S = 1; include_W = 0; include_I = 0
    elif layer=='wiring':
        include_S = 0; include_W = 1; include_I = 1
    elif layer=='W':
        include_S = 0; include_W = 1; include_I = 0
    elif layer=='I':
        include_S = 0; include_W = 0; include_I = 1
    else:
        print('Unknown layer'+layer+'. an_opts include "total", "wiring", "S, "W", and "I".')

    layer_ds = bolo['geometry']['layer_ds']
    [dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC] = layer_ds.T

    # G of individual legs
    #        G_leg(fit, an_opts, bolo, dS,           dW1,     dI1,     dW2,    dI2,    include_S, include_W, include_I)
    G_legA = G_leg(fit, an_opts, bolo, dS_ABD,       dW1_ABD, dI1_ABC, dW2_AC, dI2_AC, include_S, include_W, include_I,                legA=True)   # S-W1-I1-W2-I2
    G_legB = G_leg(fit, an_opts, bolo, dS_ABD,       dW1_ABD, dI1_ABC, dW2_B,  0.,     include_S, include_W, include_I, supG_width=2., legB=True)   # S-W1-I1-W2, textured substrate for 2 um
    # G_legB = G_leg(fit, an_opts, bolo, dS_ABD+0.087, dW1_ABD, dI1_ABC, dW2_B,  0.,     include_S, include_W, include_I, supG_width=2., legB=True)   # S-W1-I1-W2, textured substrate for 2 um
    G_legC = G_leg(fit, an_opts, bolo, dS_CF,        0.,      dI1_ABC, dW2_AC, dI2_AC, include_S, include_W, include_I,                legC=True)   # S-I1-W2-I2 (S-I1 stack)
    G_legD = G_leg(fit, an_opts, bolo, dS_ABD,       dW1_ABD, dI_DF,   0.,     0.,     include_S, include_W, include_I,                legD=True)   # S-W1-I1-I2 (I stack)
    G_legE = G_leg(fit, an_opts, bolo, dS_E,         0,       0.,      dW_E,   0.,     include_S, include_W, include_I, supG_width=4., legE=True)   # S-W1-W2 (W stack), textured substrate for 4 um
    G_legF = G_leg(fit, an_opts, bolo, dS_CF,        0.,      dI_DF,   0.,     0.,     include_S, include_W, include_I,                legF=True)   # S-I1-I2 (I stack)
    G_legG = G_leg(fit, an_opts, bolo, dS_G,         0.,      0.,      0.,     0.,     include_S, include_W, include_I, supG_width=4., legG=True)   # bare S, textured substrate for 4 um

    # G_TES for bolotest devices
    G_1b = 4*G_legA                         # aka Bolo 1 = 4x(S-W1-I1-W2-I2);                                   slightly over (same quality) with higher G_S in two-layer model
    G_24 = 1*G_legA + 3*G_legG              # aka Bolo 2 = 3x(S-W1-I1-W2-I2) + 1x(S);                           slightly under (slightly worse) with lower G_S in two-layer model
    G_23 = 2*G_legA + 2*G_legG              # aka Bolo 3 = 2x(S-W1-I1-W2-I2) + 2x(S);                           slightly under (slightly worse) with lower G_S in two-layer model
    G_22 = 3*G_legA + 1*G_legG              # aka Bolo 4 = 3x(S-W1-I1-W2-I2) + 1x(S);                           about the same, slightly higher G_W
    G_21 = 1*G_legB + 3*G_legF              # aka Bolo 5 = 1x(S-W1-I1-W2)    + 3x(S-I1-I2);                     slightly over (better fit) with higher G_S and lower G_I
    G_20 = 1*G_legB + 3*G_legE              # aka Bolo 6 = 1x(S-W1-I1-W2)    + 3x(S-W1-W2)                      slightly under (slightly better) with higher G_I
    G_7  = 2*G_legA + 1*G_legC + 1*G_legD   # aka Bolo 7 = 2x(S-W1-I1-W2-I2) + 1x(S-I1-W2-I2) + 1x(S-W1-I1-I2); under (worse) because of lower G_I
    G_13 = 1*G_legB + 3*G_legG              # aka Bolo 8 = 1x(S-W1-I1-W2)    + 3x(S);                           slightly under (slightly worse) with lower G_S

    if len(fit.shape)==1 or np.shape(fit)[0]==1:   # return values and errors
        return np.array([G_1b, G_24, G_23, G_22, G_21, G_20, G_7, G_13])   # return values
    elif np.shape(fit)[0]==2:   # return values and errors
        Gbolos = np.array([G_1b[0], G_24[0], G_23[0], G_22[0], G_21[0], G_20[0], G_7[0], G_13[0]]).T; sigma_Gbolos = np.array([G_1b[1], G_24[1], G_23[1], G_22[1], G_21[1], G_20[1], G_7[1], G_13[1]]).T
        return Gbolos, sigma_Gbolos
    else:   # return values for each row, prob result of simulation
        return np.array([G_1b, G_24, G_23, G_22, G_21, G_20, G_7, G_13]).T

### fit model
def chisq_val(params, args):   # calculates chi-squared value
    an_opts, bolo = args
    ydata = bolo['data']['ydata']; sigma = bolo['data']['sigma']
    Gbolos_model = G_bolotest(params, an_opts, bolo)   # predicted G of each bolo
    chisq_vals = (Gbolos_model-ydata)**2/sigma**2

    return np.sum(chisq_vals)

def runsim_chisq(bolo, an_opts, plot_opts, save_sim=False):
    # returns G and alpha fit parameters
    # returned G's have units of ydata (most likely pW/K)

    # sort analysis an_opts
    num_its = an_opts.get('num_its')
    model = an_opts.get('model', 'Three-Layer')
    p0 = an_opts.get('p0')
    vary_thickness = an_opts['vary_thickness']
    bounds = an_opts.get('bounds', [])
    sim_file = an_opts.get('sim_file')
    csv_file = an_opts.get('csv_file')
    ydata = bolo['data']['ydata']; sigma = bolo['data']['sigma']
    layer_ds = bolo['geometry']['layer_ds']
    derrs = bolo['geometry'].get('derrs', [])

    # save_figs = plot_opts['save_figs']
    # plot_dir = plot_opts['plot_dir']

    print('\n'); print('Running MC Simulation on '+model+' Model'); print('\n')
    pfits_sim = np.empty((num_its, len(p0)))
    y_its = np.empty((num_its, len(ydata)))
    sim_layerds = np.empty((num_its, len(layer_ds)))
    Gwires = np.empty((num_its, len(ydata)))

    print('starting sim at {now}'.format(now=datetime.now().time())); print('\n')
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error

        if vary_thickness:   # pull thicknesses from normal distribution, assuming error is some % of d
            dS_ABD0, dS_CF0, dS_E0, dS_G0, dW1_ABD0, dW1_E0, dI1_ABC0, dI_DF0, dW2_AC0, dW2_B0, dI2_AC0 = layer_ds
            dSABD_err, dSCF_err, dSE_err, dSG_err, dW1ABD_err,  dW1E_err, dI1ABC_err, dIDF_err, dW2AC_err, dW2BE_err, dI2AC_err= derrs

            dS_ABD =  normal(dS_ABD0, dSABD_err); dS_CF = normal(dS_CF0, dSCF_err); dS_E = normal(dS_E0, dSE_err); dS_G = normal(dS_G0, dSG_err)   # thickness of S for leg A, C, and G [um]
            dW1_ABD = normal(dW1_ABD0, dW1ABD_err); dW1_E = normal(dW1_E0, dW1E_err)   # thickness of W1 for legs A and E [um]
            dI1_ABC = normal(dI1_ABC0, dI1ABC_err); dI_DF = normal(dI_DF0, dIDF_err)   # thickness of I1 for legs A and G [um]
            dW2_AC =  normal(dW2_AC0, dW2AC_err); dW2_B = normal(dW2_B0, dW2BE_err)   # thickness of W2 for legs A and B [um]
            dI2_AC =  normal(dI2_AC0, dI2AC_err)   # thickness of I1 for legs A and C [um]
            sim_layerds[ii] = dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC
        else:
            sim_layerds[ii] = layer_ds

        # dicts specific to this iteration - passed to chisq_val
        it_bolo = copy.deepcopy(bolo)   # input iteration-specific layer thicknesses and G_bolotest data
        it_bolo['geometry']['layer_ds'] = sim_layerds[ii]    # overwrite layer thicknesses
        it_bolo['data']['ydata'] = y_its[ii]   # overwrite G_bolotest data
        it_result = minimize(chisq_val, p0, args=[an_opts, it_bolo], bounds=bounds)   # minimize chi-squared function with this iteration's G_TES values and film thicknesses
        pfits_sim[ii] = it_result['x']   # fit parameters for this iteration

        if   ii/num_its == 0.25: print('25% Finished at {now}'.format(now=datetime.now().time())); print('\n')
        elif ii/num_its == 0.50: print('50% Finished at {now}'.format(now=datetime.now().time())); print('\n')
        elif ii/num_its == 0.75: print('75% Finished at {now}'.format(now=datetime.now().time())); print('\n')

    print('Finished Simulation'); print('\n')

    # Predictions for bolotest G(d0) and G of the microstrip (W1-I1-W2-I2)
    Gpreds =   G_bolotest(pfits_sim, an_opts, bolo)   # simulation G predictions using d0
    Gpred_Ss = G_bolotest(pfits_sim, an_opts, bolo, layer='S')   # substrate contribution
    Gpred_Ws = G_bolotest(pfits_sim, an_opts, bolo, layer='W')   # W layer contributions
    Gpred_Is = G_bolotest(pfits_sim, an_opts, bolo, layer='I')   # I layer contributions
    Gwires =   G_bolotest(pfits_sim, an_opts, bolo, layer='wiring').T[0]/4   # function outputs G_microstrip for four legs

    sim_dict = {}
    sim_dict['sim'] = {}   # save simulation arrays
    sim_dict['sim']['fit_params'] = pfits_sim   # fit results of each iteration
    sim_dict['y_its'] = y_its   # simulated y-data from G measurement +/- 1 sigma [pW / K]
    sim_dict['sim_layerds'] = sim_layerds   # simulated layer thicknesses from d0 +/- derr[um]
    sim_dict['Gwires'] = Gwires   # G of W1-I2-W2-I2 on one leg [pW / K]
    sim_dict['Gpreds'] = Gpreds  # G(d0) predictions using the fit parameters from each iteration
    sim_dict['Gpred_Ss'] = Gpred_Ss  # G(d0) predictions using the fit parameters from each iteration
    sim_dict['Gpred_Ws'] = Gpred_Ws  # G(d0) predictions using the fit parameters from each iteration
    sim_dict['Gpred_Is'] = Gpred_Is  # G(d0) predictions using the fit parameters from each iteration

    # save analysis options and bolometer geometry/data
    sim_dict['an_opts'] = an_opts
    sim_dict['bolo']    = bolo

    # sort and save fit results
    fit_dict        = sort_results(sim_dict, print_results=True)
    sim_dict['fit'] = fit_dict

    if save_sim:
        print('Saving simulation to ', sim_file); print('\n')

        # save pkl
        with open(sim_file, 'wb') as outfile:   # save simulation pkl
            pkl.dump(sim_dict, outfile)

        # save csv
        fit_params = fit_dict['fit_params']; fit_std = fit_dict['fit_std']
        kappas = fit_dict['kappas'] ; sigma_kappas = fit_dict['sigma_kappas']   # thermal conductivity of U, W[, I] layers [pW / K / um]
        chisq_qsum = fit_dict['chisq_qsum']; rchisq_qsum = fit_dict['rchisq_qsum']  # chi-squared value for final fit parameters
        Gwire = fit_dict['Gwire']; sigma_Gwire = fit_dict['sigma_Gwire']   # G of W1-I2-W2-I2 on one leg - final result[pW / K]

        if an_opts['model']=='Three-Layer':
            vals_med = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], kappas[0],       kappas[1],       kappas[2],       Gwire,       chisq_qsum, rchisq_qsum])
            vals_err = np.array([fit_std[0],    fit_std[1],    fit_std[2],    fit_std[3],    fit_std[4],    fit_std[5],    sigma_kappas[0], sigma_kappas[1], sigma_kappas[2], sigma_Gwire, '',         ''])
            csv_params = np.array(['GS (pW/K)', 'GW (pW/K)', 'GI (pW/K)', 'alphaS', 'alphaW', 'alphaI', 'kappaS (pW/K/um)', 'kappaW (pW/K/um)', 'kappaI (pW/K/um)', 'Gwire (pW/K)', 'Chi-sq val', 'Red Chi-sq val'])
        elif an_opts['model']=='Two-Layer':
            vals_med = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], kappas[0],       kappas[1],       Gwire,       chisq_qsum, rchisq_qsum])
            vals_err = np.array([fit_std[0],    fit_std[1],    fit_std[2],    fit_std[3],    sigma_kappas[0], sigma_kappas[1], sigma_Gwire, '', ''])   # should be the same for mean and median
            csv_params = np.array(['GS (pW/K)', 'GW (pW/K)', 'alphaS', 'alphaW', 'kappaS (pW/K/um)', 'kappaW (pW/K/um)', 'Gwire (pW/K)', 'Chi-sq val', 'Red Chi-sq val'])

        # write CSV
        fields = np.array(['Parameter', an_opts['calc'], 'Error'])
        rows = [[csv_params[rr], vals_med[rr], vals_err[rr]] for rr in np.arange(len(csv_params))]
        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)  # csv writer object
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

    return sim_dict

def sort_results(sim_dict, print_results=False, spinds=np.array([])):
    # sort and print simulation fit results

    # load simulation parameters
    an_opts = sim_dict['an_opts']
    model   = an_opts['model']
    calc    = an_opts['calc']

    bolo = sim_dict['bolo']
    L   = sim_dict['bolo']['geometry']['ll']  # um, bolotest leg length
    A_S = 5*0.400; A_W = 5*0.400; A_I = 5*0.400   # um^2, for converting G(d0) to kappa

    # look at subpopulation of fit parameters?
    sim_temp = sim_dict['sim']['fit_params']
    if len(spinds)==0: spinds = np.arange(np.shape(sim_temp)[0])   # nope, look at all solutions
    Gpreds = sim_dict['Gpreds'][spinds]; Gwires = sim_dict['Gwires'][spinds]; sim = sim_dict['sim']['fit_params'][spinds]
    Gpred_Ss = sim_dict['Gpred_Ss'][spinds]; Gpred_Ws = sim_dict['Gpred_Ws'][spinds]; Gpred_Is = sim_dict['Gpred_Is'][spinds]   # U, W, and I contributions to Gpredicted

    sim_params_mean = np.mean(sim, axis=0);        Gwire_mean   = np.mean(Gwires); Gpred_mean = np.mean(Gpreds, axis=0)
    Gpred_S_mean    = np.mean(Gpred_Ss, axis=0);   Gpred_W_mean = np.mean(Gpred_Ws, axis=0); Gpred_I_mean = np.mean(Gpred_Is, axis=0)
    sim_params_med  = np.median(sim, axis=0);      Gwire_med    = np.median(Gwires); Gpred_med = np.median(Gpreds, axis=0)
    Gpred_S_med     = np.median(Gpred_Ss, axis=0); Gpred_W_med  = np.median(Gpred_Ws, axis=0); Gpred_I_med = np.median(Gpred_Is, axis=0)
    sim_std         = np.std(sim, axis=0);         sigma_Gwire  = np.std(Gwires); sigma_Gpred = np.std(Gpreds, axis=0)
    sigma_GpredS    = np.std(Gpred_Ss, axis=0);    sigma_GpredW = np.std(Gpred_Ws, axis=0); sigma_GpredI = np.std(Gpred_Is, axis=0)

    if calc == 'Mean':
        sim_params = sim_params_mean; Gwire = Gwire_mean; Gpred = Gpred_mean
        Gpred_S = Gpred_S_mean; Gpred_W = Gpred_W_mean; Gpred_I = Gpred_I_mean
    elif calc == 'Median':
        sim_params = sim_params_med; Gwire = Gwire_med; Gpred = Gpred_med
        Gpred_S = Gpred_S_med; Gpred_W = Gpred_W_med; Gpred_I = Gpred_I_med

    if model=='Three-Layer':   # 6 fit parameters
        S_sim, W_sim, I_sim, aS_sim, aW_sim, aI_sim = sim_params   # parameter fits from Monte Carlo
        Serr_sim, Werr_sim, Ierr_sim, aSerr_sim, aWerr_sim, aIerr_sim = sim_std   # parameter errors from Monte Carlo

        kappaS = GtoKappa(S_sim, A_S, L); sigkappaS = GtoKappa(Serr_sim, A_S, L)   # pW / K / um; error analysis is correct becaSse kappa(G) just depends on constants
        kappaW = GtoKappa(W_sim, A_W, L); sigkappaW = GtoKappa(Werr_sim, A_W, L)   # pW / K / um
        kappaI = GtoKappa(I_sim, A_I, L); sigkappaI = GtoKappa(Ierr_sim, A_I, L)   # pW / K / um
        kappas = [kappaS, kappaW, kappaI]; sigma_kappas = [sigkappaS, sigkappaW, sigkappaI]

        dof = 1   # 1 degree of freedom for 6 fit parameters and chi-squared calc with 8 data points

    elif model=='Two-Layer':   # 4 fit parameters
        S_sim, W_sim, aS_sim, aW_sim = sim_params   # parameter fits from Monte Carlo
        Serr_sim, Werr_sim, aSerr_sim, aWerr_sim = sim_std   # parameter errors from Monte Carlo

        kappaS = GtoKappa(S_sim, A_S, L); sigkappaS = GtoKappa(Serr_sim, A_S, L)   # pW / K / um; error analysis is correct becaSse kappa(G) just depends on constants
        kappaW = GtoKappa(W_sim, A_W, L); sigkappaW = GtoKappa(Werr_sim, A_W, L)   # pW / K / um
        kappas = [kappaS, kappaW]; sigma_kappas = [sigkappaS, sigkappaW]

        dof = 3   # 3 degrees of freedom for 6 fit parameters and chi-squared calc with 8 data points

    sigma_qsum = np.sqrt(bolo['data']['sigma']**2 + sigma_Gpred**2)   # this may double-count error of data points
    bolo_pred = copy.deepcopy(bolo); bolo_pred['data']['sigma'] = sigma_Gpred
    bolo_qsum = copy.deepcopy(bolo); bolo_qsum['data']['sigma'] = sigma_qsum
    chisq_fit =  chisq_val(sim_params, [an_opts, bolo]); rchisq_fit = chisq_fit/dof
    chisq_pred = chisq_val(sim_params, [an_opts, bolo_pred]); rchisq_pred = chisq_pred/dof
    chisq_qsum = chisq_val(sim_params, [an_opts, bolo_qsum]); rchisq_qsum = chisq_qsum/dof

    # sort final results
    fit_dict = {}
    fit_dict['fit_params']   = sim_params;   fit_dict['fit_std']      = sim_std   # fit parameters - final result
    fit_dict['Gwire']        = Gwire;        fit_dict['sigma_Gwire']  = sigma_Gwire   # G of W1-I2-W2-I2 on one leg - final result [pW / K]
    fit_dict['Gpred']        = Gpred;        fit_dict['sigma_Gpred']  = sigma_Gpred   # G(d0) prediction - final result [pW / K]
    fit_dict['Gpred_S']      = Gpred_S;      fit_dict['sigma_GpredS'] = sigma_GpredS   # G(d0) prediction - substrate contribution [pW / K]
    fit_dict['Gpred_W']      = Gpred_W;      fit_dict['sigma_GpredW'] = sigma_GpredW   # G(d0) prediction - Nb wiring layer contribution [pW / K]
    fit_dict['Gpred_I']      = Gpred_I;      fit_dict['sigma_GpredI'] = sigma_GpredI   # G(d0) prediction - insulating nitride layer contribution [pW / K]
    fit_dict['Gwire_mean']   = Gwire_mean;   fit_dict['Gwire_med']    = Gwire_med   # G of W1-I2-W2-I2 on one leg - mean and median values [pW / K]
    fit_dict['Gpred_mean']   = Gpred_mean;   fit_dict['Gpred_med']    = Gpred_med   # G of W1-I2-W2-I2 on one leg - mean and median values [pW / K]
    fit_dict['Gpred_S_mean'] = Gpred_S_mean; fit_dict['Gpred_S_med']  = Gpred_S_med   # G of W1-I2-W2-I2 on one leg - mean and median values [pW / K]
    fit_dict['Gpred_W_mean'] = Gpred_W_mean; fit_dict['Gpred_W_med']  = Gpred_W_med   # G of W1-I2-W2-I2 on one leg - mean and median values [pW / K]
    fit_dict['Gpred_I_mean'] = Gpred_I_mean; fit_dict['Gpred_I_med']  = Gpred_I_med   # G of W1-I2-W2-I2 on one leg - mean and median values [pW / K]
    fit_dict['sigma_Gqsum']  = sigma_qsum   # quadrature sum of sigma_G from power law fit and sigma_Gpredictions
    fit_dict['kappas']       = kappas;       fit_dict['sigma_kappas'] = sigma_kappas   # thermal conductivity of U, W[, I] layers [pW / K / um]
    fit_dict['chisq_fit']    = chisq_fit;    fit_dict['rchisq_fit']   = rchisq_fit   # chi-squared value for final fit parameters
    fit_dict['chisq_pred']   = chisq_pred;   fit_dict['rchisq_pred']  = rchisq_pred   # chi-squared value for final fit parameters
    fit_dict['chisq_qsum']   = chisq_qsum;   fit_dict['rchisq_qsum']  = rchisq_qsum   # chi-squared value for final fit parameters

    if print_results:
        print ('\n\n' + model + ' Model Fit taking '+ calc +' values:')
        # print('G_S(400 nm) = ', round(S_sim, 2), ' +/- ', round(Serr_sim, 2), 'pW/K')
        print('G_S(400 nm) = ', round(S_sim, 2), ' +/- ', round(Serr_sim, 2), 'pW/K')
        print('G_W(400 nm) = ', round(W_sim, 2), ' +/- ', round(Werr_sim, 2), 'pW/K')
        if model=='Three-Layer':
            print('G_I(400 nm) = ', round(I_sim, 2), ' +/- ', round(Ierr_sim, 2), 'pW/K')
        print('alpha_S = ', round(aS_sim, 2), ' +/- ', round(aSerr_sim, 2))
        print('alpha_W = ', round(aW_sim, 2), ' +/- ', round(aWerr_sim, 2))
        if model=='Three-Layer':
            print('alpha_I = ', round(aI_sim, 2), ' +/- ', round(aIerr_sim, 2))
        print('G_microstrip = ', round(Gwire, 2), ' +/- ', round(sigma_Gwire, 2), 'pW/K')

        # print('Kappa_S: ', round(kappaS, 2), ' +/- ', round(sigkappaS, 2), ' pW/K/um')
        # print('Kappa_W: ', round(kappaW, 2), ' +/- ', round(sigkappaW, 2), ' pW/K/um')
        # if model=='Three-Layer':
        #   print('Kappa_I: ', round(kappaI, 2), ' +/- ', round(sigkappaI, 2), ' pW/K/um')

        print('Chi-squared (sig_GTES)   : ', round(chisq_fit, 3))#, '; Reduced Chi-squared value: ', round(rchisq_fit, 3))
        print('Chi-squared (sig_Gpred)  : ', round(chisq_pred, 3))#, '; Reduced Chi-squared value: ', round(rchisq_pred, 3))
        print('Chi-squared (sig_quadsum): ', round(chisq_qsum, 3))#, '; Reduced Chi-squared value: ', round(rchisq_pred, 3))
        print('\n\n')

    return fit_dict

def plot_simdata(sim_dict, plot_opts):
        # check simulated GTES data is a normal dist'n
        Gpreds = sim_dict['Gpreds']; #Gwires = sim_dict['Gwires']; sim = sim_dict['sim']['fit_params'];
        y_its = sim_dict['y_its']; #data = sim_dict['data']; #calc = sim_dict['an_opts']['calc']
        num_its = sim_dict['an_opts']['num_its']
        # ydata = data['ydata']
        sigma_Gpred = np.std(Gpreds, axis=0)
        data = sim_dict['bolo']['data']; ydata = data['ydata']; sigma = data['sigma']
        fn_comments = sim_dict['an_opts'].get('fn_comments', '')

        save_figs = plot_opts.get('save_figs', False)
        plot_dir = plot_opts.get('plot_dir', './')
        bolo_labels = plot_opts['bolo_labels']

        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure(figsize=(8,6))
            plt.hist(Gpreds.T[yy], bins=10, label='Predicted G$_\\text{TES}$(d$_0$)', alpha=0.7, color='C1')
            n, bins, patches = plt.hist(yit, bins=5, label='Simulated G$_\\text{TES}$', alpha=0.9, color='C0')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.title('G$_\\text{TES}$('+bolo_labels[yy]+') = '+str(np.round(ydata[yy], 1))+' pW/K (N='+str(num_its)+')')
            plt.annotate('$\\sigma_{sim}$ = '+str(np.round(sigma_Gpred[yy]/ydata[yy]*100, 1))+'\\% \n $\\sigma_{fit}$ = '+str(np.round(sigma[yy]/ydata[yy]*100, 1))+'\\%', (max(bins), 0.9*max(n)))
            ax = plt.gca(); handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left')
            if save_figs: plt.savefig(plot_dir + 'bolo' + bolo_labels[yy] + '_simydata' + fn_comments + '.png', dpi=300)

### visualize and evaluate quality of fit
def calc_func_grid(params, an_opts, bolo):   # chi-squared parameter space
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params):
        for cc, col in enumerate(row):
            params_rc = col
            func_grid[rr, cc] = chisq_val(params_rc, [an_opts, bolo])
    return func_grid

def qualityplots(sim_dict, plot_opts, figsize=(17,5.75)):
    ### plot chisq values in 2D parameter space (alpha_x vs G_x) overlayed with resulting parameters from simulation for all three layers
    # params can be either the mean or median of the simulation values
    # spinds are indexes of a certain subpopulation to plot. if the length of this is 0, it will analyze the entire population.

    bolo = sim_dict.get('bolo')   # bolotest data and geometry
    an_opts = sim_dict.get('an_opts')

    model = an_opts.get('model')   # bolotest data
    fn_comments = an_opts.get('fn_comments')

    save_figs = plot_opts.get('save_figs', False)
    plot_dir = plot_opts.get('plot_dir', './')

    title = plot_opts.get('title', '')
    qp_lim = plot_opts.get('qp_lim')
    vmax = plot_opts.get('vmax')

    if model=='Three-Layer':
        layers = np.array(['S', 'W', 'I'])
        num_sp = 3   # three subplots
    elif model=='Two-Layer':
        layers = np.array(['S', 'W'])
        num_sp = 2   # two subplots

    if type(sim_dict)==dict:
        fit_dict = sim_dict['fit']
        fit_params = fit_dict['fit_params']; fit_errs = fit_dict['fit_std']   # fit parameters - final result

    else:   # option to pass just fit parameters
        fit_params = sim_dict['sim']['fit_params']
        fit_errs = np.array([0,0,0,0,0,0])

    xgridlim=qp_lim; ygridlim=qp_lim   # alpha_layer vs G_layer
    xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:150j, ygridlim[0]:ygridlim[1]:150j]   # make 2D grid for plotter
    wspace = 0.25

    fig = plt.figure(figsize=figsize)   # initialize figure
    fig.subplots_adjust(wspace=wspace, left=0.065)

    for ll, layer in enumerate(layers):
        xlab = '\\textbf{G}$_\\textbf{'+layer+'}$'
        ylab = '$\\boldsymbol{\\alpha_\\textbf{'+layer+'}}$'
        if layer=='S':
            if model=='Three-Layer':
                Gind=0; aind=3   # G and alpha indexes in parameters array
                gridparams = np.array([xgrid, fit_params[1]*np.ones_like(xgrid), fit_params[2]*np.ones_like(xgrid), ygrid, fit_params[4]*np.ones_like(ygrid), fit_params[5]*np.ones_like(ygrid)]).T
            elif model=='Two-Layer':
                Gind=0; aind=2   # G and alpha indexes in parameters array
                gridparams = np.array([xgrid, fit_params[1]*np.ones_like(xgrid), ygrid, fit_params[3]*np.ones_like(ygrid)]).T
            splot_ID = '\\textbf{i.}'
        elif layer=='W':
            if model=='Three-Layer':
                Gind=1; aind=4   # G and alpha indexes in parameters array
                gridparams = np.array([fit_params[0]*np.ones_like(xgrid), xgrid, fit_params[2]*np.ones_like(xgrid), fit_params[3]*np.ones_like(ygrid), ygrid, fit_params[5]*np.ones_like(ygrid)]).T
            elif model=='Two-Layer':
                Gind=1; aind=3   # G and alpha indexes in parameters array
                gridparams = np.array([fit_params[0]*np.ones_like(xgrid), xgrid, fit_params[2]*np.ones_like(ygrid), ygrid]).T
            splot_ID = '\\textbf{ii.}'
        elif layer=='I':   # only called in 3 layer model
            Gind=2; aind=5   # G and alpha indexes in parameters array
            gridparams = np.array([fit_params[0]*np.ones_like(xgrid), fit_params[1]*np.ones_like(xgrid), xgrid, fit_params[3]*np.ones_like(ygrid), fit_params[4]*np.ones_like(ygrid), ygrid]).T
            splot_ID = '\\textbf{iii.}'

        funcgrid = calc_func_grid(gridparams, an_opts, bolo)   # calculate chisq values for points in the grid
        ax = fig.add_subplot(1,num_sp,ll+1)   # select subplot
        im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmax=vmax, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.6)   # quality plot
        plt.errorbar(fit_params[Gind], fit_params[aind], xerr=fit_errs[Gind], yerr=fit_errs[aind], color='black', label='\\textbf{Model Fit}', capsize=2, linestyle='None')   # fit results
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
        plt.annotate(splot_ID, (xgridlim[0]+0.1, xgridlim[1]-0.3), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        if ll==2:
            axpos = ax.get_position()
            cax = fig.add_axes([axpos.x1+0.02, axpos.y0+0.04, 0.01, axpos.y1-axpos.y0-0.08], label='\\textbf{Chi-Sq Value}')
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label('$\\boldsymbol{\\chi^2\\textbf{ Value}}$', rotation=270, labelpad=20)
            ax.legend(loc=(0.1,0.15))
    plt.suptitle(title, fontsize=20, y=0.86)
    if save_figs: plt.savefig(plot_dir + 'qualityplots' + fn_comments + '.png', dpi=300)   # save figure

    return

def pairwise(sim_dict, plot_opts, indstp=[], indsop=[], oplotlabel='', fs=(10,8)):
    # make pairwise correlation plots with histograms on the diagonal
    # indstp = index of solutions to plot, defaSlt is all
    # indsop = index of subset of solutions to overplot on all solutions

    save_figs = plot_opts.get('save_figs', False)
    plot_dir = plot_opts.get('plot_dir', './')
    labels = sim_dict['an_opts'].get('param_labels')

    title = plot_opts.get('title', '')
    fn_comments = sim_dict['an_opts'].get('fn_comments')
    labels = sim_dict['an_opts'].get('param_labels')

    sim_dataT = sim_dict['sim']['fit_params'].T   # sim_data needs to be transposed so that it's 6 x number of iterations
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
                ax.scatter(sim_dataT[jj][indstp], sim_dataT[ii][indstp], marker='.', alpha=0.3)   # row shares the y axis, column shares the x axis
                ax.scatter(sim_dataT[jj][indsop], sim_dataT[ii][indsop], marker='.', alpha=0.3, color='C2')   # highlight subset of solutions
                ax.set_xlim(limits[jj]); ax.set_ylim(limits[ii])
                ax.set_xlabel(labels[jj]); ax.set_ylabel(labels[ii])

    axes = pairfig.get_axes()
    for ax in axes:   # only label bottom and left side
        ax.label_outer()
    if len(indsop)!=0:
        ax = axes[0]
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc=(-0.5, 0.3))
    plt.suptitle(title+'; \\textbf{ (N='+str(nsolns)+')}', fontsize=20, y=0.93)
    if save_figs: plt.savefig(plot_dir + 'pairwiseplots' + fn_comments + '.png', dpi=300)   # save figure

    return pairfig

def bolotest_AoL(an_opts, lw=5., ll=220.):
    # calculate bolotest leg xsect area over length

    layer_ds = an_opts.get('layer_ds', np.array([0.372, 0.312, 0.199, 0.181, 0.162, 0.418, 0.298, 0.596, 0.354, 0.314, 0.302]))
    dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC = layer_ds

    A_legA = dS_ABD*lw + dW1_ABD*lw     + dI1_ABC*lw    + dW2_AC*3   + dI2_AC*lw    # S-W1-I1-W2-I2
    A_legB = dS_ABD*lw + dW1_ABD*5      + dI1_ABC*3     + dW2_B*3    + 0            # S-W1-I1-W2, I1 width is actually = W2 width here from FIB measurements
    A_legC = dS_CF*lw  + 0              + dI1_ABC*lw    + dW2_AC*3   + dI2_AC*lw    # S-I1-W2-I2
    A_legD = dS_ABD*lw + dW1_ABD*5      + dI_DF*lw      + 0          + 0            # S-W1-I1-I2 (I stack)
    A_legE = dS_E*lw   + (dW1_E)*lw     + 0             + 0          + 0            # S-W1-W2 (W stack)
    A_legF = dS_CF*lw  + 0              + dI_DF*lw      + 0          + 0            # S-I1-I2 (I stack)
    A_legG = dS_G*lw   + 0              + 0             + 0          + 0            # bare S

    ### bolo 1b = 4×A, 24 = 1×A & 3×G, 23 = 2×A & 2×G, 22= 3×A & 1×G, 21 = 1×B & 3×F, 20 = 1×B & 3×E, 7 = 2×A & 1×C & 1×D, 13=1×B & 3×G
    A_bolo1b = 4*A_legA; A_bolo24 = 1*A_legA + 3*A_legG; A_bolo23 = 2*A_legA + 2*A_legG; A_bolo22 = 3*A_legA + 1*A_legG;
    A_bolo21 = 1*A_legB + 3*A_legF; A_bolo20 = 1*A_legB + 3*A_legE; A_bolo7 = 2*A_legA + 1*A_legC + 1*A_legD; A_bolo13 = 1*A_legB + 3*A_legG
    A_bolo = np.array([A_bolo1b, A_bolo24, A_bolo23, A_bolo22, A_bolo21, A_bolo20, A_bolo7, A_bolo13])
    AoL_bolo = A_bolo/ll   # A/L for bolotest devices

    return AoL_bolo   # 1b, 24, 23, 22, 21, 20, 7, 13

def legA_AoL(layer='full', lwidth=5, layer_ds = np.array([0.372, 0.312, 0.199, 0.181, 0.162, 0.418, 0.298, 0.596, 0.354, 0.314, 0.302])):

    # handle old (smaller) layer thickness arrays
    if len(layer_ds)==10:   # original number of unique film thicknesses
        dS_ABDE, dS_CF, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI1_DF, dW2_AC, dW2_B, dI2_ACDF = layer_ds
        dW2_B = dW2_B; dI2_AC = dI2_ACDF; dS_ABD = dS_ABDE; dS_E = dS_ABDE   # handle renaming
        dW1_E = dW1_E + dW2_B; dI_DF = dI1_DF + dI2_ACDF   # handle combining W and I stacks
    elif len(layer_ds)==11:   # added one more layer thickness after FIB measurements Feb 2024
        dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD, dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_B, dI2_AC = layer_ds

    if lwidth<=8:
        w1w = 5; w2w = 3   # [um] W layer widths
    elif lwidth>8:
        w1w = 8; w2w = 5   # [um] W layer widths

    if layer=='full':
        return dS_ABD*lwidth + dW1_ABD*w1w + dI1_ABC*lwidth + dW2_AC*w2w + dI2_AC*lwidth     # S-W1-I1-W2-I2
    elif layer=='W1':
        return dW1_ABD*w1w   # W1
    elif layer=='bareS':
        return dS_ABD*lwidth   # W1

def plot_modelvdata(sim_dict, plot_opts, up_bolo=None, title='', plot_bolotest=True, plot_vlength=False, pred_wfit=False, fs=(8,6)):
    # plot bolotest data vs model fit
    # plot_bolotest turns on bolos 24-20 etc with various film stacks
    # plot_vlength turns on bolos with varying leg lengths

    sim_data = sim_dict['sim']['fit_params']
    an_opts = sim_dict['an_opts']
    bolo = sim_dict['bolo'] if up_bolo==None else up_bolo   # allow user to pass an updated bolo dict
    ydata = bolo['data']['ydata']; sigma = bolo['data']['sigma']
    vlength_data = bolo['data']['vlength_data']
    layer_ds = bolo['geometry']['layer_ds']
    lw = bolo['geometry']['lw']
    dsub = bolo['geometry']['dsub']
    A_bolo = bolo['geometry']['A']

    # calc = an_opts['calc']
    fn_comments = an_opts['fn_comments']

    save_figs = plot_opts['save_figs']
    plot_dir = plot_opts['plot_dir']
    plot_comments = plot_opts['plot_comments']
    bolo_labels = plot_opts['bolo_labels']
    if not title: title = plot_opts['title']

    # calculate predictions and error bars either with fit parameters or std of predictions from all simulated fit parameters
    if pred_wfit:   # use error bars on fit parameters to calculate error bars on predicted values
        # fit                    = np.array([sim_dict['fit']['fit_params'], sim_dict['fit']['fit_std']])
        fit                    = np.array(sim_dict['fit']['fit_params'])
        Gpred   = G_bolotest(fit, an_opts, bolo); sigma_Gpred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        Gpred_S = G_bolotest(fit, an_opts, bolo, layer='S')   # predictions and error from model [pW/K]
        Gpred_W = G_bolotest(fit, an_opts, bolo, layer='W')   # predictions and error from model [pW/K]
        Gpred_I = G_bolotest(fit, an_opts, bolo, layer='I')   # predictions and error from model [pW/K]

    else:   # G predictions have already been calculated
        Gpred   = sim_dict['fit']['Gpred']; sigma_Gpred = sim_dict['fit']['sigma_Gpred']   # predictions and error [pW/K]
        Gpred_S = sim_dict['fit']['Gpred_S']   # predictions and error of SiNx substrate layers [pW/K]
        Gpred_W = sim_dict['fit']['Gpred_W']   # predictions and error of Nb wiring layers [pW/K]
        Gpred_I = sim_dict['fit']['Gpred_I']   # predictions and error of SiNx insulating layers [pW/K]
        if an_opts['stack_N']:
            sim_data   = sim_dict['sim']['fit_params']
            bolo_SiNx = copy.deepcopy(bolo)
            bolo_SiNx['geometry']['layer_ds'][4:11] = np.zeros(7)
            Gpreds_SiNx = G_bolotest(sim_data, an_opts, bolo_SiNx, layer='I')
            Gpred_SiNx = np.median(Gpreds_SiNx, axis=0)

    plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8,1])
    ax1 = plt.subplot(gs[0])   # model vs data

    if plot_bolotest:
        plt.plot(    A_bolo, Gpred_S,                 color='blue',       marker='d', markersize=9,  label='G$_\\text{S}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
        plt.plot(    A_bolo, Gpred_I,                 color='blueviolet', marker='s', markersize=8,  label='G$_\\text{I}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
        plt.plot(    A_bolo, Gpred_W,                 color='green',      marker='v', markersize=8,  label='G$_\\text{W}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
        lorder = np.array([3, 4, 0, 2, 1])
        if an_opts['stack_N']:
            plt.plot(A_bolo, Gpred_SiNx,               color='orange', marker='s', markersize=8,  label='G$_\\text{SiNx}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
            lorder = np.array([4, 5, 0, 2, 1, 3])
        plt.errorbar(A_bolo, ydata, yerr=sigma,       color='red',        marker='o', markersize=9,  label='Data',          linestyle='None', capsize=3, zorder=-1)
        plt.errorbar(A_bolo, Gpred, yerr=sigma_Gpred, color='k',          marker='*', markersize=11, label='Model',         linestyle='None', capsize=3)
        for bb, boloid in enumerate(bolo_labels):
            plt.annotate(boloid, (A_bolo[bb]+0.4, ydata[bb]))
        # plt.annotate('$\\boldsymbol{\\chi_\\nu^2}$ = '+str(round(rchisq_pred, 1)), (0.0375, 8), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.grid(linestyle = '--', which='both', linewidth=0.5)
        plt.ylim(0, 20)# ; plt.xlim(6, 28)
        # lorder = np.array([0, 1])
        handles, labels = ax1.get_legend_handles_labels()
        plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder], loc=2)   # 2 is upper left, 4 is lower right
    plt.ylabel('$\\textbf{G [pW/K]}$')
    # plt.title(title)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off

    ax2 = plt.subplot(gs[1], sharex=ax1); ax_xlim = ax1.get_xlim()   # residuals
    plt.axhline(0, color='k', alpha=0.7, zorder=-1)
    if plot_bolotest:
        normres = (ydata - Gpred)/sigma_Gpred
        # normres = (ydata - Gpred)
        plt.scatter(A_bolo, normres, color='r', s=80)
    plt.ylabel('\\textbf{N. Res.}', labelpad=-2)
    plt.xlabel('\\textbf{Leg Area [$\\boldsymbol{\mu m^\\mathit{2}}$]}')
    plt.xlim(ax_xlim)
    # plt.ylim(-1.1, 1.1)
    plt.ylim(-3, 3)
    # plt.tick_params(axis='y', which='both', right=True)
    plt.fill_between((ax_xlim), -1, 1, facecolor='k', alpha=0.2)   # +/- 1 sigma
    plt.grid(linestyle = '--', which='both', linewidth=0.5, zorder=-1)
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis

    if save_figs: plt.savefig(plot_dir + 'Gpred_bolotest' + fn_comments + plot_comments + '.png', dpi=300)

    return

### legacy data analysis
def plot_Glegacy(legacy, plot_opts, bolotest={}, analyze_vlength=False, fs=(7,5)):
    # predicts G for legacy TES data using alpha model, then plots prediction vs measurements (scaled to 170 mK)
    # legacy geometry and measurements are from Shannon's spreadsheet, then plots

    save_figs = plot_opts['save_figs']
    plot_dir = plot_opts['plot_dir']
    plot_comments = plot_opts['plot_comments']

    legacy_ll = legacy['geometry']['ll']
    legacy_lw = legacy['geometry']['lw']
    legacy_Gs = legacy['G170mK']

    L_bolo = bolotest['geometry']['ll']
    w_bolo = bolotest['geometry']['lw']

    # vs length
    plt.figure(figsize=fs)
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    # ax1 = plt.subplot(gs[0])   # model vs data

    # plt.errorbar(legacy_ll, legacy_Gs, marker='o', label='Legacy Data', markersize=7, color='darkorange', linestyle='None')
    if len(bolotest.keys())>0:   # plot bolotest 1b data and prediction
        L_bolo = bolotest['geometry']['ll']
        w_bolo = bolotest['geometry']['lw']
        vmin = min(np.append(legacy_lw, w_bolo)); vmax = max(np.append(legacy_lw, w_bolo))
        # plt.errorbar(L_bolo, bolotest['data']['ydata'][0], yerr=bolotest['data']['sigma'][0], marker='o', markersize=7, color='red', label='Bolo 1', capsize=2, linestyle='None')
        plt.scatter(L_bolo, bolotest['data']['ydata'][0], s=40, c=w_bolo, vmin=vmin, vmax=vmax, linestyle='None', alpha=0.8)
    else:
        vmin = min(legacy_lw); vmax = max(legacy_lw)
    scatter = plt.scatter(legacy_ll, legacy_Gs, c=legacy_lw, s=40, vmin=vmin, vmax=vmax, linestyle='None', alpha=0.8)
    plt.ylabel('\\textbf{G [pW/K]}')
    plt.tick_params(axis='y', which='both', right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    plt.ylim(8,1E3)
    # if len(bolotest.keys())>0:
        # plt.legend()
        # ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)   # turn x ticks off
    plt.xlabel('Leg L [$\mu$m]')
    plt.tick_params(axis='y', which='both', right=True)
    # plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
    # color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Leg Width [$\mu$m]')
    if save_figs: plt.savefig(plot_dir + 'legacydata' + plot_comments + '.png', dpi=300)

    # vs width
    plt.figure()
    if len(bolotest.keys())>0:   # plot bolotest 1b data and prediction
        vmin = min(np.append(legacy_ll, L_bolo)); vmax = max(np.append(legacy_ll, L_bolo))
        plt.scatter(w_bolo, bolotest['data']['ydata'][0], s=40, c=L_bolo, vmin=vmin, vmax=vmax, linestyle='None', cmap='plasma_r', alpha=0.8)
    else:
        vmin = min(legacy_ll); vmax = max(legacy_ll)
    scatter = plt.scatter(legacy_lw, legacy_Gs, c=legacy_ll, s=40, vmin=vmin, vmax=vmax, linestyle='None', cmap='plasma_r', alpha=0.8)
    plt.ylabel('\\textbf{G [pW/K]}')
    plt.tick_params(axis='y', which='both', right=True)
    plt.yscale('log')
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    # plt.ylim(8,1E3)
    plt.tick_params(axis='x', which='both', bottom=True, top=False)   # turn x ticks off
    plt.xlabel('Leg Width [$\mu$m]')
    plt.tick_params(axis='y', which='both', right=True)

    # color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Leg Length [$\mu$m]')

    La = legacy['geometry']['La']
    asc_legacy = ascale(legacy_ll, La)
    asc_bolo = ascale(L_bolo, La)

    # G/w vs width
    plt.figure()
    if len(bolotest.keys())>0:   # plot bolotest 1b data and prediction
        vmin = min(np.append(legacy_ll, L_bolo)); vmax = max(np.append(legacy_ll, L_bolo))
        plt.scatter(w_bolo, bolotest['data']['ydata'][0]/w_bolo/asc_bolo/0.4, s=40, c=L_bolo, vmin=vmin, vmax=vmax, linestyle='None', cmap='plasma_r', alpha=0.8)
    else:
        vmin = min(legacy_ll); vmax = max(legacy_ll)
    scatter = plt.scatter(legacy_lw, legacy_Gs/legacy_lw/asc_legacy, c=legacy_ll, s=40, vmin=vmin, vmax=vmax, linestyle='None', cmap='plasma_r', alpha=0.8)
    plt.ylabel('\\textbf{G/A$\\times$(1+L/La)')
    plt.tick_params(axis='y', which='both', right=True)
    plt.yscale('log')
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    # plt.ylim(8,1E3)
    plt.tick_params(axis='x', which='both', bottom=True, top=False)   # turn x ticks off
    plt.xlabel('Leg Width [$\mu$m]')
    cbar = plt.colorbar(scatter); cbar.set_label('Leg Length [$\mu$m]')

    plt.figure()
    if len(bolotest.keys())>0:   # plot bolotest 1b data and prediction
        vmin = min(np.append(legacy_lw, w_bolo)); vmax = max(np.append(legacy_lw, w_bolo))
        plt.scatter(L_bolo, bolotest['data']['ydata'][0]/w_bolo/0.4/asc_bolo, s=40, c=w_bolo, vmin=vmin, vmax=vmax, linestyle='None', alpha=0.8)
    else:
        vmin = min(legacy_lw); vmax = max(legacy_lw)
    scatter = plt.scatter(legacy_ll, legacy_Gs/legacy_lw/asc_legacy, c=legacy_lw, s=40, vmin=vmin, vmax=vmax, linestyle='None', alpha=0.8)
    plt.ylabel('\\textbf{G/A$\\times$(1+L/La)')
    plt.tick_params(axis='y', which='both', right=True)
    plt.yscale('log'); plt.xscale('log')
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    plt.tick_params(axis='x', which='both', bottom=True, top=False)   # turn x ticks off
    plt.xlabel('Leg Length [$\mu$m]')
    plt.tick_params(axis='y', which='both', right=True)
    cbar = plt.colorbar(scatter); cbar.set_label('Leg Width [$\mu$m]')

    if analyze_vlength:
        plt.figure()
        plt.plot(legacy_ll, legacy_Gs, '.')
        fit = np.polyfit(np.log10(legacy_ll), np.log10(legacy_Gs), 1)
        plt.plot(sorted(legacy_ll), 10**fit[1]*sorted(legacy_ll)**fit[0], '--')
        plt.title('Legacy Data - G$\\sim$L$^{Lscale}$'.format(Lscale='{'+str(round(fit[0], 3))+'}'))
        plt.yscale('log'); plt.xscale('log')
        plt.ylabel('G [pW/K]'); plt.xlabel('Leg Length [um]')
        plt.savefig(plot_dir+'legacydata_lengthscaling.png', dpi=300)

def predict_Glegacy(sim_dict, plot_opts, legacy, bolotest={}, fs=(9,6), dof=1):
    # predicts G for legacy TES data using alpha model, then plots prediction vs measurements (scaled to 170 mK)
    # legacy geometry and measurements are from Shannon's spreadsheet, then plots

    save_figs     = plot_opts['save_figs']
    title         = plot_opts['title']
    plot_comments = plot_opts['plot_comments']
    plot_dir      = plot_opts['plot_dir']
    plot_bolo1b   = plot_opts['plot_bolo1b']

    pred_wfit     = plot_opts.get('pred_wfit')
    plot_vwidth   = plot_opts.get('plot_vwidth')
    plot_bysubpop = plot_opts.get('plot_bysubpop')
    plot_wgrad    = plot_opts.get('plot_wgrad')
    show_percdiff = plot_opts.get('show_percdiff')

    an_opts     = sim_dict['an_opts']
    fn_comments = an_opts['fn_comments']
    sim_data    = sim_dict['sim']['fit_params']

    ### legacy data
    legacy_Gs = legacy['G170mK']
    legacy_ll = legacy['geometry']['ll']
    legacy_lw = legacy['geometry']['lw']

    if pred_wfit:   # use single set of fit parameters
        fit = sim_dict['fit']['fit_params']

        Gpred  = Gfrommodel(fit, an_opts, legacy)            ; sigma_Gpred  = np.zeros_like(Gpred)
        Gpred_S = Gfrommodel(fit, an_opts, legacy, layer='S'); sigma_GpredS = np.zeros_like(Gpred)
        Gpred_W = Gfrommodel(fit, an_opts, legacy, layer='W'); sigma_GpredW = np.zeros_like(Gpred)
        Gpred_I = Gfrommodel(fit, an_opts, legacy, layer='I'); sigma_GpredI = np.zeros_like(Gpred)
    else:   # use simulated set of fit parameters
        Gpreds  = Gfrommodel(sim_data, an_opts, legacy)   # predictions from each set of fit parameters [pW/K]
        GpredSs = Gfrommodel(sim_data, an_opts, legacy, layer='S')
        GpredWs = Gfrommodel(sim_data, an_opts, legacy, layer='W')
        GpredIs = Gfrommodel(sim_data, an_opts, legacy, layer='I')

        Gpred   = np.median(Gpreds, axis=0);  sigma_Gpred  = np.std(Gpreds, axis=0)   # predictions and error [pW/K]
        Gpred_S = np.median(GpredSs, axis=0); sigma_GpredS = np.std(GpredSs, axis=0)     # predictions and error of substrate layers [pW/K]
        Gpred_W = np.median(GpredWs, axis=0); sigma_GpredW = np.std(GpredWs, axis=0)   # predictions and error [pW/K]
        Gpred_I = np.median(GpredIs, axis=0); sigma_GpredI = np.std(GpredIs, axis=0)   # predictions and error [pW/K]

    # if an_opts['stack_N']:
    #     # bolo_SiNx = copy.deepcopy(legacy)
    #     # bolo_SiNx['geometry']['layer_ds'][4:11] = np.zeros(7)
    #     Gpreds_SiNx = Gfrommodel(sim_data, an_opts, legacy, layer='SiNx')
    #     Gpred_SiNx = np.median(Gpreds_SiNx, axis=0)
    #     Gpred_I = Gpred_I - Gpred_SiNx

    m1lab = 'Model'
    # m1lab = 'Constrained Model'

    if show_percdiff:   # plot residuals as a fraction of the measured value
        normres = (Gpred-legacy_Gs)/legacy_Gs*100 # normalized residuals [frac of measured value]
        rcolor = 'k'
        rylab = '\% Diff'
        rylim = [-75, 75]
    else:   # plot residuals as a fraction of prediction error
        normres = (legacy_Gs - Gpred)/(sigma_Gpred)   # normalized residuals [frac of prediction error]
        rcolor = 'darkorange'
        rylab = '\\textbf{N. Res.}'
        # rylim = [-2.5, 2.5]
        rylim = [-2.75, 2.75]
        # rylim = [-1.1, 1.1]
        rchisq_pred = np.sum(normres**2)/dof

    if 'low_lw' in legacy['geometry']:   # separate out width populations
        plot_bywidth = True
        low_lw = legacy['geometry']['low_lw']; mid_lw = legacy['geometry']['mid_lw']; high_lw = legacy['geometry']['high_lw']
    else:
        plot_bywidth = False

    # plot vs length or width?
    if plot_vwidth:
        legacy_x = legacy_lw
        ax_scale = 'linear'
        x_label  = 'Width [$\mu m$]'
        y_lim    = [0, 800]
        x_lim    = [8.5, 43]
        csq_loc  = [35, 75]

    else:
        legacy_x = legacy_ll
        ax_scale = 'log'
        # ax_scale = 'linear'
        x_label  = '\\textbf{Length [$\\boldsymbol{\mathrm{\mu m}}$]}'
        # y_lim    = [8, 1E4]
        y_lim    = [0.01, 1E4]
        x_lim    = [40, 1500]
        csq_loc  = [50, 12]

    plot_labs   = legacy['geometry']['plot_labs'] if 'plot_labs' in legacy['geometry'] else 'Legacy Data'

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])   # model vs data

    lorder = [0, 1]
    if plot_bolo1b:   # plot bolotest 1b data and prediction
        ll_1b = bolotest['geometry']['ll']
        lw_1b = bolotest['geometry']['lw']
        G_1b  = bolotest['data']['ydata'][0]

        vmin = min(np.append(legacy_lw, lw_1b)); vmax = max(np.append(legacy_lw, lw_1b))
        Gpred1bs = Gfrommodel(sim_data, an_opts, bolotest, layer='total')
        # Gpred1b_Ss = Gfrommodel(sim_data, an_opts, bolotest, layer='S')
        # Gpred1b_Ws = Gfrommodel(sim_data, an_opts, bolotest, layer='W')
        # Gpred1b_Is = Gfrommodel(sim_data, an_opts, bolotest, layer='I')

        Gpred1b = np.median(Gpred1bs); sigma_G1bpred = np.std(Gpred1bs)   # predictions and error [pW/K]
        # Gpred1b_S = np.median(Gpred1b_Ss); sigma_G1bpredU = np.std(Gpred1b_Ss)   # predictions and error [pW/K]
        # Gpred1b_W = np.median(Gpred1b_Ws); # sigma_G1bpred_W = np.std(Gpred1b_Ws)   # predictions and error [pW/K]
        # Gpred1b_I = np.median(Gpred1b_Is); # sigma_G1bpred_I = np.std(Gpred1b_Is)   # predictions and error [pW/K]

        normres_1b= (Gpred1b-G_1b)/G_1b*100 if show_percdiff else (G_1b - Gpred1b)/sigma_G1bpred   # normalized residuals [frac of measured value or frac of prediction error]

        # plt.scatter(ll_1b, G_1b, color='red', alpha=.8, label='Bolo 1', s=40)
        x_1b = lw_1b if plot_vwidth else ll_1b
        if plot_bywidth:
            plt.errorbar(ll_1b, G_1b, marker='o', label='Bolo 1 ($w=5$ $\mu$m)', markersize=7, color='red', linestyle='None', alpha=0.8)
        else:
            plt.errorbar(x_1b, G_1b, yerr=sigma_G1bpred, marker='o', label='Bolo 1', markersize=9, capsize=3, color='red', linestyle='None', alpha=0.8)
        plt.errorbar(x_1b, Gpred1b, yerr=sigma_G1bpred, marker='*', color='k', markersize=11, capsize=3, alpha=0.6)
        lorder = [0, 2, 1]
    else:
        vmin = min(legacy_lw); vmax = max(legacy_lw)

    if plot_bywidth:
        plt.plot(legacy_x[low_lw],  legacy_Gs[low_lw],  markersize=7, marker='o', label=plot_labs[0], linestyle='None')
        plt.plot(legacy_x[mid_lw],  legacy_Gs[mid_lw],  markersize=7, marker='o', label=plot_labs[1], linestyle='None')
        plt.plot(legacy_x[high_lw], legacy_Gs[high_lw], markersize=7, marker='o', label=plot_labs[2], linestyle='None')
        plt.errorbar(legacy_x, Gpred, yerr=sigma_Gpred,  marker='*', markersize=11, linestyle='None', color='k', label=m1lab, capsize=3, alpha=0.4)
        lorder = [0, 1, 2, 3]
    elif plot_bysubpop:
        sps = legacy['geometry']['subpops']
        for sp, pop in enumerate(sps):
            plt.plot(legacy_x[pop],  legacy_Gs[pop],  markersize=7, marker='o', label=plot_labs[sp], linestyle='None')
        lorder = np.concatenate([[0], np.arange(len(sps))+1])
        plt.errorbar(legacy_x, Gpred, yerr=sigma_Gpred,  marker='*', markersize=11, linestyle='None', color='k', label=m1lab, capsize=3, alpha=0.4)
    elif plot_wgrad:
        scatter = plt.scatter(legacy_x, legacy_Gs, c=legacy_lw, cmap='viridis', s=40, vmin=vmin, vmax=vmax, label='Legacy Data')

        # color mapped error bars
        norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
        mcolors = np.array([(mapper.to_rgba(lw)) for lw in legacy_lw])

        for x, Gp1, sig, color in zip(legacy_x, Gpred, sigma_Gpred, mcolors):
            plt.errorbar(x*1.07, Gp1, yerr=sig, marker='*', capsize=4, color='white', zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
            plt.errorbar(x*1.07, Gp1, yerr=sig, marker='*', capsize=4, color=color,   zorder=-1, linestyle='None', markeredgecolor='none', markersize=10, alpha=0.5, label=m1lab)

    else:
        plt.plot(legacy_x, legacy_Gs,                marker='o', markersize=9, linestyle='None', color='darkorange', label='Legacy Data')
        plt.errorbar(legacy_x, Gpred, yerr=sigma_Gpred,  marker='*', markersize=11, linestyle='None', color='k', label=m1lab, capsize=3, alpha=0.4)

    plt.plot(    legacy_x, Gpred_S, color='blue',       marker='d', markersize=7,  label='G$_\\text{S}$', linestyle='None')
    plt.plot(    legacy_x, Gpred_I, color='blueviolet', marker='s', markersize=5,  label='G$_\\text{I}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
    plt.plot(    legacy_x, Gpred_W, color='green',      marker='v', markersize=8,  label='G$_\\text{W}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
    lorder = [0, 1, 2, 3, 4]
    # if an_opts['stack_N']:
    #     plt.plot(legacy_x, Gpred_SiNx, color='r', marker='s', markersize=5,  label='G$_\\text{SiNx}$', linestyle='None', alpha=0.8)#, fillstyle='none', markeredgewidth=1.5)
    #     lorder = [0, 4, 1, 2, 3, 5]

    plt.ylabel('\\textbf{G [pW/K]}')
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # plt.tick_params(axis='y', which='both', right=True)
    plt.yscale(ax_scale); plt.xscale(ax_scale)
    plt.grid(linestyle = '--', which='both', linewidth=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder])

    # if not show_percdiff: plt.annotate('$\\boldsymbol{\\chi_\\nu^2}$ = '+str(round(rchisq_pred, 2)), csq_loc, bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off

    ### residuals
    ax2 = plt.subplot(gs[1], sharex=ax1); ax_xlim = ax2.get_xlim()   # residuals
    plt.axhline(0, color='k', alpha=0.5)
    if show_percdiff:
        if plot_wgrad:
            plt.scatter(legacy_x, normres, c=legacy_lw, cmap='viridis', vmin=vmin, vmax=vmax, s=80, marker='*')
        else:
            plt.scatter(legacy_x, normres, color='k', s=80, marker='*')
    elif plot_bywidth:
        plt.scatter(legacy_x[low_lw],  normres[low_lw],  s=40, marker='o')
        plt.scatter(legacy_x[mid_lw],  normres[mid_lw],  s=40, marker='o')
        plt.scatter(legacy_x[high_lw], normres[high_lw], s=40, marker='o')
        plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='gray', alpha=0.5)   # +/- 1 sigma
    elif plot_bysubpop:
        for pop in sps:
            plt.scatter(legacy_x[pop],  normres[pop],    s=40, marker='o')
        plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='gray', alpha=0.5)   # +/- 1 sigma
    elif plot_wgrad:
        plt.scatter(legacy_x, normres, c=legacy_lw, cmap='viridis', s=40, vmin=vmin, vmax=vmax, alpha=0.8)
        plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='gray', alpha=0.5)   # +/- 1 sigma
    else:
        plt.scatter(legacy_x, normres,              color='darkorange', s=80, alpha=0.6)
        plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='gray', alpha=0.5)   # +/- 1 sigma
    if plot_bolo1b: plt.scatter(x_1b, normres_1b,   color='red', s=80, marker='o', alpha=0.6)
    plt.xlabel(x_label); plt.xlim(ax_xlim)
    plt.ylabel(rylab)#; plt.ylim(rylim)
    # plt.tick_params(axis='y', which='both', right=True)
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
    plt.suptitle(title, y=0.92)

    if plot_wgrad:   # color bar
        cbar = fig.colorbar(scatter, ax=[ax1, ax2])
        cbar.set_label('Leg Width [$\mu$m]')

    if save_figs: plt.savefig(plot_dir + 'Gpred_legacy' + fn_comments + plot_comments + '.png', dpi=300)

    # save model predictions
    legacy['fit'] = {}
    legacy['fit']['Gpred'] = Gpred; legacy['fit']['sigma_Gpred'] = sigma_Gpred   # G(d0) prediction - final result [pW / K]
    legacy['fit']['Gpred_S'] = Gpred_S; legacy['fit']['sigma_GpredS'] = sigma_GpredS   # G(d0) prediction - substrate contribution [pW / K]
    legacy['fit']['Gpred_W'] = Gpred_W; legacy['fit']['sigma_GpredW'] = sigma_GpredW   # G(d0) prediction - Nb wiring layer contribution [pW / K]
    legacy['fit']['Gpred_I'] = Gpred_I; legacy['fit']['sigma_GpredI'] = sigma_GpredI   # G(d0) prediction - insulating nitride layer contribution [pW / K]

    return legacy

def predict_Glegacy_2models(sim_dict, plot_opts, legacy, bolotest={}, title='', fs=(9,8), plot_vwidth=False, show_percdiff=False, plot_wgrad=False):
    # predicts G for legacy TES data using alpha model, then plots prediction vs measurements (scaled to 170 mK)
    # legacy geometry and measurements are from Shannon's spreadsheet, then plots

    if title=='': title = sim_dict['an_opts'].get('title')
    save_figs = plot_opts['save_figs']
    plot_comments = plot_opts['plot_comments']
    plot_dir = plot_opts['plot_dir']
    plot_bolo1b = plot_opts['plot_bolo1b']

    an_opts = sim_dict['an_opts']
    fn_comments = an_opts['fn_comments']
    sim_data = sim_dict['sim']['fit_params']

    ### legacy data
    legacy_Gs = legacy['G170mK']
    legacy_ll = legacy['geometry']['ll']
    legacy_lw = legacy['geometry']['lw']

    # first model predictions
    Gpreds  = Gfrommodel(sim_data, an_opts, legacy)   # predictions from each set of fit parameters [pW/K]
    Gpred   = np.median(Gpreds, axis=0); sigma_Gpred = np.std(Gpreds, axis=0)   # predictions and error [pW/K]

    # second model predictions
    legacy2 = copy.deepcopy(legacy)
    legacy2['geometry']['La'] = legacy['geometry']['La2']

    Gpreds2  = Gfrommodel(sim_data, an_opts, legacy2)   # predictions from each set of fit parameters [pW/K]
    Gpred2   = np.median(Gpreds2, axis=0); sigma_Gpred2 = np.std(Gpreds2, axis=0)   # predictions and error [pW/K]
    normres2 = (legacy_Gs - Gpred2)/(sigma_Gpred2)
    # m1lab = '$L_{a,\\mathit{1}}$'; m2lab = '$L_{a,\\mathit{2}}$'
    m1lab = '$\ell_{\\mathit{1}}$'; m2lab = '$\ell_{\\mathit{2}}$'

    # type of residuals
    if show_percdiff:   # plot residuals as a fraction of the measured value
        normres = (Gpred-legacy_Gs)/legacy_Gs*100 # normalized residuals [frac of measured value]
        # rylab = '\% Diff'
        rylim = [-75, 75]
    else:   # plot residuals as a fraction of prediction error
        normres = (legacy_Gs - Gpred)/(sigma_Gpred)   # normalized residuals [frac of prediction error]
        rylim = [-2.75, 2.75]

    # plot vs length or width?
    if plot_vwidth:
        legacy_x = legacy_lw
        ax_scale = 'linear'
        x_label  = '\\textbf{Width [$\\boldsymbol{\mathrm{\mu m}}$]}'
        y_lim    = [0, 1300]
        x_lim    = [8.5, 43]

        # offset
        Gpred      = Gpred     + 500
        legacy_Gs2 = legacy_Gs
        legacy_Gs  = legacy_Gs + 500

    else:
        legacy_x = legacy_ll
        ax_scale = 'log'
        x_label  = '\\textbf{Length [$\\boldsymbol{\mu m}$]}'
        y_lim    = [8, 1e3]
        x_lim    = [45, 1500]
        csq_loc  = [50, 12]

        # offset
        legacy_Gs2  = legacy_Gs

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    ax1 = plt.subplot(gs[0])   # model vs data

    # plot data
    if plot_wgrad:   # color data points by width gradient
        vmin = min(legacy_lw); vmax = max(legacy_lw)
        scatter = plt.scatter(legacy_x, legacy_Gs,  c=legacy_lw, cmap='viridis', s=90, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.5, label='Legacy Data')
        # plt.scatter(          legacy_x, legacy_Gs2, c=legacy_lw, cmap='viridis', s=70, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.5)
    else:
        plt.plot(legacy_x, legacy_Gs,  marker='o', linestyle='None', fillstyle='none', color='k', markersize=10, markeredgewidth=1.5, alpha=1, label='Legacy Data')
        plt.plot(legacy_x, legacy_Gs2, marker='o', linestyle='None', fillstyle='none', color='k', markersize=10, markeredgewidth=1.5, alpha=1)

    # plot predictions
    if plot_wgrad:
        # color mapped error bars
        norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
        mcolors = np.array([(mapper.to_rgba(lw)) for lw in legacy_lw])

        for x, Gp1, Gp2, sig, sig2, color in zip(legacy_x, Gpred, Gpred2, sigma_Gpred, sigma_Gpred2, mcolors):
            # model 1
            plt.errorbar(x*1.07, Gp1, yerr=sig, marker='D', capsize=4, color='white', zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
            plt.errorbar(x*1.07, Gp1, yerr=sig, marker='D', capsize=4, color=color,   zorder=-1, linestyle='None', markeredgecolor='none', markersize=10, alpha=0.5, label=m1lab)
            # model 2
            plt.errorbar(x*1.14, Gp2, yerr=sig2, marker='^', capsize=4, color='white', zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
            plt.errorbar(x*1.14, Gp2, yerr=sig2, marker='^', capsize=4, color=color,   zorder=-1, linestyle='None', markeredgecolor='none', markersize=10, alpha=0.5, label=m2lab)
    else:
        plt.errorbar(legacy_x*1.1, Gpred,  yerr=sigma_Gpred,  marker='D', capsize=7, color=color1, linestyle='None', markersize=9,  label=m1lab, zorder=0, alpha=0.5)
        # plt.errorbar(legacy_x*1.1, Gpred2, yerr=sigma_Gpred2, marker='^', capsize=7, color=color2, linestyle='None', markersize=10, label=m2lab, zorder=0, alpha=0.5)
    if plot_bolo1b:   # plot bolotest 1b data and prediction
        ll_1b = bolotest['geometry']['ll']
        lw_1b = bolotest['geometry']['lw']
        G_1b  = bolotest['data']['ydata'][0]

        vmin = min(np.append(legacy_lw, lw_1b)); vmax = max(np.append(legacy_lw, lw_1b))
        Gpred1bs = Gfrommodel(sim_data, an_opts, bolotest, layer='total')
        Gpred1b = np.median(Gpred1bs); sigma_G1bpred = np.std(Gpred1bs)   # predictions and error [pW/K]
        normres_1b= (Gpred1b-G_1b)/G_1b*100 if show_percdiff else (G_1b - Gpred1b)/sigma_G1bpred   # normalized residuals [frac of measured value or frac of prediction error]

        # plt.scatter(ll_1b, G_1b, color='red', alpha=.8, label='Bolo 1', s=40)
        x_1b = lw_1b if plot_vwidth else ll_1b
        plt.errorbar(x_1b, G_1b, marker='o', label='Bolo 1', markersize=10, capsize=3, color='red', linestyle='None', markeredgecolor='none', alpha=0.7)

        color_1b =mapper.to_rgba(lw_1b)
        plt.errorbar(x_1b*1.07, G_1b, yerr=sigma_G1bpred, marker='D', capsize=4, color='white',  zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
        plt.errorbar(x_1b*1.07, G_1b, yerr=sigma_G1bpred, marker='D', capsize=4, color=color_1b, zorder=-1, linestyle='None', markeredgecolor='none', markersize=10, alpha=0.5)
        # model 2
        # plt.errorbar(x_1b*1.14, G_1b, yerr=sig2, marker='^', capsize=4, color='white', zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
        plt.errorbar(x_1b*1.14, G_1b, yerr=sigma_G1bpred, marker='^', capsize=4, color='white',    zorder=-1, linestyle='None', markeredgecolor='none', markersize=10)
        plt.errorbar(x_1b*1.14, G_1b, yerr=sigma_G1bpred, marker='^', capsize=4, color=color_1b,   zorder=-1, linestyle='None', markeredgecolor='none', markersize=10, alpha=0.5)
         # plt.errorbar(x_1b, Gpred1b, yerr=sigma_G1bpred, marker='*', color='k', markersize=13, capsize=3, alpha=0.5)
    plt.ylabel('\\textbf{G [pW/K]}', labelpad=15)
    plt.xlim(x_lim); plt.ylim(y_lim)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
    plt.yscale(ax_scale); plt.xscale(ax_scale)
    plt.grid(linestyle = '--', which='both', linewidth=0.5)

    # legend
    # lorder = [0, -1, 1, 2]
    lorder = [0, 1, 2]
    # lorder = [0, 1]
    # lorder = [0]
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder])

    ### residuals
    # shared y label
    axr = fig.add_subplot(gs[1:3, 0])
    # axr = fig.add_subplot(gs[1:2, 0])
    axr.set_ylabel('\\textbf{N. Res.}', labelpad=17)
    # axr.set_ylabel('\\textbf{N. Res.}', labelpad=35)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    ax2 = plt.subplot(gs[1], sharex=ax1); ax_xlim = ax2.get_xlim()   # residuals
    plt.axhline(0, color='k', alpha=0.5)
    plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='k', alpha=0.2)   # +/- 1 sigma
    if plot_wgrad:   # color data points by width gradient
        plt.scatter(legacy_x, normres, c=legacy_lw, cmap='viridis', s=90, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.5)
    else:
        plt.scatter(legacy_x, normres, color='k', s=80, facecolors='none', linewidths=1.5)
    if plot_bolo1b: plt.scatter(x_1b, normres_1b,   color='red', s=90, marker='o', edgecolor='none', alpha=0.7)
    plt.xlim(ax_xlim)
    # plt.ylabel('Model 1', fontsize=16, labelpad=-7)
    plt.ylabel('$\ell_{\\mathit{1}}$', fontsize=18, labelpad=-8)
    plt.ylim(rylim)
    # plt.tick_params(axis='y', which='both', right=True)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
    plt.grid(linestyle = '--', which='both', linewidth=0.5)
    plt.subplots_adjust(hspace=0.075)   # merge to share one x axis

    ax3 = plt.subplot(gs[2], sharex=ax1); ax_xlim = ax2.get_xlim()   # residuals
    plt.axhline(0, color='k', alpha=0.5)
    plt.fill_between((0, max(legacy_x)*5), -1, 1, facecolor='k', alpha=0.2)   # +/- 1 sigma
    if plot_wgrad:   # color data points by width gradient
        plt.scatter(legacy_x, normres2, c=legacy_lw, cmap='viridis', s=90, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.5)
    else:
        plt.scatter(legacy_x, normres2, color='k', s=80, facecolors='none', linewidths=1.5)
    if plot_bolo1b: plt.scatter(x_1b, normres_1b,   color='red', s=90, marker='o', edgecolor='none', alpha=0.7)
    plt.xlabel(x_label); plt.xlim(ax_xlim)
    # plt.ylabel('Model 2', fontsize=16, labelpad=-7)
    plt.ylabel('$\ell_{\\mathit{2}}$', fontsize=18, labelpad=-8)
    plt.ylim(rylim)
    # plt.tick_params(axis='y', which='both', right=True)
    plt.grid(linestyle = '--', which='both', linewidth=0.5)

    plt.subplots_adjust(hspace=0.075, wspace=0.25, left=0.09, right=0.94)   # merge to share one x axis

    if plot_wgrad:   # color bar
        cbar = fig.colorbar(scatter, ax=[ax1, ax2, ax3, axr], aspect=30, fraction=0.03)
        # cbar = fig.colorbar(scatter, ax=[ax1, ax2, axr], aspect=25, fraction=0.03)
        # cbar = fig.colorbar(scatter, ax=[ax1], aspect=20, fraction=0.03)
        cbar.set_label('\\textbf{Leg Width [$\\boldsymbol{\mu m}$]}', fontsize=16)

    if save_figs: plt.savefig(plot_dir + 'Gpred_legacy' + fn_comments + plot_comments + '.png', dpi=300)

    return legacy

### acoustic length analysis
def Geff(Gb, L, La, d, da):
    # model L and d scaling in 3D diffuse to ballistic transition
    # G_diff = G_ball * La/L * d/da
    # G_effective = (G_ball^-1 + G_diff^-1)^-1
    return Gb * (1 + L/La * da/d)**(-1)

def bolo_subpop(bolo, sp_inds):
    # create sub-population bolo dictionary

    bolo_sp = copy.deepcopy(bolo)
    bolo_sp['geometry']['ll']      = bolo['geometry']['ll'][sp_inds]
    bolo_sp['geometry']['lw']      = bolo['geometry']['lw'][sp_inds]
    bolo_sp['geometry']['dsub']    = bolo['geometry']['dsub'][sp_inds]
    bolo_sp['geometry']['w1w']     = bolo['geometry']['w1w'][sp_inds]
    bolo_sp['geometry']['w2w']     = bolo['geometry']['w2w'][sp_inds]
    bolo_sp['geometry']['dW1']     = bolo['geometry']['dW1'][sp_inds]
    bolo_sp['geometry']['dW2']     = bolo['geometry']['dW2'][sp_inds]
    bolo_sp['geometry']['dI1']     = bolo['geometry']['dI1'][sp_inds]
    bolo_sp['geometry']['dI2']     = bolo['geometry']['dI2'][sp_inds]
    bolo_sp['geometry']['A']       = bolo['geometry']['A'][sp_inds]
    bolo_sp['geometry']['La']      = bolo['geometry']['La'][sp_inds]

    bolo_sp['Tc']                  = bolo['Tc'][sp_inds]
    bolo_sp['n']                   = bolo['n'][sp_inds]
    bolo_sp['GTc']                 = bolo['GTc'][sp_inds]
    bolo_sp['G170mK']              = bolo['G170mK'][sp_inds]

    return bolo_sp

def ell_wLratio(w, ell_i, L, Pd=1):
    x = np.min(np.array([ell_i, L]), axis=0)
    return w/(np.pi*Pd) * (1 - np.log(w/(2*Pd*x)))

def L_acoust(geom, params, Pd=1, elli_term=True):

    L, w, d = geom
    A, B, C, D, E, ell_i = np.append(params[:6], [np.inf]*6)[:6]

    ell_i2 = ell_i if elli_term else np.inf
    return (A/d + (Pd * B)/w + C/ell_wLratio(w, np.inf*np.ones_like(L), L) + D/ell_wLratio(w, ell_i*np.ones_like(L), np.inf*np.ones_like(L))  + E/ell_wLratio(w, ell_i*np.ones_like(L), L) + 1/ell_i2)**(-1)
    # return (A/d + (Pd * B)/w + C/ell_wLratio(w, np.inf*np.ones_like(L), L) + D/ell_wLratio(w, ell_i*np.ones_like(L), np.inf*np.ones_like(L))  + E/ell_wLratio(w, ell_i*np.ones_like(L), L) + 0)**(-1)   # no 1/ell_i term

def GvsLa_chisq(params, args, elli_term=True):
    # chi-squared value for fitting acoustic scaling parameters to legacy data

    sim_dict, bolo = args
    sim_data = sim_dict['sim']['fit_params']
    an_opts  = sim_dict['an_opts']

    Gdata = bolo['G170mK']
    ds    = bolo['geometry']['dsub']
    lws   = bolo['geometry']['lw']
    lls   = bolo['geometry']['ll']

    fit_bolo = copy.deepcopy(bolo)   # input iteration-specific layer thicknesses and G_bolotest data

    fit_bolo['geometry']['acoustic_Lscale'] = True
    fit_bolo['geometry']['La'] = L_acoust(np.array([lls, lws, ds]), params, elli_term=elli_term)    # overwrite acoustic length
    # fit_bolo['geometry']['La'] = bolo['geometry']['La']    # overwrite acoustic length

    Gpreds = Gfrommodel(sim_data, an_opts, fit_bolo)
    Gpred = np.median(Gpreds, axis=0); sigma_Gpred = np.std(Gpreds, axis=0)     # predictions and error [pW/K]

    La_chisqvals = (Gpred-Gdata)**2/sigma_Gpred**2

    return np.sum(La_chisqvals)

def GvsLa_chisq_2parts(params, args, elli_term=True):
    # chi-squared value for fitting acoustic scaling parameters to legacy data

    sim_dict, bolo = args
    sim_data = sim_dict['sim']['fit_params']
    an_opts  = sim_dict['an_opts']

    Gdata = bolo['G170mK']

    fit_bolo1 = copy.deepcopy(bolo)   # input iteration-specific layer thicknesses and G_bolotest data
    fit_bolo2 = copy.deepcopy(bolo)   # input iteration-specific layer thicknesses and G_bolotest data


    # separate microstrip an nitride stacks to fit for two different Las
    fit_bolo2['geometry']['lw'] = fit_bolo1['geometry']['lw'] - fit_bolo1['geometry']['w1w']
    fit_bolo2['geometry']['w1w'] = 0; fit_bolo2['geometry']['w2w'] = 0
    fit_bolo1['geometry']['lw'] = fit_bolo1['geometry']['w1w']

    ds1    = fit_bolo1['geometry']['dsub']
    lws1   = fit_bolo1['geometry']['lw']
    lls1   = fit_bolo1['geometry']['ll']

    ds2    = fit_bolo2['geometry']['dsub']
    lws2   = fit_bolo2['geometry']['lw']
    lls2   = fit_bolo2['geometry']['ll']

    fit_bolo1['geometry']['acoustic_Lscale'] = True
    fit_bolo1['geometry']['La'] = params[0]    # overwrite acoustic length
    # fit_bolo['geometry']['La'] = bolo['geometry']['La']    # overwrite acoustic length

    fit_bolo2['geometry']['acoustic_Lscale'] = True
    fit_bolo2['geometry']['La'] = L_acoust(np.array([lls2, lws2, ds2]), np.array(params[1:]), elli_term=elli_term)    # overwrite acoustic length
    # fit_bolo['geometry']['La'] = bolo['geometry']['La']    # overwrite acoustic length

    # Gpreds = (1/Gfrommodel(sim_data, an_opts, fit_bolo1) + 1/Gfrommodel(sim_data, an_opts, fit_bolo2))**(-1)
    Gpreds = Gfrommodel(sim_data, an_opts, fit_bolo1) + Gfrommodel(sim_data, an_opts, fit_bolo2)
    Gpred = np.median(Gpreds, axis=0); sigma_Gpred = np.std(Gpreds, axis=0)     # predictions and error [pW/K]

    La_chisqvals = (Gpred-Gdata)**2/sigma_Gpred**2

    return np.sum(La_chisqvals)

def G_GbandLa(geom, *params, w1w=8, alpha=1, Pd=1):   # calculate mfp based on geometry
    # model width dependence of G given G_fullstack and G0_Istack are shared components of Gballistic
    # mfp is a parallel sum of other mfp's and a mfp that depends on w

    L, w, d = geom

    # parse arbitrary number of parameters
    Gb_FS = params[0]; Gb_IS = params[1]
    La_params = np.append(params[2:7], [np.inf]*5)[:5]

    # acoustic length
    La = L_acoust(geom, La_params, Pd=Pd)

    # Gballistic
    Gb = (Gb_FS*w1w/10 + Gb_IS*(w-w1w)/10) * (d/2.45)**alpha   # Gb for full microstrip section and for nitride stack section

    return Gb / (1 + L/La)   # Gb is ballistic G for a 5-um-wide leg

def fit_acoustic_length(sim_dict, plot_opts, legacy, bolotest={}, fs=(9,6), plot_fit=True, num_params=5):

    save_figs     = plot_opts['save_figs']
    plot_comments = plot_opts['plot_comments']
    plot_dir      = plot_opts['plot_dir']
    plot_bolo1b   = plot_opts['plot_bolo1b']

    an_opts     = sim_dict['an_opts']
    fn_comments = an_opts['fn_comments']
    sim_data    = sim_dict['sim']['fit_params']

    ### legacy data
    legacy_Gs = legacy['G170mK']
    legacy_ll = legacy['geometry']['ll']
    legacy_lw = legacy['geometry']['lw']
    legacy_d  = legacy['geometry']['dsub']

    # La0 = 150;   # initial guess [um]
    # num_params = 2 + 1   # number of free parameters to fit
    p0_La = np.array([10]*num_params)
    bounds_La = [([0]*num_params), ([np.inf]*num_params)]

    # afit_result = minimize(GvsLa_chisq, p0_La, args=[sim_dict, legacy], bounds=bounds_La)   # minimize chi-squared function with this iteration's G_TES values and film thicknesses
    # afit_result = minimize(GvsLa_chisq, p0_La, args=[sim_dict, legacy])   # minimize chi-squared function with this iteration's G_TES values and film thicknesses
    afit_result = minimize(GvsLa_chisq_2parts, p0_La, args=[sim_dict, legacy])   # minimize chi-squared function with this iteration's G_TES values and film thicknesses

    La_fit = afit_result['x']
    A_fit, B_fit, C_fit, D_fit, elli_fit = np.append(La_fit[:5], [0]*5)[:5]
    # perr = np.append(np.sqrt(np.diag(pcov_all))[:7], [0]*7)[:7]
    perr = np.append(0*La_fit[:5], [0]*5)[:5]

    # La_params_all, pcov_all = curve_fit(G_fromwidth, all_geom, legacyGs_all, p0_Gw, bounds=bounds_Gw)
    # GbFS_all, GbIS_all, A_all, B_all, C_all, D_all, elli_all = np.append(La_params_all[:7], [0]*7)[:7]

    legacy_fit = copy.deepcopy(legacy)   # input iteration-specific layer thicknesses and G_bolotest data
    legacy_fit['geometry']['acoustic_Lscale'] = True

    # legacy_fit['geometry']['La'] = La_fit    # overwrite acoustic length
    legacy_fit['geometry']['La'] = L_acoust(np.array([legacy_ll, legacy_lw, legacy_d]), La_fit)    # overwrite acoustic length
    # print('La = {} um'.format(La_fit))
    # legacy_fit['geometry']['La'] = La_fit * legacy_fit['geometry']['lw']   # overwrite acoustic length
    # print('La = {} * w'.format(round(La_fit, 1)))
    print('A = {} +/- {}'.format(round(A_fit, 3), round(perr[0], 3)))
    print('B = {} +/- {}'.format(round(B_fit, 3), round(perr[1], 3)))
    print('C = {} +/- {}'.format(round(C_fit, 3), round(perr[2], 3)))
    print('D = {} +/- {}'.format(round(D_fit, 3), round(perr[3], 3)))
    print('ell_i = {} +/- {} um'.format(round(elli_fit, 2), round(perr[4], 2)))
    # print('Chi-squared = {} '.format(round(rchisq_pred,2)))
    if plot_fit:

        Gpreds  = Gfrommodel(sim_data, an_opts, legacy_fit)   # predictions from each set of fit parameters [pW/K]
        Gpred   = np.median(Gpreds,  axis=0); sigma_Gpred = np.std(Gpreds, axis=0)   # predictions and error [pW/K]
        normres = (legacy_Gs - Gpred)/sigma_Gpred   # normalized residuals [frac of prediction error]
        # dof = len(normres)-5 if acoust_d else len(normres)-2
        dof         = len(normres)-2
        rchisq_pred = np.sum(normres**2)/dof

        plt.figure(figsize=fs)
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax1 = plt.subplot(gs[0])   # model vs data
        plt.scatter( legacy_ll, legacy_Gs, color='darkorange', alpha=.8, label='Legacy Data', s=35)
        plt.errorbar(legacy_ll, Gpred, yerr=sigma_Gpred, color='k', marker='*', label='G$_\text{pred}$', capsize=2, linestyle='None', markersize=7)
        plt.ylabel('\\textbf{G [pW/K]}')
        plt.tick_params(axis='y', which='both', right=True)
        plt.yscale('log'); plt.xscale('log')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
        plt.ylabel('G [pW/K]')
        plt.annotate('$\\boldsymbol{\\chi_\\nu^2}$ = '+str(round(rchisq_pred, 1)), (50, 20), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))

        if plot_bolo1b:   # plot bolotest 1b data and prediction
            ll_1b = bolotest['geometry']['ll']
            G_1b  = bolotest['data']['ydata'][0]; bolo1b_sigma = bolotest['data']['sigma'][0]

            Gpred1bs = Gfrommodel(sim_data, an_opts, bolotest, layer='total')
            Gpred1b  = np.median(Gpred1bs); sigma_G1bpred = np.std(Gpred1bs)   # predictions and error [pW/K]

            plt.errorbar(ll_1b, G_1b, yerr=bolo1b_sigma,     marker='o', color='red', label='Bolo 1', capsize=2, linestyle='None', markersize=7)
            plt.errorbar(ll_1b, Gpred1b, yerr=sigma_G1bpred, marker='*', color='k', markersize=7, capsize=2)

        ax2 = plt.subplot(gs[1], sharex=ax1); ax_xlim = ax2.get_xlim()   # residuals
        plt.axhline(0, color='k', alpha=0.7)
        plt.fill_between((0, max(legacy_ll)*5), -1, 1, facecolor='gray', alpha=0.5)   # +/- 1 sigma
        plt.scatter(legacy_ll, normres, color='darkorange', s=35, alpha=0.8)
        if plot_bolo1b: plt.scatter(ll_1b, (G_1b-Gpred1b)/sigma_G1bpred, color='red', alpha=0.8, s=35)
        plt.xlabel('L [$\mu$m]')
        plt.ylabel('\\textbf{Norm. Res.}')
        plt.xlim(ax_xlim)
        plt.ylim(-2, 2)
        plt.tick_params(axis='y', which='both', right=True)
        plt.subplots_adjust(hspace=0.075)   # merge to share one x axis

        if save_figs: plt.savefig(plot_dir+'legacydata_acousticscale_fit'+fn_comments+plot_comments+'.png', dpi=300)

    return La_fit, legacy_fit

def plot_GandTFNEP(sim_dict, an_opts, plot_opts, bolo):

    # plots GTES and thermal fluctuation noise equivalent power predictions from the alpha model vs leg width
    # G values in pW/K, leg dimensions in um, temperature in K
    # can turn off/on G_TES errors and NEP errors
    # plots GTES and TFNEP vs leg area/length

    sim_data = sim_dict['sim']['fit_params']
    fn_comments = an_opts['fn_comments']
    lwidths = bolo['geometry']['lw']
    Tc = bolo['Tc']

    # plotting options
    save_figs = plot_opts['save_figs']
    plot_comments = plot_opts['plot_comments']
    plot_dir = plot_opts['plot_dir']
    plot_G = plot_opts['plot_G']; plot_Gerr = plot_opts['plot_Gerr']
    plot_NEPsq = plot_opts['plot_NEPsq']
    plot_NEPerr = plot_opts['plot_NEPerr']
    Glims = plot_opts.get('Glims'); NEPlims = plot_opts.get('NEPlims')

    # calculate predicted G's and NEP's from model
    Gpreds    = Gfrommodel(sim_data, an_opts, bolo)   # predictions from each set of fit parameters [pW/K]
    GpredSs   = Gfrommodel(sim_data, an_opts, bolo, layer='S')
    GpredW1s  = GpredSs + Gfrommodel(sim_data, an_opts, bolo, layer='W1')
    NEPs_full = TFNEP(Tc, Gpreds*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEPs_S    = TFNEP(Tc, GpredSs*1E-12)*1E18
    NEPs_W1   = TFNEP(Tc, GpredW1s*1E-12)*1E18

    NEP_full = np.median(NEPs_full, axis=0); NEPerr_full = np.std(NEPs_full, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEP_S    = np.median(NEPs_S, axis=0); NEPerr_S = np.std(NEPs_S, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEP_W1   = np.median(NEPs_W1, axis=0); NEPerr_W1 = np.std(NEPs_W1, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK

    # predicted G vs substrate width
    if plot_G:

        G_full = np.median(Gpreds, axis=0); Gerr_full = np.std(Gpreds, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
        G_S    = np.median(GpredSs, axis=0); Gerr_S = np.std(GpredSs, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
        G_W1   = np.median(GpredW1s, axis=0); Gerr_W1 = np.std(GpredW1s, axis=0)   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK

        fig, ax1 = plt.subplots()
        ax1.plot(lwidths, G_full, color='rebeccapurple', label='G$_\\text{TES}$, Microstrip', alpha=0.8)
        ax1.plot(lwidths, G_W1, color='green', label='G$_\\text{TES}$, 200nm Nb', alpha=0.8)
        ax1.plot(lwidths, G_S, color='royalblue', label='G$_\\text{TES}$, Bare S', alpha=0.8)
        if plot_Gerr:
            plt.fill_between(lwidths, G_full-Gerr_full, G_full+Gerr_full, facecolor='mediumpurple', alpha=0.2)   # error
            plt.fill_between(lwidths, G_W1-Gerr_W1, G_W1+Gerr_W1, facecolor='limegreen', alpha=0.2)   # error
            plt.fill_between(lwidths, G_S-Gerr_S, G_S+Gerr_S, facecolor='cornflowerblue', alpha=0.2)   # error
        ax1.set_xlabel('Leg Width [$\mu$m]')
        ax1.set_ylabel('G$_\\text{TES}$(170mK) [pW/K]')
        # if len(Glims)>0: ax1.set_ylim(ymin=Glims[0], ymax=Glims[1])   # user specified G y-axis limits

        # TFNEP vs substrate width
        ax2 = ax1.twinx()
        ax2.plot(lwidths, NEP_full, '--', color='rebeccapurple', label='NEP')   # this varies as G^1/2
        ax2.plot(lwidths, NEP_W1, '--', color='green', label='NEP')   # this varies as G^1/2
        ax2.plot(lwidths, NEP_S, '--', color='royalblue', label='NEP')   # this varies as G^1/2
        if plot_NEPerr:
            plt.fill_between(lwidths, NEP_full-NEPerr_full, NEP_full+NEPerr_full, facecolor='rebeccapurple', alpha=0.2)   # error
            plt.fill_between(lwidths, NEP_W1-NEPerr_W1, NEP_W1+NEPerr_W1, facecolor='green', alpha=0.2)   # error
            plt.fill_between(lwidths, NEP_S-NEPerr_S, NEP_S+NEPerr_S, facecolor='royalblue', alpha=0.2)   # error
        # if len(NEPlims)>0: ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1])   # user specified TFNEP y-axis limits
        ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')

        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='12', ncol=2)
        plt.tick_params(axis='y', which='both', right=True)
        plt.tight_layout()
        if save_figs: plt.savefig(plot_dir + 'NEPandG' + fn_comments + plot_comments + '.png', dpi=300)

    elif plot_NEPsq:   # plot NEP^2

        NEPfull_sq = (NEP_full)**2; NEPerrfull_sq = NEPerr_full**2   # aW^2 / Hz; Kenyan 2006 measured (1E-17)^2 for a TES with comparable G at 170 mK
        NEPW1_sq = NEP_W1**2; NEPerrW1_sq = NEPerr_W1**2   # aW^2 / Hz; Kenyan 2006 measured (1E-17)^2 for a TES with comparable G at 170 mK
        NEPS_sq = NEP_S**2; NEPerrS_sq = NEPerr_S**2   # aW^2 / Hz; Kenyan 2006 measured (1E-17)^2 for a TES with comparable G at 170 mK

        # Compare to SPIDER NEP?
        Psat_sp = 3E-12   # W; SPIDER 280 GHz target Psat; Hubmayr 2019
        # NEP_spider = 17E-18   # W/rt(Hz), detector NEP of 90 and 150 GHz SPIDER bolos, MaSskopf et al 2018
        # NEP_FIR = 1E-20   # W/rt(Hz), NEP necessary to do background limited FIR spectroscopy, Kenyon et al 2006
        # NEP_Psat = TFNEP(0.170, Psat_sp/(0.170-0.100), Tb=0.100)*1E18   # NEP for target SPIDER Psat at 100 mK; aW/rt(Hz); G(Tc)~Psat/(Tc-Tb)

        plt.figure(figsize=(5.5,5))
        plt.plot(lwidths, NEPfull_sq, color='rebeccapurple', label='S-W1-I1-W2-I2')
        plt.plot(lwidths, NEPW1_sq, color='green', label='S-W1')
        plt.plot(lwidths, NEPS_sq, color='royalblue', label='S')
        if plot_NEPerr:
            plt.fill_between(lwidths, NEPfull_sq-NEPerrfull_sq, NEPfull_sq+NEPerrfull_sq, facecolor='rebeccapurple', alpha=0.3)   # error
            plt.fill_between(lwidths, NEPW1_sq-NEPerrW1_sq, NEPW1_sq+NEPerrW1_sq, facecolor='green', alpha=0.3)   # error
            plt.fill_between(lwidths, NEPS_sq-NEPerrS_sq, NEPS_sq+NEPerrS_sq, facecolor='royalblue', alpha=0.3)   # error
        # if len(NEPlims)>0: plt.ylim(ymin=NEPlims[0], ymax=NEPlims[1])   # user specified TFNEP y-axis limits
        plt.xlim(min(lwidths)-max(lwidths)*0.02, max(lwidths)*1.02)
        plt.ylabel('(NEP$_\\text{TF}/\\gamma)^2$\ \  [aW$^2$/Hz]', fontsize=14)
        plt.xlabel('Leg Width [$\mu$m]', fontsize=14)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)
        plt.tick_params(axis='y', which='both', right=True)
        plt.tight_layout()
        if save_figs: plt.savefig(plot_dir + 'NEPsq' + fn_comments + plot_comments + '.png', dpi=500)

    else:   # plot NEP

        # Compare to SPIDER NEP?
        Psat_sp = 3E-12   # W; SPIDER 280 GHz target Psat; Hubmayr 2019
        NEP_spider = 17E-18   # W/rt(Hz), detector NEP of 90 and 150 GHz SPIDER bolos, MaSskopf et al 2018
        NEP_FIR = 1E-20   # W/rt(Hz), NEP necessary to do background limited FIR spectroscopy, Kenyon et al 2006
        NEP_Psat = TFNEP(0.170, Psat_sp/(0.170-0.100), Tb=0.100)*1E18   # NEP for target SPIDER Psat at 100 mK; aW/rt(Hz); G(Tc)~Psat/(Tc-Tb)

        # plt.figure(figsize=(5.5,5))
        plt.figure(figsize=(7,5))
        plt.plot(lwidths, NEP_full, color='rebeccapurple', label='S-W1-I1-W2-I2')
        plt.plot(lwidths, NEP_W1, color='green', label='S-W1')
        plt.plot(lwidths, NEP_S, color='royalblue', label='S')

        if plot_NEPerr:
            plt.fill_between(lwidths, (NEP_full+NEPerr_full), (NEP_full-NEPerr_full), facecolor='rebeccapurple', alpha=0.3)   # error
            plt.fill_between(lwidths, (NEP_W1+NEPerr_W1), (NEP_W1-NEPerr_W1), facecolor='green', alpha=0.3)   # error
            plt.fill_between(lwidths, (NEP_S+NEPerr_S), (NEP_S-NEPerr_S), facecolor='royalblue', alpha=0.3)   # error
        if NEPlims: plt.ylim(ymin=NEPlims[0], ymax=NEPlims[1])   # user specified TFNEP y-axis limits
        # plt.tick_params(axis='y', which='both', right=True)
        plt.ylabel('\\textbf{NEP$_\\textbf{TF}\\boldsymbol{/\\sqrt{\\gamma}}$\ \ \ [aW/$\\boldsymbol{\\sqrt{Hz}}$]}', fontsize=16)
        plt.xlabel('\\textbf{Leg Width [$\\boldsymbol{\mu m}$]}', fontsize=16) # $\\boldsymbol{\mathrm{\mu m}}$
        plt.xlim(min(lwidths), max(lwidths)); plt.ylim(0, 12)
        plt.legend(loc='upper left', fontsize=14)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)
        plt.tight_layout()

        if save_figs: plt.savefig(plot_dir + 'NEP' + fn_comments + plot_comments + '.png', dpi=500)

        return np.array([NEP_full, NEPerr_full])   # aW/rt(Hz) for bolo with microstrip on all four legs

def predict_PsatSO(sim_dict, an_opts, bolo, Tc=0.160, Tbath=0.100, n=3., plot_predsvdata=False,
                   title='', print_results=False, pred_wfit=False, plot_dir='./', save_figs=False, fn_comments='', linscale=False, lsind=1):
    """
    Uses alpha model to predict saturation power for a given bolometer leg geometry, Tc, and Tbath

    INPUT:
    sim_dict = simulated fit parameters (i.e., the model)
    dsub = substrate thickness [um], including nitride and oxide layers
    lw = leg width [um]
    ll = leg length [um]
    Lscale = how G/Psat scale with 1/leg length L, i.e., 1/L^(Lscale). For linear scaling, Lscale = 1.0
    plot_predsvdata = option to plot Psat predictions vs data. This will require inputing data (Psat_data) and leg area over leg length (AoL) in um.
    print_results = option to print Psat predictions with input geometry to screen

    RETURNS:
    Psat_pred = Psat predictions in [pW/K] at Tc, assuming bolometers with four legs
    sigma_Psat = 1 sigma error bar on Psat predictions [pW/K] --- not currently calculated
    """
    dsub = bolo['geometry']['dsub']
    lw   = bolo['geometry']['lw']
    ll   = bolo['geometry']['ll']
    Psat_data = bolo['Psat']
    # La = bolo['geometry']['La']
    # Lscale = bolo['geometry']['pLscale']

    if linscale:
        # lscale = ascale(ll, La)/ascale(ll[lsind], La)
        lscale = ll[lsind]/ll
        Psat_lin = Psat_data[lsind] * lw/lw[lsind] * lscale    # naive scaling from first 90 GHz bolo

    if pred_wfit:
        fit = sim_dict
        Gpred_170mK = Gfrommodel(fit, an_opts, bolo)   # predictions from each set of fit parameters [pW/K]
        Gpred       = scale_G(Tc, Gpred_170mK, 0.170, n); #sigma_GpredSO = sigma_GscaledT(Tc, Gpred_170mK, 0.170, n, sigma_GpredSO_170mK, 0, 0)   # [pW/K] scale from 170 mK to Tc
        Psat_pred   = Psat_fromG(Gpred, n, Tc, Tbath)
        sigma_Psat  = 0
    else:
        # sim values to use for predictions
        sim_data = sim_dict['sim']['fit_params']
        # Gfrommodel(fit, an_opts, bolo, layer='total'):   # model params, thickness of substrate, leg width, and leg length in um

        # get predictions from model, which predict for 170 mK
        # Gpreds_170mK = Gfrommodel(sim_data, dsub, lw, ll, fab='SO', Lscale=Lscale, stack_I=stack_I, stack_N=stack_N)   # predictions from each set of fit parameters [pW/K]
        Gpreds_170mK = Gfrommodel(sim_data, an_opts, bolo)   # predictions from each set of fit parameters [pW/K]
        Gpreds       = scale_G(Tc, Gpreds_170mK, 0.170, n); #sigma_GpredSO = sigma_GscaledT(Tc, Gpred_170mK, 0.170, n, sigma_GpredSO_170mK, 0, 0)   # [pW/K] scale from 170 mK to Tc
        Psat_preds   = Psat_fromG(Gpreds, n, Tc, Tbath)

        Psat_pred    = np.median(Psat_preds, axis=0); sigma_Psat = np.std(Psat_preds, axis=0)

    if print_results:
        if np.isscalar(dsub):
            # print('For d_sub = {dsub} um, width = {lw} um, length = {ll}, and Lscale = {Lscale}: Psat({Tc} mK) = {Ppred} pW'.format(dsub=dsub, lw=lw, ll=ll, Tc=round(Tc*1E3), Ppred=round(Psat_pred,2), Lscale=round(Lscale,3)))
            print('For d_sub = {dsub} um, width = {lw} um, length = {ll}: Psat({Tc} mK) = {Ppred} pW'.format(dsub=dsub, lw=lw, ll=ll, Tc=round(Tc*1E3), Ppred=round(Psat_pred,2)))
        else:
            for dd, ds in enumerate(dsub):
                print('For d_sub = {dsub} um, width = {lw} um, and length = {ll}: Psat({Tc} mK) = {Ppred} pW'.format(dsub=ds, lw=lw[dd], ll=ll[dd], Tc=round(Tc*1E3), Ppred=round(Psat_pred[dd],2)))

    if plot_predsvdata:

        plt.figure(figsize=[7, 6])
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        ax1 = plt.subplot(gs[0])   # model vs data
        plt.plot([0, max(Psat_data)*1.2], [0, max(Psat_data)*1.2], '--', alpha=0.7, color='k')#, label='Pred. = Data')
        plt.errorbar(Psat_data, Psat_pred, yerr=sigma_Psat, marker='*', capsize=2, linestyle='None', color='k', ms=10, label='Model Pred.')
        # plt.errorbar(Psat_pred, Psat_data, yerr=sigma_Psat, marker='*', capsize=2, linestyle='None', color='k', ms=10, label='Model Pred.')
        if linscale:
            plt.plot(Psat_data, Psat_lin, marker='d', linestyle='None', color='b', ms=10, label='Linear Pred.', alpha=1)
            plt.plot(Psat_data[lsind], Psat_lin[lsind], marker='d', linestyle='None', color='r', ms=10, alpha=1)
            # plt.plot(Psat_lin, Psat_data, marker='d', linestyle='None', color='b', ms=10, label='Linear Pred.', alpha=1)
            # plt.plot(Psat_lin[lsind], Psat_data[lsind], marker='d', linestyle='None', color='r', ms=10, alpha=1)
        # plt.errorbar(Psat_data, Psat_data, marker='o', capsize=2, linestyle='None', color='k', ms=8, label='Data')
        plt.ylabel('Predicted P$_\\text{sat}$ [pW]')
        # plt.ylabel('Measured P$_\\text{sat}$(160mK) [pW]')
        plt.title(title)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
        # plt.ylim(0, 12)
        plt.ylim(2, 10); plt.xlim(2, 10)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)
        # if linscale: plt.legend(loc='upper left')

        # normres = (Psat_pred - Psat_data)/sigma_Psat
        normres = (Psat_pred - Psat_data)/Psat_data
        # normres = (Psat_data-Psat_pred)/Psat_data
        ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
        plt.axhline(0, color='k', alpha=0.7, linestyle='--')
        plt.errorbar(Psat_data, normres*100, yerr=sigma_Psat/Psat_data*100, marker='*', capsize=2, linestyle='None', color='k', ms=10)
        # plt.errorbar(Psat_pred, normres*100, yerr=sigma_Psat/Psat_data*100, marker='*', capsize=2, linestyle='None', color='k', ms=10)
        if linscale:
            plt.plot(Psat_data, (Psat_lin-Psat_data)/Psat_data*100, marker='d', linestyle='None', color='b', ms=10, alpha=1)
            plt.plot(Psat_data[lsind], (Psat_lin[lsind]-Psat_data[lsind])/Psat_data[lsind]*100, marker='d', linestyle='None', color='r', ms=10, alpha=1)
            # plt.plot(Psat_lin, (Psat_data-Psat_lin)/Psat_data*100, marker='d', linestyle='None', color='b', ms=10, alpha=1)
            # plt.plot(Psat_lin[lsind], (Psat_data[lsind]-Psat_lin[lsind])/Psat_data[lsind]*100, marker='d', linestyle='None', color='r', ms=10, alpha=1)
        # plt.ylabel('Norm. Res.')
        plt.ylabel('\% Diff')
        # plt.ylim(-30, 30)
        plt.ylim(-50, 50)
        plt.xlabel('Measured P$_\\text{sat}$ [pW]')
        # plt.xlabel('Predicted P$_\\text{sat}$ [pW]')
        # plt.xlim(2, 9)
        # plt.xlim(2, 10)
        plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
        plt.grid(linestyle = '--', which='both', linewidth=0.5)

        if save_figs: plt.savefig(plot_dir + 'SOPsat_predsvdata_'+fn_comments+'.pdf')

        # # vs width
        # plt.figure(figsize=[7, 6])
        # gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        # ax1 = plt.subplot(gs[0])   # model vs data
        # plt.errorbar(lw, Psat_data, marker='o', linestyle='None', color='C3', ms=9, label='Data', zorder=-1)
        # plt.errorbar(lw, Psat_pred, yerr=sigma_Psat, marker='*', capsize=2, linestyle='None', color='k', ms=10, label='Model Pred.')
        # if linscale: plt.plot(lw, Psat_lin, marker='d', linestyle='None', color='orange', ms=9, label='Lin. Scale', alpha=0.8)
        # plt.ylabel('Predicted P$_\\text{sat}$(160mK) [pW]')
        # plt.title(title)
        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)   # turn x ticks off
        # plt.ylim(0, 10)
        # plt.grid(linestyle = '--', which='both', linewidth=0.5)

        # lorder = np.array([1, 2, 0])
        # handles, labels = ax1.get_legend_handles_labels()
        # plt.legend([handles[idx] for idx in lorder],[labels[idx] for idx in lorder], loc=2)   # 2 is upper left, 4 is lower right
        # # if linscale: plt.legend(loc='upper left')

        # # normres = (Psat_pred - Psat_data)/sigma_Psat
        # normres = (Psat_pred - Psat_data)/Psat_data
        # ax2 = plt.subplot(gs[1], sharex=ax1)   # residuals
        # plt.axhline(0, color='k', alpha=0.7, linestyle='--')
        # plt.errorbar(lw, normres*100, yerr=sigma_Psat/Psat_data*100, marker='*', capsize=2, linestyle='None', color='k', ms=10)
        # if linscale: plt.plot(lw, (Psat_lin-Psat_data)/Psat_data*100, marker='d', linestyle='None', color='orange', ms=9, label='Lin. Scale', alpha=0.8)
        # plt.ylabel('\% Diff')
        # # plt.ylim(-30, 30)
        # plt.ylim(-50, 50)
        # plt.xlabel('Width $[\mu m]$')
        # # plt.xlim(2, 9)
        # plt.subplots_adjust(hspace=0.075)   # merge to share one x axis
        # plt.grid(linestyle = '--', which='both', linewidth=0.5)

    return Psat_pred, sigma_Psat

def target_width(target_Psat, fit, dsub, ll, lw0=15., Lscale=1.0, model='Three-Layer', stack_I=False, stack_N=False):

    def width_finder(lw, target_Psat, fit, dsub, ll, Lscale=1.0, model='Three-Layer', stack_I=False, stack_N=False):
        Psat, sigma_Psat = predict_PsatSO(fit, dsub, lw, ll, Lscale=Lscale, pred_wfit=True, model=model, plot_predsvdata=False, print_results=False, stack_I=stack_I, stack_N=stack_N)
        return Psat - target_Psat

    twidth = fsolve(width_finder, lw0, args=(target_Psat, fit, dsub, ll, Lscale, model, stack_I, stack_N))[0]   # solve for root
    # print('Leg width for target Psat of {tPsat} pW/K is {tlw} um'.format(tPsat=target_Psat, tlw=round(twidth, 2)))
    return twidth

def predictPsat_vlwidth(sim_dict, lws, dsub, ll, Lscale=1.0, tPsat=None, lw0=15, model='Three-Layer', title='', plot_dir='./', save_figs=False, fn_comments='', stack_I=False, stack_N=False):

    Psats, sigma_Psats = predict_PsatSO(sim_dict, dsub, lws, ll, Lscale=Lscale, model=model, title=title, stack_I=stack_I, stack_N=stack_N)
    plt.figure()
    plt.plot(lws, Psats, color='k')

    plt.fill_between(lws, Psats-sigma_Psats, Psats+sigma_Psats, facecolor='gray', alpha=0.5)
    plt.hlines(tPsat, min(lws), max(lws), linestyle='--', color='k')
    plt.xlabel('Leg Width [um]'); plt.ylabel('Psat [pW]')
    if tPsat:
        fit = sim_dict['fit']['fit_params']   # fit parameters - final result
        twidth = target_width(tPsat, fit, dsub[0], ll[0], lw0=lw0, Lscale=Lscale, model=model, stack_I=stack_I, stack_N=stack_N)
        print('\nLeg width for target Psat of {tPsat} pW is {tlw} um, assuming L = {L} um; dsub = {dsub} um; Lscale = {Lscale}\n'.format(tPsat=tPsat, tlw=round(twidth, 2), L=ll[0], dsub=dsub[0], Lscale=round(Lscale,3)))
        plt.scatter(twidth, tPsat, marker='x', color='k', s=60)
        plt.annotate('{tw} um'.format(tw=round(twidth,2)), (twidth, tPsat*(1-0.08)), fontsize='14')
        plt.title(title)
    if save_figs: plt.savefig(plot_dir + 'Psat_vwidth_prediction_target'+str(tPsat)+'pW.png', dpi=300)

    return Psats, sigma_Psats

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
    # calculates specific heat for bulk SiN or supercondcuting Nb
    # electron gamma is for Nb [J/mol/K^2] - LEUPOLD & BOORSE 1964
    a = 8.21; b = 1.52

    # if carrier=='electron': C_v = (gamma*Tc*a*np.exp(-b*Tc/T))/volmol  # electron specific heat, electron from LEUPOLD & BOORSE 1964, pJ/K/um^3 (=J/K/cm^3)
    # elif carrier=='phonon': C_v = ((12*np.pi**4*kB)/5 * (T/TD)**3) * NA/volmol  # phonon specific heat from low temp limit of Debye model, pJ/K/um^3 (=J/K/cm^3)
    if carrier=='electron': C_v = (gamma*Tc*a*np.exp(-b*Tc/T))/volmol  # electron specific heat, electron from LEUPOLD & BOORSE 1964, J/K/m^3 (= 1E-6 pJ/K/um^3)
    elif carrier=='phonon': C_v = ((12*np.pi**4*kB)/5 * (T/TD)**3) * NA/volmol  # phonon specific heat from low temp limit of Debye model, J/K/m^3 (= 1E-6 pJ/K/um^3)
    # elif carrier=='phonon': C_v = ((12*np.pi**4*kB)/5 * (T/TDebye(vs_Nb, volmol_Nb))**3) * NA/volmol  # phonon specific heat from low temp limit of Debye model, J/K/m^3 (= 1E-6 pJ/K/um^3)
    else: print("Invalid carrier, an_opts are 'phonon' or 'electron'")
    return C_v

def TDebye(vs, volmol):
    return planck * vs / (2*np.pi * kB) * (6*np.pi**2 * NA/volmol)**(1/3)   # K
    # return planck * vs / kB * (3/(4*np.pi) * NA/volmol)**(1/3)   # K

def kappa_permfp(T, material=''):   # Leopold and Boorse 1964, Nb
    # calculates theoretical thermal conductivity via specific heat for bulk SC Nb and SiN
    # INPUT: bath temp, Debye temperature, critical temperature, carrier velocity (Fermi vel for electrons, sound speed for phonons) in um/s
    # RETURNS: thermal conductivity per mean free path in pW / K / um^2

    if material=='Nb':
        # Cv_ph = Cv(T, TD_Nb, Tc_Nb, volmol_Nb, carrier='phonon')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of phonons from low temp limit of Debye model
        # Cv_el = Cv(T, TD_Nb, Tc_Nb, volmol_Nb, carrier='electron')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of electrons from LEUPOLD & BOORSE 1964
        Cv_ph = Cv(T, TDebye(vs_Nb, volmol_Nb), Tc_Nb, volmol_Nb, carrier='phonon')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of phonons from low temp limit of Debye model
        Cv_el = Cv(T, TDebye(vs_Nb, volmol_Nb), Tc_Nb, volmol_Nb, carrier='electron')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of electrons from LEUPOLD & BOORSE 1964
        kappapmfp = 1/3*(Cv_ph*vs_Nb + Cv_el*vF_Nb)*1E6 # [pW/K/um^2] thermal conductivity via phenomenological gas kinetic theory / mfp

    elif material=='SiN':
        # Cv_ph = Cv(T, TD_SiN, np.nan, volmol_SiN, carrier='phonon')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of phonons from Debye model
        Cv_ph = Cv(T, TDebye(vs_SiN, volmol_SiN), np.nan, volmol_SiN, carrier='phonon')*1E-6  # [pJ/K/um^3 = 1E6 J/K/m^3] vol heat capacity of phonons from Debye model
        kappapmfp = 1/3*(Cv_ph*vs_SiN)*1E6   # [pW/K/um^2] thermal conductivity via phenomenological gas kinetic theory / mfp

    else:
        print('Invalid material. Options are Nb and SiN')
        return

    return kappapmfp

# extract mean free path from measured G
def solve_La(G, Gb, L):
    return L*G / (Gb-G)

def mfpa_SiNic(Gb, G, L):
    # mfp from Gd = Gb * pi * mfp / (4L)
    return 4 / (np.pi*Gb)   * (1/G - 1/Gb)**(-1) * L

def mfp_kappa(G, Gb, C_v, vs, w, d, L):
    # mfp using parallel sum and Gd = A/L * kappa
    return 3 / (w * d * C_v * vs) * (1/G - 1/Gb)**(-1) * L

def mfpk_fromLa(Gb, w, d, C_v, vs, La):   # diffusive mfp from acoustic length
    return 3 * Gb / (w * d * C_v * vs) * La

def mfpa_fromLa(La):   # acoustic mfp from acoustic length
    return 4 / np.pi * La

def Cv_wang(T):
    # amorphous SiNx specific heat = a*T + b*T^3
    # a and b were measured in W+11
    return 0.082*T + 0.502*T**3   # J/K/m^3

def kpermfp_wang(T, vs):
    # kappa per mfp from Casimir for Si3N4 - volumetric heat capacity * average sound speed
    Cv_w = Cv_wang(T)*1E-6   # pW/K/um^3 = 1E-6 J/K/m^3
    return 1/3 * Cv_w * vs*1E6   #  pW/K/um^2, vs has units [m/s]

def sigma_rad(v_l, v_t):   # phonon stefan boltzmann constant; Hoevers ea 2005 and Holmes ea 1998
    return np.pi**5 * kB**4 / (15 * planck**3) * (1/v_l**2 + 2/v_t**2)   # equivalent to ... * (3/vs**2)

def I_mfp(x):
    # return x/2 * np.arcsinh(x) + 1/6*((1+x**2)**(1/2) * (x**2-2) + (2-x**3))
    return x/2 * np.arcsinh(x) + 1/6 * (np.sqrt(1+x**2) * (x**2-2) + (2-x**3))

def xi_Cas(w, d, L):
    # numerical factor in the Casimir (diffusive) limit
    n = w/d   # aspect ratio
    return 3/(2*L) * (3*d**2)/(2*w) * ((n)**3 * I_mfp(1/n) + I_mfp(n))

def G_radtheory(d, w, L, vst, vsl, T=0.170, dim='3D', lim='diff'):
    # Theoretical G in diffusive or ballistic limit, selectable dimensionality
    # Holmes Thesis Section 4.5.1

    if lim=='diff':   # diffusive (Casimir) limit
        xi = xi_Cas(w, d, L)
    elif lim=='ball':   # ballistic limit
        xi = 1

    if dim=='3D':
        vsq_sum = 2/(vst**2) + 1/(vsl**2)   # sum with transverse and longitudinal sound speeds
        sigma = np.pi**5 * kB**4 /(15*planck**3) * vsq_sum   # W/K^4/m^2
        G0 = 4*sigma*d*w*T**3   # W / K if d and w are in meters = 0.1 mW/m^2/K^4
    elif dim=='2D':
        P = L + w   # perimeter of 2D surface
        RZ = zeta(3)   # Reimann Zeta Function (returns scalar)
        v_sum = 2/(vst) + 1/(vsl)
        G0 = 3*P*RZ*kB**3 / (6*planck**2) * v_sum * T**2   # W / K
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

    # def calc_AoL(dsub, lwidth, llength, dW1=0.200, dW2=0.400, dI1=0.350, dI2=0.400, w1w=8.0, w2w=5.0):
    #     """
    #     Calculate leg area / leg length [um] for bolometers with 4 legs, including thickness of microstrip layers (W1-I1-W2-I2)

    #     INPUT:
    #     all dimensions in [um]
    #     dsub = thickness of substrate (often includes nitride + oxide)
    #     lwidth = width of a single leg
    #     llength = length of a single leg
    #     dI1, dI2, dW1, dW2 = thickness of wiring I and W layers
    #     w1w, w2w = width of W1 and W2 layers, often < lwidth

    #     RETURNS:
    #     AoL = [um] cross-sectional area / length for 4 legs
    #     """
    #     A = 4*( (dsub + dI1 + dI2)*lwidth + dW1*w1w + dW2*w2w )   # [um^2] cross-sectional area of four legs, thickness is substrate + wiring stack
    #     return A/llength   # [um] cross=sectional area of four legs / length of the legs

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))