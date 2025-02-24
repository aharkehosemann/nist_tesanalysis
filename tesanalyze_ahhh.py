import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit
# import pylab
import matplotlib.pyplot as plt
import pdb
from collections import OrderedDict
import pickle as pkl
import csv

"""
Adapted from Doug Bennett's TES IV analysis module, now with error analysis!
AHHH

If curve_fit is just given sigma, it just uses the relative values of those errors, but pcov is rescaled by the sample variance of the data such that you get reduced chi-squared equal to one.
Set absolute_sigma=True to get true error.
"""

kB = 1.3806503e-23   # Boltzmann constant
mcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

# map pad number to bolo id
pad_bolo_map = {'1':'1f', '2':'1e', '3':'1d', '4':'1c', '5':'1b', '6':'1a', '7':'1f', '8':'1e', '9':'1d', '10':'1c', '11':'1b', '12':'1a',
                '13':'24', '14':'23', '15':'22', '16':'13', '17':'21', '18':'20', '19':'24', '20':'23', '21':'22', '22':'13', '23':'21',
                '24':'20','25':'7','26':'26','27':'27','28':'7','29':'26','30':'27'}

# readout circuit parameters
M_r = 8.5   # (SQUID-input coupling: current = arbs / (M_r * R_fb))
Rfb = 2070.   # Ohms (SQUID feeback resistor), Kelsey: accurate to ~5%
Rsh = 370.e-6   # Ohms (bias circuit shunt), Kelsey: accurate to 10-15%
Rb = 1020.   # Ohms (TES bias resistor), Kelsey: accurate to ~5%

class TESAnalyze():
    '''
    Analysis functions
    '''

    def ivAnalyzeTDM(self, Vbias, Vfb, rfb, rbias, rsh, mr, v_nfit, rpar=None, Vbias_offset=0.0,
                     i_offset=0.0, rm_fl=False, rm_ones=False, dlim=1.4, slim=0.4, ftol=0.05, iv_plots=False, branch_plots=False):   # dlim=1.3 doesn't work
        '''Fit supeconducting and normal branches and use information to convert raw data to IV.'''
        # dlim and flim are diff(Vfb)/diff(sc Vfb) thresholds for finding discontinuities and flagged data points

        # remove flagged raw data points
        flinds = np.array([]); end_sc = None
        if rm_fl:   # remove data points between discontinuities and with low derivatives
            Vfb_diff = np.abs(np.diff(Vfb))   # first derivative
            firstdiff = np.mean(Vfb_diff[np.isfinite(Vfb_diff)][0:3])   # SC branch first derivative
            normdiff = np.abs(Vfb_diff/firstdiff)   # value to compare derivative values to
            disinds = np.where(normdiff>=dlim)[0]+1   # discontinuities (high derivative values)
            if len(disinds)>0: end_sc=max(disinds)

            ldinds = np.where((normdiff<=slim) & (np.arange(len(normdiff)) < max(np.append(0, disinds))))[0]   # data points before normal branch with suspiciously low first derivatives
            if len(ldinds)>0: ldinds = np.append([max(ldinds)+1, max(disinds)-1], ldinds)   # include point just before last disc and last low deriv point which will have a large deriv (could be the same)
            sinds = np.arange(min(ldinds), max(ldinds)+1) if len(ldinds)>0 else ldinds   # remove all points in range
            flinds = np.unique(np.append(flinds, sinds))

            if len(disinds)>1:   # if more than one discontinuity in raw data, remove the points between first and last discontinuity
                btwnds = np.arange(min(disinds), max(disinds))
                flinds = np.append(flinds, btwnds)
        if rm_ones:   # remove raw data around Vfb = 1 (naive data flagging)
            oneinds = np.where((Vfb<1.+ftol) & (Vfb>1.-ftol))[0]  # flag data points at Vfb = 1 +/- flag tolerance
            flinds = np.append(flinds, oneinds)
        flinds = np.unique(flinds).astype(np.int64)

        if iv_plots:   # show raw IV
            rawIVfig = plt.figure()
            plt.plot(Vbias, Vfb, '.', label='data', alpha=0.7)
            plt.plot(Vbias[:-1], normdiff, '.', label='first deriv', alpha=0.7)
            plt.plot(Vbias[disinds], Vfb[disinds], 'x', color='red', label='discontinuities')
            plt.plot(Vbias[flinds], Vfb[flinds], 'x', color='k', label='removed datapoints', alpha=0.7)
            plt.xlabel('Vbias [V]'); plt.ylabel('Vfb [V]')
            plt.title('Raw IV')
            plt.legend()

        Vfb[flinds] = np.nan   # remove flagged data points

        vb_step = Vbias[1]-Vbias[0]
        if vb_step < 0:   # reverse order if data was taken from high V -> 0
            Vbias = Vbias[::-1]
            Vfb = Vfb[::-1]
        if branch_plots: branchfig = plt.figure(); plt.xlabel('Vbias [V]'); plt.ylabel('Vfb [V]')

        n_output = self.ivFitNormal(Vbias, Vfb, Vbias_offset, v_nfit, show_plot=branch_plots)
        n_fit = n_output[0]; norm_inds = n_output[1]

        if n_fit[0] < 0:   # Flip y axis if slope of normal branch is negative
            Vfb = -Vfb + np.nanmax(Vfb) if np.nanmax(Vfb)>=0 else -Vfb   # flip and move to positive quadrant
            if branch_plots: branchfig.clear(True); plt.xlabel('Vbias [V]'); plt.ylabel('Vfb [V]')
            n_output = self.ivFitNormal(Vbias, Vfb, Vbias_offset, v_nfit, show_plot=branch_plots)
            n_fit = n_output[0]; norm_inds = n_output[1]
        norm_offset = n_fit[1]

        sc_fit, end_sc = self.ivFitSuperconduct(Vbias, Vfb, Vbias_offset, show_plot=branch_plots, end_sc=end_sc)
        m_sc = sc_fit[0]; sc_offset = sc_fit[1]

        # remove offsets, norm and SC branches can have different offsets
        Vfb[end_sc:] = Vfb[end_sc:] - norm_offset   # adjust for SC branch offset, include last superconducting point
        Vfb[:end_sc] = Vfb[:end_sc] - sc_offset   # adjust for SC branch offset, include last superconducting point

        if rpar is None:   # calculate Rparasitic
            rpar = rsh*(rfb*mr/(rbias*m_sc)-1.0) if np.isfinite(m_sc) else 0.

        Vtes, Ites, Rtes, Ptes, i_meas = self.ivConvertTDM(Vbias, Vfb, rfb, rbias, rsh, mr, rpar, 0, Vbias_offset=Vbias_offset, i_offset=i_offset)

        if iv_plots:
            plt.figure()
            plt.plot(Vtes, Ites, '.', alpha=0.7)
            plt.title('Real Units')
            plt.xlabel('V_TES [V]'); plt.ylabel('I_TES [A]')

        return Vtes, Ites, Rtes, Ptes, i_meas, n_fit, norm_inds, sc_fit, end_sc, rpar

    def ivConvertTDM(self, Vbias, Vfb, rfb, rbias, rsh, mr, rpar, norm_offset, Vbias_offset=0.0, i_offset=0.0):
        '''Convert IV data.'''
        vb_step = Vbias[1]-Vbias[0]
        if vb_step < 0:
            Vbias = Vbias[::-1]
            Vfb = Vfb[::-1]
        i_meas = (Vfb-norm_offset)/(rfb*mr) + i_offset   # current measured by the SQ?
        Ites = i_meas - i_offset   # current through the TES measured by the SQ?
        i_bias = (Vbias-Vbias_offset)/rbias   # TES bias current
        Vtes = (i_bias-i_meas)*rsh-i_meas*rpar
        Rtes = Vtes/Ites
        Ptes = Vtes*Ites

        return  Vtes, Ites, Rtes, Ptes, i_meas


    def ivFitNormal(self, Vbias, Vfb, Vbias_offset, v_nfit, show_plot=False):
        '''Fit the normal branch of the IV.'''

        x = Vbias-Vbias_offset
        y = Vfb
        norm_inds = np.where(x>v_nfit)
        m_a, b_a = np.polyfit(x[norm_inds], y[norm_inds], 1)
        n_fit = [m_a, b_a]
        xf = x[norm_inds]
        yfa = xf*m_a+b_a
        if show_plot is True:
            plt.plot(x,y,'.', label='Data')
            plt.plot(xf,yfa, label='Normal Branch Fit')

        return n_fit, norm_inds

    def ivFitSuperconduct(self, Vbias, Vfb, Vbias_offset, show_plot=False, end_sc=None):
        '''Fit the superconducting branch of the IV and return the slope and intercept. '''

        if end_sc==None:   # find end of SC branch if not already given
            fb_diff = np.diff(Vfb)
            neg_slope = np.where(fb_diff<0)[0]
            end_sc =  min(neg_slope)  # index of first point in transition
        if end_sc<2:
            print('TES is normal, cannot fit SC branch')
            return [np.nan, np.nan], end_sc
        # Vbias_sc = Vbias[:end_sc+1] - Vbias_offset
        # Vfb_sc = Vfb[:end_sc+1]
        Vbias_sc = Vbias[:end_sc] - Vbias_offset
        Vfb_sc = Vfb[:end_sc]

        finds = np.isfinite(Vfb_sc)   # ignore nans
        sc_m, sc_b = scipy.polyfit(Vbias_sc[finds], Vfb_sc[finds], 1)
        Vfb_sc_fit = scipy.polyval([sc_m,sc_b],Vbias_sc)
        sc_fit = [sc_m, sc_b]
        if show_plot is True:
            plt.plot(Vbias_sc,Vfb_sc_fit, label='SC Branch Fit')
            plt.legend()

        return sc_fit, end_sc


    def ivInterpolate(self, Vtes, Ites, Rtes, percentRns, rn, tran_perRn_start=0.050, plot=False):
        # interpolate IV data points at requested % Rn's
        btran_indices = np.where((Rtes/rn)<tran_perRn_start)
        if len(btran_indices[0]) == 0:   # check for normal IVs, this throws off the interpolator
            print('TES is normal, interpolator failed')
            return None, None, None
        tran_start = np.max(btran_indices)
        Vtes_tran = Vtes[tran_start:]
        Ites_tran = Ites[tran_start:]
        Rtes_tran = Rtes[tran_start:]
        # Check that the r data being used for interpolation is monotonically increasing
        if np.all(np.diff(Rtes_tran) > 0) is False:
            print('IV %s not monotomicaly increasing in R')
        # Find the I and V that corespond to the requested percent Rns
        v_pnts = np.interp(percentRns/100.0*rn, Rtes_tran, Vtes_tran)
        i_pnts = np.interp(percentRns/100.0*rn, Rtes_tran, Ites_tran)
        p_pnts = v_pnts*i_pnts
        if plot:
            # plt.figure()
            plt.plot(v_pnts*1e9, i_pnts*1e6, 'o', alpha=0.8)
            plt.xlabel('V [nV]'); plt.ylabel('I [uA]')

        return v_pnts, i_pnts, p_pnts

    def ivPRnLinearFit(self, v_pnt_array, i_pnt_array):

        plt.figure()
        for index in range(len(v_pnt_array[:,0])):
            plt.plot(v_pnt_array[index]*1e6, i_pnt_array[index]*1e3,'o')
            rvt_m, rvt_b = scipy.polyfit(v_pnt_array[index], i_pnt_array[index], 1)
            Ites_rvt_fit = scipy.polyval([rvt_m,rvt_b],v_pnt_array[index])
            rvt_residuals = i_pnt_array[index]-Ites_rvt_fit
            plt.plot(v_pnt_array[index]*1e6, Ites_rvt_fit*1e3)
        plt.title('Fits to percent Rn values')
        plt.xlabel('VTES (mu)')
        plt.ylabel('ITES (mA)')

    # def fitPowerLaw(self, perRn, Tbaths, powerAtRns, init_guess, fitToLast=True, TbsToReturn=[0.080,0.094,0.095], plot=True, suptitle='', sigma=None, pfigpath=None, constT=False, fitGexplicit=False, fixedK=False, fixedn=False):
    def fitPowerLaw(self, perRn, Tbaths, powerAtRns, init_guess, fitToLast=True, Tb_inds=[], plot=True, suptitle='', sigma=None, pfigpath=None, constT=False, fitGexplicit=False, fixedK=False, fixedn=False):
        '''Fit power versus T_base with power law.'''

        if plot is True:
            f1 = plt.figure(figsize=(6,6))   # power law fit figure

        # fit for G, k, or n explicitly
        if fitGexplicit:
            fitfnct = self.tespowerlaw_fitfunc_G
        elif fixedK:   # fix K, fit for n and Tc
            fitfnct = lambda x, n, Tc: self.tespowerlaw_fitfunc_k(x, fixedK, n, Tc)
        elif fixedn:   # fix K, fit for n and Tc
            fitfnct = lambda x, k, Tc: self.tespowerlaw_fitfunc_k(x, k, fixedn, Tc)
        else:
            fitfnct = self.tespowerlaw_fitfunc_k

        Ks = np.zeros(len(perRn)); ns = np.zeros(len(perRn)); Tcs = np.zeros(len(perRn)); GTcs = np.zeros(len(perRn))   # initialize arrays
        Ks_error = np.zeros(len(perRn)); ns_error = np.zeros(len(perRn)); Tcs_error = np.zeros(len(perRn)); GTcs_err = np.zeros(len(perRn))

        for index in range(len(perRn)):   # for each percent Rn, assumes constant T_TES that is very close to Tc

            # fit power law
            if type(sigma) is np.ndarray: # measurement error included in fit
                p0, p0cov = curve_fit(fitfnct, Tbaths, powerAtRns[index], p0=init_guess)   # get global p0 values by neglecting measurement errors first
                pfit, pcov = curve_fit(fitfnct, Tbaths, powerAtRns[index], p0=p0, sigma=sigma[index], absolute_sigma=True)   # non-linear least squares fit, set absolute_sigma=True to get true variance in pcov
            else: # don't include measurement error
                pfit, pcov = curve_fit(fitfnct, Tbaths, powerAtRns[index], p0=init_guess)   # non-linear least squares fit, set absolute_sigma=True to get true variance in pcov
            perr = np.sqrt(np.diag(pcov))   # error of fit

            # sort fit parameters
            if fitGexplicit:
                GTcs[index] = pfit[0]; ns[index] = pfit[1]; Tcs[index] = pfit[2]   # fit parameters
                GTcs_err[index] = perr[0]; ns_error[index] = perr[1]; Tcs_error[index] = perr[2]   # error of fit parameters
                Ks, Ks_err = self.k(GTcs, GTcs_err, ns, ns_error, Tcs, Tcs_error)   # k for each % Rn
            elif fixedK:   # only fit n and Tc
                Ks[index] = fixedK; Ks_error[index] = 0
                ns[index] = pfit[0]; Tcs[index] = pfit[1]   # fit parameters
                ns_error[index] = perr[0]; Tcs_error[index] = perr[1]   # error of fit parameters
                GTcs, GTcs_err = self.GatTc(ns, ns_error, Ks, Ks_error, Tcs, Tcs_error)   # G(Tc) for each %Rn
            elif fixedn:   # only fit n and Tc
                ns[index] = fixedn; ns_error[index] = 0
                Ks[index] = pfit[0]; Tcs[index] = pfit[1]   # fit parameters
                Ks_error[index] = perr[0]; Tcs_error[index] = perr[1]   # error of fit parameters
                GTcs, GTcs_err = self.GatTc(ns, ns_error, Ks, Ks_error, Tcs, Tcs_error)   # G(Tc) for each %Rn
            else:
                Ks[index] = pfit[0]; ns[index] = pfit[1]; Tcs[index] = pfit[2]   # fit parameters
                Ks_error[index] = perr[0]; ns_error[index] = perr[1]; Tcs_error[index] = perr[2]   # error of fit parameters
                GTcs, GTcs_err = self.GatTc(ns, ns_error, Ks, Ks_error, Tcs, Tcs_error)   # G(Tc) for each %Rn

            temp_pnts = np.linspace(min(Tbaths)-0.01,Tcs[index]+0.01,25)
            if plot:
                if type(sigma) is np.ndarray:
                    plt.errorbar(Tbaths*1e3, powerAtRns[index]*1e12, yerr=sigma[index]*1e12, fmt='o', color=plt.cm.viridis(perRn[index]/100-0.05), alpha=0.8)   # matching data and model colors
                else:
                    plt.errorbar(Tbaths*1e3, powerAtRns[index]*1e12, fmt='o', color=plt.cm.viridis(perRn[index]/100-0.05), alpha=0.8)   # matching data and model colors
                plt.plot(temp_pnts*1e3, fitfnct(temp_pnts, *pfit)*1e12, label='{}\% Rn'.format(perRn[index]), color=plt.cm.viridis(perRn[index]/100-0.05), alpha=0.8)
                # prepare confidence level curves
                # pfit_up = pfit + nstd * perr; pfit_dw = pfit - nstd * perr
                # fit_up = self.tespowerlaw_fitfunc_k(temp_pnts, *pfit_up); fit_down = self.tespowerlaw_fitfunc_k(temp_pnts, *pfit_dw)
                # plt.fill_between(temp_pnts*1e3, fit_down*1e12, fit_up*1e12, alpha=.3, label='{}-sigma interval'.format(nstd), color=mcolors[index+6])
        if plot:
            plt.legend()
            plt.xlabel('Bath Temperature [mK]')
            plt.ylabel('Power [pW]')
            plt.title(suptitle)
            if pfigpath: plt.savefig(pfigpath, dpi=300)

        Tbaths = Tbaths[Tb_inds]

        # Find the T_TES and G for a given Tbath using the highest % Rn parameters
        if fitToLast:

            K = Ks[-1]; Kerr = Ks_error[-1]
            n = ns[-1]; nerr = ns_error[-1]
            Ttes = np.zeros((len(Tbaths),len(perRn))); Ttes_error = np.zeros((len(Tbaths),len(perRn)))   # T_TES is a function of %Rn and Tbath
            GTbs = np.zeros((len(Tbaths),len(perRn))); GTbs_err = np.zeros((len(Tbaths),len(perRn)))   # G(T_TES) for each Tb and %Rn

            for bb in range(len(Tbaths)):
                # pdb.set_trace()

                Tb = Tbaths[bb]
                Tbind = Tb_inds[bb]
                Ttes[bb], Ttes_error[bb] = self.T_TES(powerAtRns[:, Tbind], sigma[:, Tbind], n, nerr, K, Kerr, Tb)   # T_TES for all % Rn at one Tbath
                GTbs[bb], GTbs_err[bb] = self.GatTc(n, nerr, K, Kerr, Ttes[bb], Ttes_error[bb])    # G varies with Tbath because T_TES varies with Tbath

        if plot:
            tsort = np.argsort(Tbaths)   # plot lines in order of bath temp
            f2 = plt.figure(figsize=(7,6))   # fit parameters
            p1 = f2.add_subplot(2,2,1)
            p1.errorbar(perRn, Ks*1e12, yerr=Ks_error*1E12, fmt='o')
            plt.title('k')
            plt.ylabel('k [$pW/K^n$]')
            p2 = f2.add_subplot(2,2,2)
            p2.errorbar(perRn, ns, yerr=ns_error, fmt='o')
            plt.title('n')
            plt.ylabel('n')
            p3 = f2.add_subplot(2,2,3)
            for tt in tsort:
                p3.errorbar(perRn, Ttes[tt]*1E3, yerr=Ttes_error[tt]*1E3, fmt='o', label='{:.1f} mK'.format(Tbaths[tt]*1e3), alpha=0.7, color=plt.cm.plasma((Tbaths[tt]-min(Tbaths)*1)/(max(Tbaths)*0.8)))
            p3.errorbar(perRn, Tcs*1E3, yerr=Tcs_error*1E3, fmt='o', label='constant T', alpha=0.7, color='k')
            plt.title('$T_{TES}$')
            plt.xlabel('% $R_n$')
            plt.ylabel('$T_{TES}$ [$mK$]')
            leg3 = plt.legend(loc='upper left')
            try:
                for t in leg3.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')

            p4 = f2.add_subplot(2,2,4)
            for tt in tsort:
                p4.errorbar(perRn, GTbs[tt]*1e12, yerr=GTbs_err[tt]*1E12, fmt='o',label='{:.1f} mK'.format(Tbaths[tt]*1e3), alpha=0.7, color=plt.cm.plasma((Tbaths[tt]-min(Tbaths)*1)/(max(Tbaths)*0.8)))
            p4.errorbar(perRn, GTcs*1e12, yerr=GTcs_err*1E12, fmt='o', label='constant T', alpha=0.7, color='k')
            plt.title('G')
            plt.xlabel('% $R_n$')
            plt.ylabel('G [pW/K]')
            leg4 = plt.legend(loc='upper left')
            try:
                for t in leg4.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')
            plt.suptitle(suptitle, x=.55, y=.95)
            f2.tight_layout()

        if constT:   # return constant T fit values of G(Tc) (can be less accurate at lower %Rn)
            return GTcs, Ks, ns, Tcs, GTcs_err, Ks_error, ns_error, Tcs_error
        else:
            return GTbs, Ks, ns, Ttes, GTbs_err, Ks_error, ns_error, Ttes_error


#************************

    def tespowerlaw_fitfunc_k(self, x, k, n, Tc):
        return k*(Tc**n-np.power(x,n))   # P = kappa * (Tc^n - x^n)

    def tespowerlaw_fitfunc_G(self, x, G, n, Tc):
        return G/n * Tc *(1-np.power(x/Tc,n))   # P = G/n * Tc * (1 - (Tb/Tc)^n)

    # def powerlaw_err_func(self, k, n, Tc, x, y):
    #     return y - self.powerlaw_fitfunc(p,x)

    def calculateAlphaFromIV(self, percentRn, Ts_array, rn, plot=True):
        '''Calculates alpha from Tc values for each percent Rn'''

        if plot is True:
            plt.figure()

        alpha_array = np.zeros((len(Ts_array),len(Ts_array[0])-1))
        Tmid_array = np.zeros_like(alpha_array)
        for index in range(len(Ts_array)):
            Ts = Ts_array[index]
            R = percentRn*rn
            dR = np.diff(R)
            dT = np.diff(Ts)
            dRdT = np.divide(dR,dT)
            Rmid = R[:-1]+dR/2.0
            Tmid = Ts[:-1]+dT/2.0
            alpha = Tmid/Rmid*dRdT
            perRnMid = Rmid/rn
            alpha_array[index] = alpha
            Tmid_array[index] = Tmid

            if plot is True:
                plt.plot(perRnMid,alpha, 'o')

        if plot is True:
            plt.title('Alpha from extracted TC values')
            plt.xlabel('% $R_n$')
            plt.ylabel('Alpha')

        return alpha_array, perRnMid, Tmid_array

    def sigma_power(self, i, sigma_I, v, sigma_V):   # calculate error in power measurement from error in voltage and current measurement
        return np.sqrt(np.multiply(i**2,sigma_V**2) + np.multiply(v**2,sigma_I**2))   # sigma_power^2 = (I*sigma_V)^2 + (V*sigma_I)^2

    def Psat_atT(self, T, Tc, k, n):
        return k*(Tc**n-T**n)

    def Psat_atTandGTc(self, T, Tc, GTc, n):
        return GTc*Tc/n*(1-(T/Tc)**n)

    def GatTc(self, n, sigma_n, k, sigma_k, Tc, sigma_Tc):
        G = n*k*Tc**(n-1)
        sigma_G = np.sqrt( G**2 * ((1+n*np.log(Tc))**2*(sigma_n/n)**2 + (sigma_k/k)**2 + (n-1)**2*(sigma_Tc/Tc)**2) )
        # sigma_G = np.sqrt( G**2 * ((sigma_n/n)**2 + (sigma_k/k)**2 + (n-1)**2*(sigma_Tc/Tc)**2) )    # INCORRECT
        return G, sigma_G

    def scale_G(self, T, GTc, Tc, n):
        return GTc * T**(n-1)/Tc**(n-1)

    def sigma_GscaledT(self, T, GTc, Tc, n, sigma_GTc, sigma_Tc, sigma_n):
        Gterm = sigma_GTc * T**(n-1)/(Tc**(n-1))
        Tcterm = sigma_Tc * GTc * (1-n) * T**(n-1)/(Tc**(n))   # this is very tiny
        nterm = sigma_n * GTc * T/Tc**(n-1) * np.log(T/Tc)
        return np.sqrt( Gterm**2 + Tcterm**2 + nterm**2 )   # quadratic sum of sigma G(Tc), sigma Tc, and sigma_n terms

    def T_TES(self, P, Perr, n, nerr, k, kerr, Tb):   # value and error on T_TES measurement for a given power, n, k, and Tbath (& their errors)
        Ttes = (P/k+Tb**(n))**(1./n)
        Pterm = ((P/k+Tb**(n))**(1./n-1) / (k*n))**2 * Perr**2   # power term of sigma T_TES
        kterm = ((P/k+Tb**(n))**(1./n-1) * P / (n*k**2))**2  * kerr**2
        nterm = ( (P/k+Tb**(n))**(1./n) * ((Tb**n*np.log(Tb))/(n*(P/k+Tb**n)) - np.log(P/k+Tb**n)/n**2) )**2 * nerr**2
        sigma_Ttes = Pterm + kterm + nterm
        return Ttes, sigma_Ttes

    def k(self, GTc, GTcerr, n, nerr, Tc, Tcerr):   # value and error on T_TES measurement for a given power, n, k, and Tbath (& their errors)
        k = GTc/n * Tc**(1-n)
        Gterm = (1/(n*Tc**(n-1)))**2 * GTcerr**2   # power term of sigma T_TES
        Tcterm = (GTc/n*(n-1)*Tc**n)**2  * Tcerr**2
        nterm = ( GTc * Tc**(n-1)*(n*np.log(Tc)+1)/(n**2) )**2 * nerr**2
        sigma_k = Gterm + Tcterm + nterm
        return k, sigma_k


### functions to analyze all bolotest bolometer data at once

def load_data(dfiles):

    bays = ['BayAY', 'BayAX']
    tesids = []
    data = {}   # data dict

    for bb in np.arange(len(bays)):
        ### load data
        dfile = dfiles[bb]
        with open(dfile, "rb") as f:
            data_temp = pkl.load(f, encoding='latin1')
        tesids_temp = [(str(bays[bb])) + '_' + key for key in data_temp[bays[bb]]]
        if bays[bb] == 'BayAY':
            # remove noisy data (probably unlocked SQUID)
            tesids_temp.remove('BayAY_Row00')   # noisy
            tesids_temp.remove('BayAY_Row09')   # noisy
            tesids_temp.remove('BayAY_Row15')   # noisy
        elif bays[bb] == 'BayAX':
            # remove noisy data (probably unlocked SQUID)
            tesids_temp.remove('BayAX_Row05')
            tesids_temp.remove('BayAX_Row11')
            tesids_temp.remove('BayAX_Row14')
            tesids_temp.remove('BayAX_Row15')
            pass
        data[bays[bb]] = data_temp[bays[bb]]
        tesids.extend(tesids_temp)

    tesids = np.array(tesids)

    return data, tesids

def convert_rawdata(data, tesids, constT=True, v_nfit=0.3, save_figs=False, tran_perRn_start=0.2, perRn=np.array([25, 30, 40, 50, 60, 70, 80]),
                    fn_comments='', rm_fl=False, iv_plots=False, branch_plots=False, interp_plots=False, bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/'):
    ## sort raw data and convert to real units
    # v_nfit [V] = v_bias above which TES is normal (approximate)
    # tran_perRn_start = Fraction of Rn dubbed beginning of SC transition
    # Tbq_ind = index in Tbaths of Tbath to quote, 2 = 168 mK?

    # sort raw data and convert to real
    ivs = {}   # master dictionary

    for tesid in tesids:   # iterate through bolos

        tes = TESAnalyze()
        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        if bay=='BayAY':
            pad = str(int(tesid.split('Row')[1])+1)  # pad number
        elif bay=='BayAX':
            pad = str(int(tesid.split('Row')[1])+16)  # pad number
        boloid = pad_bolo_map[pad]   # map TES ID to pad # to bolo ID

        ivs[tesid] = {}
        ivs[tesid]['Pad'] = pad; ivs[tesid]['Bolometer'] = boloid   # save IDs to master dictionary
        tlabels = [key for key in data[bay][row]['iv']]
        if tesid == 'BayAY_Row12' or tesid == 'BayAX_Row10' or tesid == 'BayAX_Row03' or tesid == 'BayAX_Row00' or tesid == 'BayAX_Row06':
            tlabels.remove('iv014')   # wonky IV
        maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
        asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
        vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
        vtes = np.full(asize, np.nan); ites = np.full(asize, np.nan); rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); ptes = np.full(asize, np.nan)
        Tbaths = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(Tbaths))
        v_pnts = np.zeros((len(tlabels), len(perRn))); i_pnts = np.zeros((len(tlabels), len(perRn))); p_pnts = np.zeros((len(tlabels), len(perRn)))   # initialize interpolated IVs
        sigma_v = np.array([np.nan]*len(tlabels)); sigma_i = np.array([np.nan]*len(tlabels)); nfits_real = np.zeros((len(tlabels), 2))

        for tt, tlab in enumerate(tlabels):   # iterate through temperatures

            Tbaths[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, iv_plots=iv_plots, branch_plots=branch_plots, rm_fl=rm_fl)
            if iv_plots: plt.title('Bolo {boloid}, Pad {pad} - IV at {tlab} mK'.format(boloid=boloid, pad=pad, tlab=round(Tbaths[tt]*1E3)))

            nfits_real[tt] = np.polyfit(vtes[tt, norm_inds][0], ites[tt, norm_inds][0], 1)   # normal branch line fit in real IV units
            ifit_norm = vtes[tt, norm_inds]*nfits_real[tt, 0] + nfits_real[tt, 1]   # normal branch line fit
            rn_temp[tt] = np.nanmean(rtes[tt, norm_inds])   # ohms, should be consistent with 1/nfit_real[0]
            sigma_v[tt] = np.nanstd(vtes[tt,:end_sc-2])   # V, error in voltage measurement = std of SC branch
            sigma_i[tt] = np.nanstd(ites[tt,norm_inds] - ifit_norm)  # A, error in current measurement = std in normal branch after line subtraction
            v_pnts[tt], i_pnts[tt], p_pnts[tt] =  tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], perRn, rn_temp[tt], tran_perRn_start=tran_perRn_start, plot=False)

        if interp_plots:
            plt.figure()
            tsort = np.argsort(Tbaths)
            for tt in tsort:
                finds = np.isfinite(ites[tt])
                plt.plot(v_pnts[tt]*1e6, i_pnts[tt]*1e3, 'o', alpha=0.7, color=plt.cm.plasma((Tbaths[tt]-0.070)/(0.170)))   # Tbaths/max(Tbaths)
                plt.plot(vtes[tt][finds]*1e6, ites[tt][finds]*1e3, alpha=0.6, label='{} mK'.format(round(Tbaths[tt]*1E3,0)), color=plt.cm.plasma((Tbaths[tt]-0.070)/(0.170)))
            plt.xlabel('Voltage [$\mu$V]')
            plt.ylabel('Current [mA]')
            plt.title('Interpolated IV Points - Bolo {boloid}, Pad {pad}'.format(boloid=boloid, pad=pad))
            plt.legend()
            if save_figs: plt.savefig(bolotest_dir + 'Plots/IVs/pad' + pad + '_interpIVs' + fn_comments + '.png', dpi=300)

        Rn = np.nanmean(rn_temp); Rn_err = np.nanstd(rn_temp)   # estimate normal resistance

        # calculate TES power at interpolated IV points
        sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
        for ii, ipnt in enumerate(i_pnts):
            sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])

        # store results in dict
        sort_inds = np.argsort(Tbaths)   # sort by temp, ignore nans
        ivs[tesid]['TES ID']   = tesid
        ivs[tesid]['Tbaths']   = Tbaths[sort_inds]   # K
        ivs[tesid]['constT']   = constT
        ivs[tesid]['vbias']    = vbias[sort_inds]
        ivs[tesid]['vfb']      = vfb[sort_inds]
        ivs[tesid]['vtes']     = vtes[sort_inds]   # volts
        ivs[tesid]['ites']     = ites[sort_inds]   # amps
        ivs[tesid]['rtes']     = rtes[sort_inds]   # ohms
        ivs[tesid]['ptes']     = ptes[sort_inds]   # power
        ivs[tesid]['ites_int'] = i_pnts[sort_inds]   # interpolated IV points
        ivs[tesid]['vtes_int'] = v_pnts[sort_inds]   # interpolated IV points
        ivs[tesid]['ptes_int'] = p_pnts[sort_inds]   # power at interpolated IV points
        ivs[tesid]['ptes_err'] = sigma_p[sort_inds]
        ivs[tesid]['i_meas']   = i_meas[sort_inds]   # amps
        ivs[tesid]['sigma_i']  = sigma_i[sort_inds]   # amps
        ivs[tesid]['sigma_v']  = sigma_v[sort_inds]   # volts
        ivs[tesid]['Rn [mOhms]'] = Rn*1e3   # mOhms
        ivs[tesid]['Rn_err [mOhms]'] = Rn_err*1e3   # mohms

    return ivs


def fit_powerlaws(ivs, save_figs=False, fitGexplicit=True, perRn_toquote=80, Tb_inds=[0, 3, 8, 11, 15, 2], fn_comments='', Tbq_ind=2, constT=True,
                    perRn=np.array([25, 30, 40, 50, 60, 70, 80]), init_guess=np.array([1.E-10, 2.5, .170]), show_psatcalc=False, plot_fitparams=True,
                    bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/'):
    ### wrapper to fit power laws for all TESs and organize results
    # const_T = quote constant T_TES fit values vs values at a particular T_TES

    tesids = np.array([key for key in ivs.keys()])

    for tesid in tesids:

        tes = TESAnalyze()

        Rn      = ivs[tesid]['Rn [mOhms]']*1E-3; Rn_err = ivs[tesid]['Rn_err [mOhms]']*1E-3   # mohms
        p_pnts  = ivs[tesid]['ptes_int']; sigma_p = ivs[tesid]['ptes_err']   # TES power at % Rns from interpolated IV points
        ites    = ivs[tesid]['ites']; vtes = ivs[tesid]['vtes']; rtes = ivs[tesid]['rtes']   # current; voltage; resistance at each point in the IV
        sigma_i = ivs[tesid]['sigma_i']; sigma_v = ivs[tesid]['sigma_v']   # measured error on current; voltage

        Tbaths      = np.array(ivs[tesid]['Tbaths'])
        TbsToReturn = Tbaths[Tb_inds]   # Tbaths to plot
        Tb_toquote  = Tbaths[Tbq_ind]   # Tbath to quote results from
        print('Quoting Results at Tbath = {Tb} mK'.format(Tb = round(Tb_toquote, ndigits=4)*1E3))
        tind = np.where(TbsToReturn==Tb_toquote)[0][0]   # quote Psat (and fit params if constT=False) from IV at this Tbath
        qind = np.where(perRn==perRn_toquote)[0][0]   # save results from chosen %Rn fit

        ### fit power law
        pfig_path = bolotest_dir + 'Plots/Psat_fits/' + tesid + '_Pfit' + fn_comments + '.png' if save_figs else None
        GTcs, Ks, ns, Ttes, GTcs_err, Ks_err, ns_err, Ttes_err = tes.fitPowerLaw(perRn, Tbaths, p_pnts.T, init_guess, fitToLast=True, suptitle=tesid,
                plot=plot_fitparams, sigma=sigma_p.T, pfigpath=pfig_path, constT=constT, fitGexplicit=fitGexplicit)   # pass error to fitter
        if save_figs: plt.savefig(bolotest_dir + 'Plots/fit_params/' + tesid + '_fitparams' + fn_comments + '.png', dpi=300)

        if constT:
            GTc_toquote = GTcs[qind]; GTcerr_toquote = GTcs_err[qind]
            Tc_toquote = Ttes[qind]; Tcerr_toquote = Ttes_err[qind]
        else:
            GTc_toquote = GTcs[tind, qind]; GTcerr_toquote = GTcs_err[tind, qind]
            Tc_toquote = Ttes[tind, qind]; Tcerr_toquote = Ttes_err[tind, qind]

        print(' ')
        print(' ')
        print(tesid)
        print('G@Tc   = ', round(GTc_toquote*1e12, 2), ' +/- ', round(GTcerr_toquote*1e12, 2), 'pW/K')
        print('K      = ',  round(Ks[qind]*1E11, 3),   ' +/- ',  round(Ks_err[qind]*1E11, 3), ' E-11')
        print('n      = ', round(ns[qind], 2),         ' +/- ', round(ns_err[qind], 4))
        print('Tc     = ', round(Tc_toquote*1e3, 2),   ' +/- ',  round(Tcerr_toquote*1e3, 2), 'mK')
        print('TES Rn = ', round(Rn*1E3, 2),           ' +/- ', round(Rn_err*1E3, 2), ' mOhms')

        ### calculate Psat
        # find transition & normal branch
        sc_inds = np.where((rtes[Tbq_ind]/Rn)<.2)[0]
        start_ind = np.max(sc_inds)
        end_ind = np.max(np.where(((rtes[Tbq_ind]/Rn)>.2) & (rtes[Tbq_ind]!=np.nan)))
        vtes_tran = vtes[Tbq_ind, start_ind:end_ind]
        ites_tran = ites[Tbq_ind, start_ind:end_ind]

        # calculate Psat
        ptes_tran = vtes_tran * ites_tran
        sat_ind = np.where(ites_tran == np.min(ites_tran))[0][0]   # where the TES goes normal
        Psat = ptes_tran[sat_ind]
        Psat_err = tes.sigma_power(ites_tran[sat_ind], sigma_i[Tbq_ind], vtes_tran[sat_ind], sigma_v[Tbq_ind])
        Psat_calc = tes.Psat_atT(Tb_toquote, Tc_toquote, Ks[qind], ns[qind])
        Psat_156 = tes.Psat_atTandGTc(0.156, Tc_toquote, GTc_toquote, ns[qind])
        print('Psat@'+str(round(Tb_toquote*1e3))+'mK [pW] from IV = ', round(Psat*1e12, 4), ' +/- ', round(Psat_err*1e12, 4), 'pW')
        print('Psat@'+str(round(Tb_toquote*1e3))+'mK [pW] (calc)  = ', round(Psat_calc*1e12, 4), 'pW')
        print('Psat@156 mK [pW] (calc)  = ', round(Psat_156*1e12, 4), 'pW')
        print(' ')
        print(' ')

        if show_psatcalc:   # double-check Psat calculation (i've never seen this not work)
            plt.figure(figsize=[7,6])
            # plt.plot(vtes_tran.T*1e6, ites_tran.T/np.max(ites_tran), label='TES IV')
            # plt.xlabel('Voltage [$\mu$V]'); plt.ylabel('Normalized Current'); plt.legend()
            # plt.plot(vtes_tran.T*1e9, ites_tran.T*1e6, 'k', label='TES IV')
            iv_curve = plt.plot(vtes[Tbq_ind]*1e9, ites[Tbq_ind]*1e6, 'k', label='IV')
            plt.xlabel('Voltage [nV]'); plt.ylabel('Current [$\mu$A]')
            # plt.xlim(-2, max(vtes[Tbq_ind]*1e9)*1.03)
            # plt.ylim(-2, max(ites[Tbq_ind]*1e6)*1.03)

            # power
            ax1 = plt.gca(); ax2 = ax1.twinx()
            ax2.set_ylabel('TES Power [pW]')
            p_curve = ax2.plot(vtes_tran.T*1e9,        ptes_tran.T*1e12, 'k--', label='Power')
            psat_curve = ax2.plot(vtes_tran[sat_ind]*1e9, Psat*1e12, 'rx', markersize=10, mew=2, label='P$_{sat}$')
            # plt.ylim(-2, max(ptes_tran.T*1e12)*1.03)
            # plt.title('TES IV and Calculated Power at Tbath = ' + str(round(Tb_toquote*1000, 1)) + 'mK')
            plt.grid(linestyle = '--', which='both', linewidth = 0.5)
            # plt.title('$T_{bath}$ = ' + str(round(Tb_toquote*1000)) + ' mK')

            lns = iv_curve + p_curve + psat_curve
            labs = [l.get_label() for l in lns]
            plt.legend(lns, labs, loc='upper left', fontsize=16)
            if save_figs: plt.savefig(bolotest_dir + 'Plots/past_calc/' + tesid + '_psatcalc' + str(round(Tb_toquote*1000)) + ' mK' + fn_comments + '.png', dpi=300)

        # store results in dict
        ivs[tesid]['fitGexplicit'] = fitGexplicit
        ivs[tesid]['G@Tc [pW/K]'] = GTc_toquote*1e12
        ivs[tesid]['G_err@Tc [pW/K]'] = GTcerr_toquote*1e12
        ivs[tesid]['G@170mK [pW/K]'] = tes.scale_G(.170, GTc_toquote, Tc_toquote, ns[qind])*1e12
        ivs[tesid]['G_err@170mK [pW/K]'] = tes.sigma_GscaledT(.170, GTc_toquote, Tc_toquote, ns[qind], GTcerr_toquote, Tcerr_toquote, ns_err[qind])*1e12
        ivs[tesid]['G@156mK [pW/K]'] = tes.scale_G(.156, GTc_toquote, Tc_toquote, ns[qind])*1e12
        ivs[tesid]['G_err@156mK [pW/K]'] = tes.sigma_GscaledT(.156, GTc_toquote, Tc_toquote, ns[qind], GTcerr_toquote, Tcerr_toquote, ns_err[qind])*1e12
        ivs[tesid]['k'] = Ks[qind]
        ivs[tesid]['k_err'] = Ks_err[qind]
        ivs[tesid]['n'] = ns[qind]
        ivs[tesid]['n_err'] = ns_err[qind]
        ivs[tesid]['Tc [mK]'] = Tc_toquote*1e3
        ivs[tesid]['Tc_err [mK]'] = Tcerr_toquote*1e3
        ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat*1e12
        ivs[tesid]['Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat_err*1e12
        ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc'] =  Psat_calc*1e12
        ivs[tesid]['Psat@156mK [pW], Calc'] =  Psat_156*1e12
        ivs[tesid]['TbsToReturn'] = TbsToReturn*1E3   # mK
        ivs[tesid]['Tb_inds'] = Tb_inds
        # ivs[tesid]['Tb_toquote'] = Tb_toquote*1E3   # mK
        ivs[tesid]['Tb_toquote'] = Tb_toquote   # K
        ivs[tesid]['perRn'] = perRn
        ivs[tesid]['perRn_toquote'] = perRn_toquote

    return ivs

def savedata(ivs, pkl_file, csv_file):

    tesids_temp = np.array([key for key in ivs.keys()])
    tesdicts = [dict(tesd) for tesd in ivs.values()]   # make individual tes nested dicts scriptable
    Tb_toquote = ivs[tesids_temp[0]]['Tb_toquote']   # assumes this is the same for all TESs

    # sort CSV by pad number for readability
    # this is dumb but it works
    tesids    = [tesd.get('TES ID') for tesd in tesdicts]; boloids = [tesd.get('Bolometer') for tesd in tesdicts]; pads = [tesd.get('Pad') for tesd in tesdicts]
    Tcs       = [tesd.get('Tc [mK]') for tesd in tesdicts]; Tcerrs = [tesd.get('Tc_err [mK]') for tesd in tesdicts]
    Rns       = [tesd.get('Rn [mOhms]') for tesd in tesdicts]; Rnerrs = [tesd.get('Rn_err [mOhms]') for tesd in tesdicts]
    GTcs      = [tesd.get('G@Tc [pW/K]') for tesd in tesdicts]; GTcerrs = [tesd.get('G_err@Tc [pW/K]') for tesd in tesdicts]
    G170s     = [tesd.get('G@170mK [pW/K]') for tesd in tesdicts]; G170errs = [tesd.get('G_err@170mK [pW/K]') for tesd in tesdicts]
    G156s     = [tesd.get('G@156mK [pW/K]') for tesd in tesdicts]; G156errs = [tesd.get('G_err@156mK [pW/K]') for tesd in tesdicts]
    ks        = [tesd.get('k') for tesd in tesdicts]; kerrs = [tesd.get('k_err') for tesd in tesdicts]
    ns        = [tesd.get('n') for tesd in tesdicts]; nerrs = [tesd.get('n_err') for tesd in tesdicts]
    PsatIVs   = [tesd.get('Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV') for tesd in tesdicts]; Psaterrs = [tesd.get('Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV') for tesd in tesdicts]
    PsatCalcs = [tesd.get('Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc') for tesd in tesdicts]
    Psat156 =   [tesd.get('Psat@156mK [pW], Calc') for tesd in tesdicts]

    fields = np.array(['TES ID', 'Bolometer', 'Pad', 'Tc [mK]', 'Tc_err [mK]', 'Rn [mOhms]', 'Rn_err [mOhms]', 'G@Tc [pW/K]', 'G_err@Tc [pW/K]', 'G@170mK [pW/K]',
                        'G_err@170mK [pW/K]', 'G@156mK [pW/K]', 'G_err@156mK [pW/K]', 'k', 'k_err', 'n', 'n_err', 'Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV', 'Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV', 'Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc', 'Psat@156 mK [pW], Calc'])
    # rows = np.array([[ivs[tesids[pp]][field] for field in fields] for pp in np.argsort(pads)])
    rows = np.array([[tesids[pind], boloids[pind], pads[pind], Tcs[pind], Tcerrs[pind], Rns[pind], Rnerrs[pind], GTcs[pind], GTcerrs[pind], G170s[pind], G170errs[pind], G156s[pind], G156errs[pind],
                    ks[pind], kerrs[pind], ns[pind], nerrs[pind], PsatIVs[pind], Psaterrs[pind], PsatCalcs[pind], Psat156[pind]] for pind in np.argsort(pads)])
    # pdb.set_trace()

    with open(csv_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)  # csv writer object
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    # write pickle
    with open(pkl_file, 'wb') as pklfile:
        pkl.dump(ivs, pklfile)

    return ivs


def plotfit_singlebolo(ivs, tesid, save_figs=False, fn_comments='', xlims=[[-0.5, 200], [0, 200]], ylims=[[0, 90], [0, 0.7]],
                    perRn=np.array([25, 30, 40, 50, 60, 70, 80]),
                    bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/'):
    ### wrapper to fit power law of single bolometer and plot results only from chosen %Rn
    ### uses const_T = quote constant T_TES fit values vs values at a particular T_TES

    tes = TESAnalyze()   # initialize single bolo object

    Rn      = ivs[tesid]['Rn [mOhms]']*1E-3; Rn_err = ivs[tesid]['Rn_err [mOhms]']*1E-3   # mohms
    p_pnts  = ivs[tesid]['ptes_int']; sigma_p = ivs[tesid]['ptes_err']   # TES power at % Rns from interpolated IV points
    i_pnts  = ivs[tesid]['ites_int']; v_pnts = ivs[tesid]['vtes_int']   # interpolated IV points
    ites    = ivs[tesid]['ites'];     vtes = ivs[tesid]['vtes']; rtes = ivs[tesid]['rtes']   # current; voltage; resistance at each point in the IV
    sigma_i = ivs[tesid]['sigma_i'];  sigma_v = ivs[tesid]['sigma_v']   # measured error on current; voltage

    # analysis options
    Tb_toquote = ivs[tesid]['Tb_toquote']; TbsToReturn = ivs[tesid]['TbsToReturn']; Tb_inds = ivs[tesid]['Tb_inds']
    perRn = ivs[tesid]['perRn']; perRn_toquote = ivs[tesid]['perRn_toquote']; fitGexplicit = ivs[tesid]['fitGexplicit']

    Tc      = ivs[tesid]['Tc [mK]']; Tc_err = ivs[tesid]['Tc_err [mK]']
    n       = ivs[tesid]['n']; n_err = ivs[tesid]['n_err']
    k       = ivs[tesid]['k']; k_err = ivs[tesid]['k_err']
    GTc     = ivs[tesid]['G@Tc [pW/K]']; GTc_err = ivs[tesid]['G_err@Tc [pW/K]']
    G170    = ivs[tesid]['G@170mK [pW/K]']; G170_err = ivs[tesid]['G_err@170mK [pW/K]']
    G156    = ivs[tesid]['G@156mK [pW/K]']; G156_err = ivs[tesid]['G_err@156mK [pW/K]']
    PsatIV  = ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV']; Psaterr = ivs[tesid]['Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV']
    Psat_156 = ivs[tesid]['Psat@156mK [pW], Calc']; # Psaterr = ivs[tesid]['Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV']
    Tbaths  = np.array(ivs[tesid]['Tbaths'])
    prn_ind = np.where(perRn==perRn_toquote)[0][0]
    print('Quoting Results at Tbath = {Tb} mK'.format(Tb = round(Tb_toquote, ndigits=4)*1E3))

    print(' ')
    print(' ')
    print(tesid)
    print('G@Tc =        ', round(GTc, 2), ' +/- ', round(GTc_err, 2), 'pW/K')
    print('G@170 mK =    ', round(G170, 2), ' +/- ', round(G170_err, 2), 'pW/K')
    print('G@156 mK =    ', round(G156, 2), ' +/- ', round(G156_err, 2), 'pW/K')
    print('Psat@'+str(round(Tb_toquote*1e3))+'mK [pW] = ', round(PsatIV*1E3, 1), ' +/- ', round(Psaterr*1E3, 1), 'aW')
    print('Psat@156 mK [pW] = ', round(Psat_156, 3), ' pW')
    print('Tc =          ', round(Tc, 2), ' +/- ',  round(Tc_err, 2), 'mK')
    print('TES Rn =      ', round(Rn*1E3, 2), ' +/- ', round(Rn_err*1E3, 4), ' mOhms')
    print('n =           ', round(n, 2), ' +/- ', round(n_err, 2))
    print('k =           ', round(k*1E12, 4), ' +/- ', k_err*1E12, ' pW/K^n/um')

    # plot IVs
    plt.figure(figsize=[7,6])
    tsort = np.argsort(Tbaths)
    for tt in tsort:
        finds = np.isfinite(ites[tt])
        # plt.plot(vtes[tt][finds]*1e9, ites[tt][finds]*1e6, alpha=0.8, label='{} mK'.format(round(Tbaths[tt]*1E3)), color=plt.cm.plasma((Tbaths[tt]-0.1)/max(Tbaths-0.08)))   # IVs
        plt.plot(vtes[tt][finds]*1e9, ites[tt][finds]*1e6, alpha=0.8, label='{} mK'.format(round(Tbaths[tt]*1E3)), color=plt.cm.plasma((Tbaths[tt]-0.05)/max(Tbaths-0.02)))   # IVs
    v_pRnplot = np.append(0, max(v_pnts[:,prn_ind]))*5e9
    i_pRnplot = np.append(0, max(i_pnts[:,prn_ind]))*5e6
    plt.plot(v_pRnplot, i_pRnplot, 'k--', label='{}\% R$_N$'.format(perRn_toquote))
    plt.xlabel('Voltage [nV]')
    plt.ylabel('Current [$\mu$A]')
    # plt.xlim(-0.5, 200); plt.ylim(0, 90)
    plt.xlim(xlims[0]); plt.ylim(ylims[0])

    handles, labels = plt.gca().get_legend_handles_labels()
    linds = np.append(Tb_inds, -1)   # include %Rn label
    plt.legend([handles[ii] for ii in linds], [labels[ii] for ii in linds], loc=(0.15, 0.45), fontsize=16)
    # plt.legend([handles[ii] for ii in linds], [labels[ii] for ii in linds], loc=(0.1, 0.53), fontsize=16)

    # power law fit subplot
    if fitGexplicit:
        fitfnct = tes.tespowerlaw_fitfunc_G
        pfit = np.array([GTc*1E-12, n, Tc*1E-3])
    else:
        fitfnct = tes.tespowerlaw_fitfunc_k
        pfit = np.array([k, n, Tc*1E-3])

    # power law fit inset
    # ax = plt.axes([0.625, 0.425, .25, .35])
    ax = plt.axes([0.61, 0.47, .25, .35])
    # temp_pnts = np.linspace(min(Tbaths)*1E-3 - 0.01, Tc*1E-3+0.01,25)
    temp_pnts = np.linspace(0, Tc*1E-3+0.01,25)
    plt.plot(temp_pnts*1e3, fitfnct(temp_pnts, *pfit)*1e12, color='k')   # power law fit
    plt.errorbar(Tbaths*1e3, p_pnts[:,prn_ind]*1e12, yerr=sigma_p[:,prn_ind]*1e12, fmt='o', color='k')   # data points
    plt.xlabel('Bath Temp [mK]', fontsize=14)
    plt.ylabel('Power [pW]', fontsize=14)
    plt.title('Power Law Fit', fontsize=14)
    # plt.xlim(0, 200); plt.ylim(0, 0.7)
    plt.xlim(xlims[1]); plt.ylim(ylims[1])
    ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=14)
    if save_figs: plt.savefig(bolotest_dir + 'Plots/IVs/tes' + tesid + '_IVsandfit' + fn_comments + '.png', dpi=300)

    return ivs