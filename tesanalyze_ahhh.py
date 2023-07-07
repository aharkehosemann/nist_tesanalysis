import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit
import pylab
import pdb
from collections import OrderedDict

""" 
Adapted from Doug Bennett's TES IV analysis module, now with error analysis!
AHHH

If curve_fit is just given sigma, it just uses the relative values of those errors, but pcov is rescaled by the sample variance of the data such that you get reduced chi-squared equal to one.
Set absolute_sigma=True to get true error.
"""

kB = 1.3806503e-23   # Boltzmann constant
mcolors = pylab.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

class TESAnalyze():
    '''
    Analysis functions
    '''

    def ivAnalyzeTDM(self, vbias, vfb, rfb, rbias, rsh, mr, v_nfit, rpar=None, vbias_offset=0.0,
                     i_offset=0.0, show_plot=False, ignore_nans=True):
        '''Fit supeconducting and normal branches and use information to convert raw data to IV.'''

        vb_step = vbias[1]-vbias[0]
        if vb_step < 0:   # reverse order if data was taken from high V -> 0
            vbias = vbias[::-1]
            vfb = vfb[::-1] 
        n_output = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot) 
        n_fit = n_output[0]; norm_inds = n_output[1]   
        if n_fit[0] < 0:
            vfb = -vfb
            n_output = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot)  
            n_fit = n_output[0]; norm_inds = n_output[1] 
        normal_offset = n_fit[1]
        sc_fit, end_sc = self.ivFitSuperconduct(vbias, vfb, vbias_offset, show_plot=show_plot, ignore_nans=ignore_nans)  
        m_sc = sc_fit[0]
        if rpar is None:
            rpar = rsh*(rfb*mr/(rbias*m_sc)-1.0)
        vtes, ites, rtes, ptes, i_meas = self.ivConvertTDM(vbias, vfb, rfb, rbias, rsh, mr, rpar, normal_offset, vbias_offset, i_offset)
        
        return vtes, ites, rtes, ptes, i_meas, n_fit, norm_inds, sc_fit, end_sc, rpar  

    def ivConvertTDM(self, vbias, vfb, rfb, rbias, rsh, mr, rpar, normal_offset, vbias_offset=0.0, i_offset=0.0):
        '''Convert IV data.'''
        
        vb_step = vbias[1]-vbias[0]
        if vb_step < 0:
            vbias = vbias[::-1]
            vfb = vfb[::-1] 
        i_meas = (vfb-normal_offset)/(rfb*mr) + i_offset
        ites = i_meas - i_offset
        i_bias = (vbias-vbias_offset)/rbias
        vtes = (i_bias-i_meas)*rsh-i_meas*rpar
        rtes = vtes/ites
        ptes = vtes*ites
        
        return  vtes, ites, rtes, ptes, i_meas
        
    
    def ivFitNormal(self, vbias, vfb, vbias_offset, v_nfit, show_plot=False):
        '''Fit the normal branch of the IV.'''

        x = vbias-vbias_offset
        y = vfb
        norm_inds = np.where(x>v_nfit)
        m_a, b_a = np.polyfit(x[norm_inds], y[norm_inds], 1)
        n_fit = [m_a, b_a]
        xf = x[norm_inds]
        yfa = xf*m_a+b_a
        if show_plot is True:
            pylab.figure()
            pylab.plot(x,y,'.', label='Data')
            pylab.plot(xf,yfa, label='Normal Branch Fit')
        
        return n_fit, norm_inds   
    
    def ivFitSuperconduct(self, vbias, vfb, vbias_offset, show_plot=False, ignore_nans=True):
        '''Fit the superconducting branch of the IV and return the slope and intercept. '''

        fb_diff = np.diff(vfb)
        nan_indicies = np.where(np.isnan(vfb))
        if len(nan_indicies[0]) > 0 and ignore_nans==False:
            # transition data is at values larger than the NaNs, set ignore_nans to False to turn this on
            print('not igonring nans')
            end_sc = int(min(nan_indicies[0])-1)
        else:       
            neg_slope_array = np.where(fb_diff<0)
            end_sc =  min(neg_slope_array[0]) - 2 # Go down two points to get away from the edge
        vbias_sc = vbias[:end_sc] - vbias_offset
        vfb_sc = vfb[:end_sc]
        sc_m, sc_b = scipy.polyfit(vbias_sc, vfb_sc, 1)
        vfb_sc_fit = scipy.polyval([sc_m,sc_b],vbias_sc)
        sc_fit = [sc_m, sc_b]
        if show_plot is True:
            pylab.plot(vbias_sc,vfb_sc_fit, label='SC Branch Fit')
            pylab.legend()
        
        return sc_fit, end_sc   
        

    def ivInterpolate(self, vtes, ites, rtes, percentRns, rn, tran_pRn_start=0.050, plot=False):
      
        btran_indices = np.where((rtes/rn)<tran_pRn_start)
        if len(btran_indices[0]) == 0:   # check for normal IVs, this throws off the interpolator
            print('TES is normal, interpolator failed')
            return None, None, None
        tran_start = np.max(btran_indices)
        vtes_tran = vtes[tran_start:]
        ites_tran = ites[tran_start:]
        rtes_tran = rtes[tran_start:]
        # Check that the r data being used for interpolation is monotonically increasing
        if np.all(np.diff(rtes_tran) > 0) is False:
            print('IV %s not monotomicaly increasing in R')
        # Find the I and V that corespond to the requested percent Rns
        v_pnts = np.interp(percentRns/100.0*rn, rtes_tran, vtes_tran)
        i_pnts = np.interp(percentRns/100.0*rn, rtes_tran, ites_tran)
        p_pnts = v_pnts*i_pnts
        if plot: pylab.plot(v_pnts*1e6, i_pnts*1e3, 'o', alpha=0.8)
        
        return v_pnts, i_pnts, p_pnts

    def ivPRnLinearFit(self, v_pnt_array, i_pnt_array):
        
        pylab.figure()
        for index in range(len(v_pnt_array[:,0])):
            pylab.plot(v_pnt_array[index]*1e6, i_pnt_array[index]*1e3,'o')
            rvt_m, rvt_b = scipy.polyfit(v_pnt_array[index], i_pnt_array[index], 1)
            ites_rvt_fit = scipy.polyval([rvt_m,rvt_b],v_pnt_array[index])
            rvt_residuals = i_pnt_array[index]-ites_rvt_fit
            pylab.plot(v_pnt_array[index]*1e6, ites_rvt_fit*1e3)
        pylab.title('Fits to percent Rn values')
        pylab.xlabel('VTES (mu)')
        pylab.ylabel('ITES (mA)') 

    def fitPowerLaw(self, pRn, temperatures, powerAtRns, init_guess, fitToLast=True, TbsToReturn=[0.080,0.094,0.095], plot=True, suptitle='', sigma=None, nstd=1, pfigpath=None, constT=False, fitGexplicit=False, fixedK=False, fixedn=False):
        '''Fit power versus T_base with power law.'''

        if plot is True:
            f1 = pylab.figure(figsize=(6,6))   # power law fit figure
        
        if fitGexplicit:
            fitfnct = self.tespowerlaw_fit_func_G 
        elif fixedK:   # fix K, fit for n and Tc
            fitfnct = lambda x, n, Tc: self.tespowerlaw_fit_func_k(x, fixedK, n, Tc)
        elif fixedn:   # fix K, fit for n and Tc
            fitfnct = lambda x, k, Tc: self.tespowerlaw_fit_func_k(x, k, fixedn, Tc)
        else: 
            fitfnct = self.tespowerlaw_fit_func_k
        # pdb.set_trace()
        Ks = np.zeros(len(pRn)); ns = np.zeros(len(pRn)); Tcs = np.zeros(len(pRn)); GTcs = np.zeros(len(pRn))   # initialize arrays
        Ks_error = np.zeros(len(pRn)); ns_error = np.zeros(len(pRn)); Tcs_error = np.zeros(len(pRn)); GTcs_err = np.zeros(len(pRn))
        for index in range(len(pRn)):   # for each percent Rn, assumes constant T_TES that is very close to Tc
            if type(sigma) is np.ndarray: # measurement error included in fit
                # pdb.set_trace()
                p0, p0cov = curve_fit(fitfnct, temperatures, powerAtRns[index], p0=init_guess)   # get global p0 values by neglecting measurement errors first
                pfit, pcov = curve_fit(fitfnct, temperatures, powerAtRns[index], p0=p0, sigma=sigma[index], absolute_sigma=True)   # non-linear least squares fit, set absolute_sigma=True to get true variance in pcov
                # print(''); print(pcov); print('')
                # pdb.set_trace()
            else: # don't include measurement error
                pfit, pcov = curve_fit(fitfnct, temperatures, powerAtRns[index], p0=init_guess)   # non-linear least squares fit, set absolute_sigma=True to get true variance in pcov
            perr = np.sqrt(np.diag(pcov))   # error of fit
            if fitGexplicit:
                GTcs[index] = pfit[0]; ns[index] = pfit[1]; Tcs[index] = pfit[2]   # fit parameters
                GTcs_err[index] = perr[0]; ns_error[index] = perr[1]; Tcs_error[index] = perr[2]   # error of fit parameters
            elif fixedK:   # only fit n and Tc
                Ks[index] = fixedK; Ks_error[index] = 0
                ns[index] = pfit[0]; Tcs[index] = pfit[1]   # fit parameters
                ns_error[index] = perr[0]; Tcs_error[index] = perr[1]   # error of fit parameters               
            elif fixedn:   # only fit n and Tc
                ns[index] = fixedn; ns_error[index] = 0
                Ks[index] = pfit[0]; Tcs[index] = pfit[1]   # fit parameters
                Ks_error[index] = perr[0]; Tcs_error[index] = perr[1]   # error of fit parameters 
            else:
                Ks[index] = pfit[0]; ns[index] = pfit[1]; Tcs[index] = pfit[2]   # fit parameters
                Ks_error[index] = perr[0]; ns_error[index] = perr[1]; Tcs_error[index] = perr[2]   # error of fit parameters
            temp_pnts = np.linspace(min(temperatures)-0.01,Tcs[index]+0.01,25)
            if plot:
                if type(sigma) is np.ndarray:
                    pylab.errorbar(temperatures*1e3, powerAtRns[index]*1e12, yerr=sigma[index]*1e12, fmt='o', color=pylab.cm.viridis(pRn[index]/100-0.05), alpha=0.8)   # matching data and model colors
                else:
                    pylab.errorbar(temperatures*1e3, powerAtRns[index]*1e12, fmt='o', color=pylab.cm.viridis(pRn[index]/100-0.05), alpha=0.8)   # matching data and model colors
                pylab.plot(temp_pnts*1e3, fitfnct(temp_pnts, *pfit)*1e12, label='{}% Rn'.format(pRn[index]), color=pylab.cm.viridis(pRn[index]/100-0.05), alpha=0.8)
                # prepare confidence level curves
                # pfit_up = pfit + nstd * perr; pfit_dw = pfit - nstd * perr
                # fit_up = self.tespowerlaw_fit_func_k(temp_pnts, *pfit_up); fit_down = self.tespowerlaw_fit_func_k(temp_pnts, *pfit_dw)
                # pylab.fill_between(temp_pnts*1e3, fit_down*1e12, fit_up*1e12, alpha=.3, label='{}-sigma interval'.format(nstd), color=mcolors[index+6])
        if plot:
            pylab.legend()
            pylab.xlabel('Bath Temperature [mK]')
            pylab.ylabel('Power [pW]')
            pylab.title(suptitle)
            if pfigpath: pylab.savefig(pfigpath, dpi=300)
        
        if fitGexplicit:
            Ks, Ks_err = self.k(GTcs, GTcs_err, ns, ns_error, Tcs, Tcs_error)   # k for each % Rn
        else:
            GTcs, GTcs_err = self.GatTc(ns, ns_error, Ks, Ks_error, Tcs, Tcs_error)   # G(Tc) for each %Rn

        if fitToLast:
            # Find the T and G for a given Tb using the highest n and K in the transistion
            # Find Tbs matching temperatures where data was taken
            Tbs = np.zeros(len(TbsToReturn))
            Tbs_index = np.zeros(len(TbsToReturn),int)
            for Tind, Tb in enumerate(TbsToReturn):
                Tb_index = np.where(Tb == temperatures)
                if len(Tb_index[0]) > 0:
                    Tbs_index[Tind] = int(Tb_index[0][0]); Tbs[Tind] = temperatures[Tb_index[0][0]]
                else:
                    print('Could not find Tbath=', Tb)
            
            K = Ks[-1]; Kerr = Ks_error[-1]
            n = ns[-1]; nerr = ns_error[-1]

            Ttes = np.zeros((len(Tbs),len(pRn))); Ttes_error = np.zeros((len(Tbs),len(pRn)))   # T_TES is a function of %Rn and Tbath
            GTbs = np.zeros((len(Tbs),len(pRn))); GTbs_err = np.zeros((len(Tbs),len(pRn)))   # G(T_TES) for each Tb and %Rn
            
            for bb in range(len(Tbs)):
                Tb = Tbs[bb]
                Tb_index = Tbs_index[bb]            
                Ttes[bb], Ttes_error[bb] = self.T_TES(powerAtRns[:, Tb_index], sigma[:, Tb_index], n, nerr, K, Kerr, Tb)   # T_TES for all % Rn at one Tbath
                GTbs[bb], GTbs_err[bb] = self.GatTc(n, nerr, K, Kerr, Ttes[bb], Ttes_error[bb])    # G varies with Tbath because T_TES varies with Tbath
        # pdb.set_trace()
        if plot:
            tsort = np.argsort(Tbs)   # plot lines in order of bath temp
            f2 = pylab.figure(figsize=(7,6))   # fit parameters
            p1 = f2.add_subplot(2,2,1)
            p1.errorbar(pRn, Ks*1e12, yerr=Ks_error*1E12, fmt='o')
            pylab.title('k')
            pylab.ylabel('k [$pW/K^n$]')
            p2 = f2.add_subplot(2,2,2)
            p2.errorbar(pRn, ns, yerr=ns_error, fmt='o')
            pylab.title('n')
            pylab.ylabel('n')
            p3 = f2.add_subplot(2,2,3)
            for tt in tsort:
                p3.errorbar(pRn, Ttes[tt]*1E3, yerr=Ttes_error[tt]*1E3, fmt='o', label='{:.1f} mK'.format(Tbs[tt]*1e3), alpha=0.7, color=pylab.cm.plasma((Tbs[tt]-min(Tbs)*1)/(max(Tbs)*0.8)))
            p3.errorbar(pRn, Tcs*1E3, yerr=Tcs_error*1E3, fmt='o', label='constant T', alpha=0.7, color='k')
            pylab.title('$T_{TES}$')
            pylab.xlabel('% $R_n$')
            pylab.ylabel('$T_{TES}}$ [$mK$]')  
            leg3 = pylab.legend(loc='upper left')
            try:
                for t in leg3.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')
                
            p4 = f2.add_subplot(2,2,4)
            for tt in tsort:
                p4.errorbar(pRn, GTbs[tt]*1e12, yerr=GTbs_err[tt]*1E12, fmt='o',label='{:.1f} mK'.format(Tbs[tt]*1e3), alpha=0.7, color=pylab.cm.plasma((Tbs[tt]-min(Tbs)*1)/(max(Tbs)*0.8)))
            p4.errorbar(pRn, GTcs*1e12, yerr=GTcs_err*1E12, fmt='o', label='constant T', alpha=0.7, color='k')
            pylab.title('G')
            pylab.xlabel('% $R_n$')
            pylab.ylabel('G [pW/K]')
            leg4 = pylab.legend(loc='upper left')
            try:
                for t in leg4.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')
            pylab.suptitle(suptitle, x=.55, y=.95)
            f2.tight_layout()        
            
        if constT:   # return constant T fit values of G(Tc) (can be less accurate at lower %Rn)
            return GTcs, Ks, ns, Tcs, GTcs_err, Ks_error, ns_error, Tcs_error
        else:
            return GTbs, Ks, ns, Ttes, GTbs_err, Ks_error, ns_error, Ttes_error


#************************


    def tespowerlaw_fit_func_k(self, x, k, n, Tc):
        return k*(Tc**n-np.power(x,n))   # P = kappa * (T^n - x^n)
    
    def tespowerlaw_fit_func_G(self, x, G, n, Tc):
        return G/n * Tc *(1-np.power(x/Tc,n))   # P = G/n * Tc * (1 - (Tb/Tc)^n)


    # def powerlaw_err_func(self, k, n, Tc, x, y):  
    #     return y - self.powerlaw_fit_func(p,x) 

    def calculateAlphaFromIV(self, percentRn, Ts_array, rn, plot=True):
        '''Calculates alpha from Tc values for each percent Rn'''
        
        if plot is True:
            pylab.figure()
        
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
                pylab.plot(perRnMid,alpha, 'o')

        if plot is True:
            pylab.title('Alpha from extracted TC values')
            pylab.xlabel('% $R_n$')
            pylab.ylabel('Alpha')
                
        return alpha_array, perRnMid, Tmid_array
    

    def sigma_power(self, i, sigma_i, v, sigma_v):   # calculate error in power measurement from error in voltage and current measurement
        return np.sqrt(np.multiply(i**2,sigma_v**2) + np.multiply(v**2,sigma_i**2))   # sigma_power^2 = (I*sigma_V)^2 + (V*sigma_I)^2

    def Psat_atT(self, T, Tc, k, n):
        return k*(Tc**n-T**n)

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
