import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit
import pylab
import pdb
from collections import OrderedDict
# from time import sleep
# import math
# from scipy import linspace, stats, fftpack, optimize, interpolate, odr
# from scipy.signal import hilbert

# if you give it sigma, it just uses the relative values of those errors, but pcov is rescaled by the sample variance of the data such that you get reduced chi-squared equal to one
# if you want your pcov that comes out to acutally use the sigmas you're putting in and take them as real sigma, set aboslute_sigma=True
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
        n_output = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot)   # *
        n_fit = n_output[0]; norm_inds = n_output[1]   # *
        # n_fit = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot)
        if n_fit[0] < 0:
            # print('normal branch has negative slope, flipping v_fb')
            vfb = -vfb
            n_output = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot)   # *
            n_fit = n_output[0]; norm_inds = n_output[1]  # *
            # n_fit = self.ivFitNormal(vbias, vfb, vbias_offset, v_nfit, show_plot=show_plot)   # *
        normal_offset = n_fit[1]
        sc_fit, end_sc = self.ivFitSuperconduct(vbias, vfb, vbias_offset, show_plot=show_plot, ignore_nans=ignore_nans)   # *
        # sc_fit = self.ivFitSuperconduct(vbias, vfb, vbias_offset, show_plot=show_plot, ignore_nans=ignore_nans)   # *
        m_sc = sc_fit[0]
        if rpar is None:
            rpar = rsh*(rfb*mr/(rbias*m_sc)-1.0)
            # print('Rpar = ', rpar)  
        # pdb.set_trace()         
        vtes, ites, rtes, ptes, i_meas = self.ivConvertTDM(vbias, vfb, rfb, rbias, rsh, mr, rpar, normal_offset, vbias_offset, i_offset)
        
        return vtes, ites, rtes, ptes, i_meas, n_fit, norm_inds, sc_fit, end_sc, rpar   # *
        # return vtes, ites, rtes, ptes, i_meas, n_fit, sc_fit, rpar   # *

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
        
        return n_fit, norm_inds   # *
        # return n_fit   # *
    
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
            end_sc =  min(neg_slope_array[0]) - 2 # Go down two point to get away from the edge
        vbias_sc = vbias[:end_sc] - vbias_offset
        vfb_sc = vfb[:end_sc]
        sc_m, sc_b = scipy.polyfit(vbias_sc, vfb_sc, 1)
        vfb_sc_fit = scipy.polyval([sc_m,sc_b],vbias_sc)
        sc_fit = [sc_m, sc_b]
        if show_plot is True:
            pylab.plot(vbias_sc,vfb_sc_fit, label='SC Branch Fit')
            pylab.legend()
        
        return sc_fit, end_sc   # *
        # return sc_fit   # *
        

    def ivInterpolate(self, vtes, ites, rtes, percentRns, rn, tran_pRn_start=0.050):
      
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
        pylab.plot(v_pnts*1e6, i_pnts*1e3, 'o', alpha=0.8)
        
        return v_pnts, i_pnts, p_pnts

    def ivPRnLinearFit(self, v_pnt_array, i_pnt_array):
        
        pylab.figure()
        for index in range(len(v_pnt_array[:,0])):
            pylab.plot(v_pnt_array[index]*1e6, i_pnt_array[index]*1e3,'o')
            rvt_m, rvt_b = scipy.polyfit(v_pnt_array[index], i_pnt_array[index], 1)
            ites_rvt_fit = scipy.polyval([rvt_m,rvt_b],v_pnt_array[index])
            rvt_residuals = i_pnt_array[index]-ites_rvt_fit
            pylab.plot(v_pnt_array[index]*1e6, ites_rvt_fit*1e3)
            #pylab.plot(v_pnt_array[index], rvt_residuals,'o')
        pylab.title('Fits to percent Rn values')
        pylab.xlabel('VTES (mu)')
        pylab.ylabel('ITES (mA)') 

    def fitPowerLaw(self, pRn, temperatures, powerAtRns, init_guess, fitToLast=True, TbsToReturn=[0.080,0.094,0.095], plot=True, suptitle='', sigma=None, nstd=1):
        '''Fit power versus T_base with power law.'''

        if plot is True:
            pylab.figure(figsize=(20,16))
        
        Ks = np.zeros(len(pRn)); ns = np.zeros(len(pRn)); Tcs = np.zeros(len(pRn)); Gs = np.zeros(len(pRn))   # initialize arrays
        Ks_error = np.zeros(len(pRn)); ns_error = np.zeros(len(pRn)); Tcs_error = np.zeros(len(pRn))
        for index in range(len(pRn)):   # for each percent Rn
            pfit, pcov = curve_fit(self.powerlaw_fit_func, temperatures, powerAtRns[index], p0=init_guess, sigma=sigma[index], absolute_sigma=True)   # non-linear least squares fit
            Ks[index] = pfit[0]; ns[index] = pfit[1]; Tcs[index] = pfit[2]   # fit parameters
            perr = np.sqrt(np.diag(pcov)); Ks_error[index] = perr[0]; ns_error[index] = perr[1]; Tcs_error[index] = perr[2]   # error of fit
            temp_pnts = np.linspace(min(temperatures)-0.01,Tcs[index]+0.01,25)
            if plot:
                pylab.errorbar(temperatures*1e3, powerAtRns[index]*1e12, xerr=sigma[index]*1e12, fmt='o', label='Data', color=mcolors[index+6], alpha=0.8)   # matching data and model colors
                pylab.plot(temp_pnts*1e3, self.powerlaw_fit_func(temp_pnts, *pfit)*1e12, label='Power Law', color=mcolors[index+6], alpha=0.8)
                # prepare confidence level curves
                # pfit_up = pfit + nstd * perr; pfit_dw = pfit - nstd * perr
                # fit_up = self.powerlaw_fit_func(temp_pnts, *pfit_up); fit_down = self.powerlaw_fit_func(temp_pnts, *pfit_dw)
                # pylab.fill_between(temp_pnts*1e3, fit_down*1e12, fit_up*1e12, alpha=.3, label='{}-sigma interval'.format(nstd), color=mcolors[index+6])
            handles, labels = pylab.gca().get_legend_handles_labels()   # deal with redundant labels
            by_label = OrderedDict(zip(labels, handles))
            pylab.legend(by_label.values(), by_label.keys())
        # Gs = ns*Ks*Tcs**(ns-1)
        # pdb.set_trace()
        Gs, Gs_err = self.GatTc(ns, ns_error, Ks, Ks_error, Tcs, Tcs_error)

        if fitToLast:
            # Find the T and G for a given Tb using the highest n and K in the transistion
            # Find Tbs matching temperatures where data was taken
            Tbs = np.zeros(len(TbsToReturn))
            Tbs_index = np.zeros(len(TbsToReturn),int)
            for Tind, Tb in enumerate(TbsToReturn):
                Tb_index = np.where(Tb == temperatures)
                if len(Tb_index[0]) > 0:
                    # Tbs_index = np.hstack((Tbs_index, int(Tb_index[0][0])))
                    # Tbs = np.hstack((Tbs, temperatures[int(Tb_index[0][0])]))
                    Tbs_index[Tind] = int(Tb_index[0][0]); Tbs[Tind] = temperatures[Tb_index[0][0]]
                else:
                    print('Could not find Tbath=', Tb)
            # print(Tbs)
            
            K = Ks[-1]
            n = ns[-1]
            Ts = np.zeros((len(Tbs),len(pRn)))
            Gs_Tb = np.zeros((len(Tbs),len(pRn))); Gs_Tb_err = np.zeros((len(Tbs),len(pRn)))
            for index in range(len(Tbs)):
                Tb = Tbs[index]
                Tb_index = Tbs_index[index]
                Ts[index] = (powerAtRns[:,Tb_index]/K+Tb**(n))**(1./n)
                Gs_Tb[index] = n*K*(Ts[index])**(n-1.)
                Gs_Tb[index] = Gs[index]; Gs_Tb_err[index] = Gs_err[index]
        
        if plot:
            pylab.xlabel('Bath Temperature [mK]')
            pylab.ylabel('Power [pW]')
            pylab.title(suptitle)
            f1 = pylab.figure(figsize=(7,6))
            p1 = f1.add_subplot(2,2,1)
            p1.plot(pRn,Ks*1e12,'o')
            pylab.title('$\kappa$')
            pylab.ylabel('$\kappa$ [$pW/K^n$]')
            p2 = f1.add_subplot(2,2,2)
            p2.plot(pRn,ns,'o')
            pylab.title('n')
            pylab.ylabel('n')
            p3 = f1.add_subplot(2,2,3)
            p3.plot(pRn,Tcs,'o', label='constant T')
            for index in range(len(Ts)):
                p3.plot(pRn,Ts[index],'o',label=str(round(Tbs[index]*1e3, 1))+' mK')
            pylab.title('$T_c$')
            pylab.xlabel('% $R_n$')
            pylab.ylabel('$T_c$ [$K$]')  
            leg3 = pylab.legend(loc='best')
            try:
                for t in leg3.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')
                
            p4 = f1.add_subplot(2,2,4)
            p4.plot(pRn,Gs*1e12,'o',label='constant T')
            for index in range(len(Ts)):
                p4.plot(pRn,Gs_Tb[index]*1e12,'o',label='{tt:.1f} mK'.format(tt=Tbs[index]*1e3))
            pylab.title('G')
            pylab.xlabel('% $R_n$')
            pylab.ylabel('G [pW/K]')
            leg4 = pylab.legend(loc='best')
            try:
                for t in leg4.get_texts():
                    t.set_fontsize('small')
            except:
                print('Legend error')
            pylab.suptitle(suptitle, x=.55, y=.99, fontsize=14)
            f1.tight_layout()        
            
        # return Gs_Tb, Ks, ns, Ts, Gs_Tb_err, Ks_error, ns_error, Tcs_error
        return Gs_Tb, Ks, ns, Tcs, Gs_Tb_err, Ks_error, ns_error, Tcs_error
        
#************************


    def powerlaw_fit_func(self, x, k, n, Tc):
        return k*(Tc**n-np.power(x,n))   # P = kappa * (T^n - x^n)

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
    
    def GatTc(self, n, sigma_n, k, sigma_k, Tc, sigma_Tc):
        G = n*k*Tc**(n-1)
        sigma_G = np.sqrt( G**2 * ((sigma_n/n)**2 + (sigma_k/k)**2 + (n-1)**2*(sigma_Tc/Tc)**2) ) 
        return G, sigma_G
