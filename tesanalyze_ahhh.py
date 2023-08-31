import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit
import pylab
import pdb
from collections import OrderedDict
import pickle as pkl

""" 
Adapted from Doug Bennett's TES IV analysis module, now with error analysis!
AHHH

If curve_fit is just given sigma, it just uses the relative values of those errors, but pcov is rescaled by the sample variance of the data such that you get reduced chi-squared equal to one.
Set absolute_sigma=True to get true error.
"""

kB = 1.3806503e-23   # Boltzmann constant
mcolors = pylab.rcParams['axes.prop_cycle'].by_key()['color']*5   # iterate through matplotlib default colors

# map pad number to bolo id
pad_bolo_map = {'1':'1f', '2':'1e', '3':'1d', '4':'1c', '5':'1b', '6':'1a', '7':'1f', '8':'1e', '9':'1d', '10':'1c', '11':'1b', '12':'1a', 
                '13':'24', '14':'23', '15':'22', '16':'13', '17':'21', '18':'20', '19':'24', '20':'23', '21':'22', '22':'13', '23':'21', 
                '24':'20','25':'7','26':'26','27':'27','28':'7','29':'26','30':'27'}

# readout circuit parameters
M_r = 8.5   # (SQUID-input coupling: current = arbs / (M_r * R_fb))
Rfb = 2070.   # Ohms (SQUID feeback resistor), Kelsey: accurate to ~5%
Rsh = 370.e-6   # microOhms (bias circuit shunt), Kelsey: accurate to 10-15%
Rb = 1020.   # Ohms (TES bias resistor), Kelsey: accurate to ~5%

class TESAnalyze():
    '''
    Analysis functions
    '''

    def ivAnalyzeTDM(self, Vbias, Vfb, rfb, rbias, rsh, mr, v_nfit, rpar=None, Vbias_offset=0.0,
                     i_offset=0.0, show_plot=False, ignore_nans=True):
        '''Fit supeconducting and normal branches and use information to convert raw data to IV.'''

        vb_step = Vbias[1]-Vbias[0]
        if vb_step < 0:   # reverse order if data was taken from high V -> 0
            Vbias = Vbias[::-1]
            Vfb = Vfb[::-1] 
        if show_plot: rawIVfig = pylab.figure()
        n_output = self.ivFitNormal(Vbias, Vfb, Vbias_offset, v_nfit, show_plot=show_plot) 
        n_fit = n_output[0]; norm_inds = n_output[1]   
        pylab.show()
        pdb.set_trace()
        # m_norm = n_fit[0]
        # diff_norm = m_norm*np.diff(Vbias)[0]

        # # pdb.set_trace()
        # end_sc = None
        # fb_diff = np.diff(Vfb)
        # # slope = np.diff(Vfb)/np.diff(Vbias)
        # # diff_slope = np.diff(slope)
        # diff_sc = np.nanmean(fb_diff[0:5])   # assumes the first 5 data points are in the SC branch
        # # discs = np.where(np.abs(slope)/np.abs(m_norm) > 1.5)[0]
        # # pdb.set_trace()
        # discs = np.where(np.abs(fb_diff/diff_sc) > 1.2)[0]
    
        # discs = discs[np.insert(np.diff(discs) != 1, 0, True)]   # remove sequential data points
        # end_sc = discs[0]
        # noslope = np.where(np.abs(fb_diff[:discs[-1]]/diff_norm)<=0.1)[0]
        # finds = []
        # # print(Vbias[discs])
        # # pdb.set_trace()
        # if len(noslope)!=0:
        #     # finds = np.concatenate([finds, noslope[~np.isin(noslope, finds)]]).astype(int)
        #     # finds = np.concatenate([discs[1]+1, noslope]).astype(int)
        #     finds = np.arange(min(noslope), discs[-1]+1)
        # if len(discs)>1:   # ignore flagged data points
        #     # discs = np.where((np.abs(fb_diff)/np.abs(diff_sc) > 1.5) & (np.abs(np.diff(Vfb))<np.max(np.abs(np.diff(Vfb)))))[0]
        #     finds_temp = np.arange(discs[0]+1, discs[1]+1)
        #     finds = np.concatenate([finds, finds_temp[~np.isin(finds_temp, finds)]]).astype(int)
        # if show_plot:
        #     pylab.plot(Vbias[:-1], np.abs(np.diff(Vfb)/diff_norm)*1E-3, alpha=0.5)
        #     pylab.plot(Vbias[:-1], np.abs(np.diff(Vfb)/diff_sc)/10, alpha=0.5)
        #     pylab.plot(Vbias, Vfb, '.')
        #     pylab.plot(Vbias[finds], Vfb[finds], 'x')
        #     # pylab.show()
        # # pdb.set_trace()
        # Vfb[finds] = np.nan        # pylab.plot(Vbias[end_sc], Vfb[end_sc], 'o')
        # # pylab.show()
        # pdb.set_trace()

        if n_fit[0] < 0:   # Flip y axis if slope of normal branch is negative
            # Vfb = -Vfb   
            Vfb = -Vfb + max(Vfb)   # flip and move to positive quadrant
            if show_plot: rawIVfig.clear(True)
            n_output = self.ivFitNormal(Vbias, Vfb, Vbias_offset, v_nfit, show_plot=show_plot)  
            n_fit = n_output[0]; norm_inds = n_output[1] 
            
        normal_offset = n_fit[1] 

        # sc_fit, end_sc = self.ivFitSuperconduct(Vbias, Vfb, Vbias_offset, show_plot=False, end_sc=end_sc)  
        sc_fit, end_sc = self.ivFitSuperconduct(Vbias, Vfb, Vbias_offset, show_plot=show_plot)  
        m_sc = sc_fit[0]; sc_offset = sc_fit[1]
        
        # pylab.show()
        # pdb.set_trace()
        # exclude flagged data points where Vfb is set to -1
        # fb_diff = np.diff(Vfb)
        # diff_sc = np.max(fb_diff[:end_sc-3])
        
        # etinds = np.arange(end_sc, end_sc+10)
        # finds = np.where(fb_diff>diff_sc)
        # finds = np.where(((Vfb[:-1]>0.95) & (Vfb[:-1]<1.1) & (Vbias[:-1]>Vbias[end_sc-1])) | (fb_diff>diff_sc))[0]   # remove flagged data points
        # finds = np.where(((Vfb[:-1]>0.95) & (Vfb[:-1]<1.5) & (Vbias[:-1]>Vbias[end_sc-1])) | (fb_diff>diff_sc))[0]   # remove flagged data points
        # if show_plot: 
        #     pylab.plot(Vbias[finds], Vfb[finds], 'x', label='flagged data points')
        #     pylab.plot(Vbias[:-1], fb_diff/max(fb_diff)*max(Vbias))
        # Vfb[finds] = np.nan; Vbias[finds] = np.nan   # ignore flagged data points 

        # if show_plot: pylab.plot(Vbias[finds], Vfb[finds], 'x')

        # if len(finds)>0:
        #     # print(end_sc)
        #     end_sc = min([end_sc, min(finds)])
        #     # print(end_sc)
        #     Vfb[:end_sc] = Vfb[:end_sc] - sc_offset + normal_offset   # adjust for SC branch offset
        # else:
        #     Vfb[:end_sc-1] = Vfb[:end_sc-1] - sc_offset + normal_offset   # adjust for SC branch offset
        # pylab.show()
        # pdb.set_trace()


        Vfb[:end_sc] = Vfb[:end_sc] - sc_offset + normal_offset   # adjust for SC branch offset, include last superconducting point
        
        if show_plot: pylab.plot(Vbias, Vfb, '.', label='after offset adjustments', alpha=0.5); pylab.legend()
        # pylab.show()
        

        if rpar is None:   # calculate Rparasitic
            rpar = rsh*(rfb*mr/(rbias*m_sc)-1.0)
        Vtes, Ites, Rtes, Ptes, i_meas = self.ivConvertTDM(Vbias, Vfb, rfb, rbias, rsh, mr, rpar, normal_offset, Vbias_offset, i_offset)
        ninds = np.where(Ites<0)   # where is current still negative?
        Vtes[ninds] = np.nan; Ites[ninds] = np.nan; Rtes[ninds] = np.nan; Ptes[ninds] = np.nan; i_meas[ninds] = np.nan


        return Vtes, Ites, Rtes, Ptes, i_meas, n_fit, norm_inds, sc_fit, end_sc, rpar  

    def ivConvertTDM(self, Vbias, Vfb, rfb, rbias, rsh, mr, rpar, normal_offset, Vbias_offset=0.0, i_offset=0.0):
        '''Convert IV data.'''
        vb_step = Vbias[1]-Vbias[0]
        if vb_step < 0:
            Vbias = Vbias[::-1]
            Vfb = Vfb[::-1] 
        i_meas = (Vfb-normal_offset)/(rfb*mr) + i_offset   # current measured by the SQ?
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
            pylab.plot(x,y,'.', label='Data')
            pylab.plot(xf,yfa, label='Normal Branch Fit')
        
        return n_fit, norm_inds   
    
    def ivFitSuperconduct(self, Vbias, Vfb, Vbias_offset, show_plot=False, end_sc=None):
        '''Fit the superconducting branch of the IV and return the slope and intercept. '''

        if end_sc==None:   # find end of SC branch if not already given
            fb_diff = np.diff(Vfb)
            neg_slope_array = np.where(fb_diff<0)
            # where is slope negative 10x in a row?
            end_sc =  min(neg_slope_array[0])  
        Vbias_sc = Vbias[:end_sc+1] - Vbias_offset
        Vfb_sc = Vfb[:end_sc+1]

        finds = np.isfinite(Vfb_sc)   # ignore nans
        sc_m, sc_b = scipy.polyfit(Vbias_sc[finds], Vfb_sc[finds], 1)
        Vfb_sc_fit = scipy.polyval([sc_m,sc_b],Vbias_sc)
        sc_fit = [sc_m, sc_b]
        if show_plot is True:
            pylab.plot(Vbias_sc,Vfb_sc_fit, label='SC Branch Fit')
            pylab.legend()
        
        return sc_fit, end_sc   
        

    def ivInterpolate(self, Vtes, Ites, Rtes, percentRns, rn, tran_pRn_start=0.050, plot=False):
      
        btran_indices = np.where((Rtes/rn)<tran_pRn_start)
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
            pylab.figure()
            pylab.plot(v_pnts*1e6, i_pnts*1e3, 'o', alpha=0.8)
            pylab.xlabel('V [mV]'); pylab.ylabel('I [uA]')
        
        return v_pnts, i_pnts, p_pnts

    def ivPRnLinearFit(self, v_pnt_array, i_pnt_array):
        
        pylab.figure()
        for index in range(len(v_pnt_array[:,0])):
            pylab.plot(v_pnt_array[index]*1e6, i_pnt_array[index]*1e3,'o')
            rvt_m, rvt_b = scipy.polyfit(v_pnt_array[index], i_pnt_array[index], 1)
            Ites_rvt_fit = scipy.polyval([rvt_m,rvt_b],v_pnt_array[index])
            rvt_residuals = i_pnt_array[index]-Ites_rvt_fit
            pylab.plot(v_pnt_array[index]*1e6, Ites_rvt_fit*1e3)
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
    

    def sigma_power(self, i, sigma_I, v, sigma_V):   # calculate error in power measurement from error in voltage and current measurement
        return np.sqrt(np.multiply(i**2,sigma_V**2) + np.multiply(v**2,sigma_I**2))   # sigma_power^2 = (I*sigma_V)^2 + (V*sigma_I)^2

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


    def load_data(self, dfiles):

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

    def convert_rawdata(self, data, tesids, constT=True, v_nfit=0.3, save_figs=False, show_aplots=False, tran_pRn_start=0.2, pRn=np.array([25, 30, 40, 50, 60, 70, 80]), 
                        fn_comments='', bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/'):
        ## sort raw data and convert to real units
        # v_nfit [V] = v_bias above which TES is normal (approximate)
        # tran_pRn_start = Fraction of Rn dubbed beginning of SC transition
        # Tb_ind = index in Tbaths of Tbath to quote, 2 = 168 mK?

        ivs = {}   # master dictionary

        for tesid in tesids:   # iterate through bolos

            bay = tesid.split('_')[0]; row = tesid.split('_')[1]
            if bay=='BayAY':
                pad = str(int(tesid.split('Row')[1])+1)  # pad number
            elif bay=='BayAX':
                pad = str(int(tesid.split('Row')[1])+16)  # pad number
            
            tes = TESAnalyze() 
            boloid = pad_bolo_map[pad]   # map TES ID to pad # to bolo ID
            ivs[tesid] = {}  
            ivs[tesid]['Pad'] = pad; ivs[tesid]['Bolometer'] = boloid; ivs[tesid]['Bay'] = bay  # save IDs to main dictionary
            
            iv_labels = [key for key in data[bay][row]['iv']]   # label for IV taken at some base temp
            if tesid == 'BayAY_Row12' or tesid == 'BayAX_Row10' or tesid == 'BayAX_Row03' or tesid == 'BayAX_Row00' or tesid == 'BayAX_Row06': 
                iv_labels.remove('iv014')   # wonky IV

            maxiv = max([len(data[bay][row]['iv'][ivlab]['data'][0]) for ivlab in iv_labels])   # handle IVs of different lengths
            asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
            
            # initialize arrays
            Vbias = np.full(asize, np.nan); Vfb = np.full(asize, np.nan)   
            Vtes = np.full(asize, np.nan); Ites = np.full(asize, np.nan); Rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); Ptes = np.full(asize, np.nan)
            Tbaths = np.full((len(iv_labels), 1), np.nan); Rns = np.full((len(Tbaths), 1), np.nan)
            v_pnts = np.zeros((len(iv_labels), len(pRn))); i_pnts = np.zeros((len(iv_labels), len(pRn))); p_pnts = np.zeros((len(iv_labels), len(pRn)))   # interpolated IV points at each % Rn
            sigma_V = np.full((len(iv_labels), 1), np.nan); sigma_I = np.full((len(iv_labels), 1), np.nan); nfits_real = np.zeros((len(iv_labels), 2))

            for ii, ivlab in enumerate(iv_labels):   # iterate through IVs at different base temperatures

                # sort raw data
                Tbaths[ii] = data[bay][row]['iv'][ivlab]['measured_temperature']
                ivlen = len(data[bay][row]['iv'][ivlab]['data'][0])   # handle IVs of different lengths
                Vbias[ii,:ivlen] = data[bay][row]['iv'][ivlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
                Vfb[ii,:ivlen] = data[bay][row]['iv'][ivlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
                
                # convert to real units
                Vtes[ii], Ites[ii], Rtes[ii], Ptes[ii], i_meas[ii], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(Vbias[ii], Vfb[ii], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)
                nfits_real[ii] = np.polyfit(Vtes[ii,norm_inds][0], Ites[ii,norm_inds][0], 1)   # normal branch line fit in real units
                Ifit_norm = Vtes[ii,norm_inds]*nfits_real[ii, 0]+nfits_real[ii, 1]   # normal branch line fit in real units
                
                Rns[ii] = np.mean(Rtes[ii, norm_inds])   # ohms, should be consistent with 1/nfit_real[0]
                sigma_V[ii] = np.std(Vtes[ii,:end_sc])   # V, error in voltage measurement = std of SC branch
                sigma_I[ii] = np.std(Ites[ii,norm_inds] - Ifit_norm)  # A, error in current measurement = std in normal branch after line subtraction
                
                # interpolate IV points at various % Rn at this Tbath
                v_pnts[ii], i_pnts[ii], p_pnts[ii] = tes.ivInterpolate(Vtes[ii], Ites[ii], Rtes[ii], pRn, Rns[ii], tran_pRn_start=tran_pRn_start, plot=False)    

            if show_aplots: 
                pylab.figure()
                ivsort = np.argsort(Tbaths)   # plot IVs with increasing Tbath
                for ii in ivsort:
                    pylab.plot(v_pnts[ii]*1e6, i_pnts[ii]*1e3, 'o', alpha=0.7, color=pylab.cm.plasma((Tbaths[ii]-min(TbsToReturn)*1)/(max(Tbaths))))   # Tbaths/max(Tbaths)
                    pylab.plot(Vtes[ii]*1e6, Ites[ii]*1e3, alpha=0.6, label='{} mK'.format(round(Tbaths[ii]*1E3,2)), color=pylab.cm.plasma((Tbaths[ii]-min(TbsToReturn)*1)/(max(Tbaths)*0.8)))
                pylab.xlabel('Voltage [$\mu$V]')
                pylab.ylabel('Current [mA]')
                pylab.title('Interpolated IV Points')
                pylab.legend()
                if save_figs: pylab.savefig(bolotest_dir + 'Plots/IVs/' + tesid + '_interpIVs' + fn_comments + '.png', dpi=300)

            Rn = np.nanmean(Rns); Rn_err = np.nanstd(Rns)
            sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
            for ii in np.arange(len(i_pnts)):
                sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_I[ii], v_pnts[ii], sigma_V[ii])

        # store results in dict
        sort_inds = np.argsort(Tbaths)   # sort by temp, ignore nans
        ivs[tesid]['TES ID'] = tesid  
        ivs[tesid]['meas_temps'] = Tbaths[sort_inds]   # K
        ivs[tesid]['constT'] = constT
        ivs[tesid]['vbias'] = Vbias[sort_inds]  
        ivs[tesid]['vfb'] = Vfb[sort_inds]  
        ivs[tesid]['vtes'] = Vtes[sort_inds]   # volts
        ivs[tesid]['ites'] = Ites[sort_inds]   # amps
        ivs[tesid]['rtes'] = Rtes[sort_inds]   # ohms
        ivs[tesid]['ptes'] = Ptes[sort_inds]   # power
        ivs[tesid]['ptes_fit'] = p_pnts   # TES power at % Rns
        ivs[tesid]['ptes_err'] = sigma_p   # error bar on TES power at % Rns
        ivs[tesid]['i_meas'] = i_meas[sort_inds]   # amps
        ivs[tesid]['rn_meas'] = Rn   # Ohms?
        ivs[tesid]['rn_err'] = Rn_err   # Ohms?

        return ivs
    
    def fit_powerlaws(self, ivs,  save_figs=False, fitGexplicit=True, pRn_toquote=80, Tinds_return=np.array([0, 3, 8, 11, 15, 2]), fn_comments='', Tb_ind=2, constT=True, 
                      pRn=np.array([25, 30, 40, 50, 60, 70, 80]), init_guess=np.array([1.E-10, 2.5, .170]), bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/'):
        ### wrapper to fit power laws for all TESs
        # const_T = quote constant T_TES fit values vs values at a particular T_TES

        pfig_path = bolotest_dir + 'Plots/Psat_fits/' + tesid + '_Pfit' + fn_comments + '.png' if save_figs else None
        tesids = np.array([key for key in ivs.keys()])

        Tbaths = ivs[tesid]['meas_temps']
        TbsToReturn = Tbaths[Tinds_return]   # Tbaths to show on P vs % Rn plots
        Tb_toquote = Tbaths[Tb_ind]
        qind = np.where(pRn==pRn_toquote)[0][0]   # save results from chosen %Rn fit
        if not constT: tind = np.where(TbsToReturn==Tb_toquote)[0][0]  # if T_TES is allowed to vary with Tb (i.e. not assumed to be Tc), which 
        
        for tesid in tesids:
            tes = TESAnalyze() 

            p_pnts = ivs[tesid]['ptes_fit']; sigma_p = ivs[tesid]['ptes_err']   # TES power at % Rns from interpolated IV points

            GTcs, Ks, ns, Ttes, GTcs_err, Ks_err, ns_err, Ttes_err = tes.fitPowerLaw(pRn, Tbaths, p_pnts.T, init_guess, fitToLast=True, 
                    suptitle=tesid, TbsToReturn=TbsToReturn, plot=True, sigma=sigma_p.T, nstd=5, pfigpath=pfig_path, constT=constT, fitGexplicit=fitGexplicit)   # pass error to fitter     
            if save_figs: pylab.savefig(bolotest_dir + 'Plots/fit_params/' + tesid + '_fitparams' + fn_comments + '.png', dpi=300) 
            
            if constT:
                GTc_toquote = GTcs[qind]; GTcerr_toquote = GTcs_err[qind]
                Tc_toquote = Ttes[qind]; Tcerr_toquote = Ttes_err[qind]
            else:
                GTc_toquote = GTcs[tind, qind]; GTcerr_toquote = GTcs_err[tind, qind]
                Tc_toquote = Ttes[tind, qind]; Tcerr_toquote = Ttes_err[tind, qind]

            print(' ')
            print(' ')
            print(tesid)
            print('G = ', round(GTc_toquote*1e12, 2), ' +/- ', round(GTcerr_toquote*1e12, 2), 'pW/K')
            print('K = ',  round(Ks[qind]*1E11, 3), ' +/- ',  round(Ks_err[qind]*1E11, 3), ' E-11') 
            print('n = ', round(ns[qind], 2), ' +/- ', round(ns_err[qind], 4))
            print('Tc = ', round(Tc_toquote*1e3, 2), ' +/- ',  round(Tcerr_toquote*1e3, 2), 'mK')
            print('TES Rn = ', round(Rn*1e3, 2), ' +/- ', round(Rn_err*1e3, 2), ' mOhms')

            ### calculate Psat
            # find transition + normal branch
            sc_inds = np.where((Rtes[Tb_ind]/Rn)<.2)[0]
            start_ind = np.max(sc_inds)
            end_ind = np.max(np.where(((Rtes[Tb_ind]/Rn)>.2) & (Rtes[Tb_ind]!=np.nan)))
            Vtes_tran = Vtes[Tb_ind, start_ind:end_ind]
            Ites_tran = Ites[Tb_ind, start_ind:end_ind]
            Rtes_tran = Rtes[Tb_ind, start_ind:end_ind]

            # calculate Psat
            Ptes_tran = Vtes_tran * Ites_tran
            sat_ind = np.where(Ites_tran == np.min(Ites_tran))[0][0]   # where the TES goes normal
            Psat = Ptes_tran[sat_ind]
            Psat_err = tes.sigma_power(Ites_tran[sat_ind], sigma_I[Tb_ind], Vtes_tran[sat_ind], sigma_V[Tb_ind])
            Psat_calc = tes.Psat_atT(Tb_toquote, Tc_toquote, Ks[qind], ns[qind])
            print('Psat = ', round(Psat*1e12, 4), ' +/- ', round(Psat_err*1e12, 4), 'pW')
            print('Psat (calc) = ', round(Psat_calc*1e12, 4), 'pW')
            print(' ')
            print(' ')
            if show_psatcalc:
                pylab.figure()
                pylab.plot(Vtes_tran.T*1e6, Ites_tran.T/np.max(Ites_tran), label='TES IV')
                pylab.plot(Vtes_tran.T*1e6, Ptes_tran.T/np.max(Ptes_tran), label='Power')
                pylab.plot(Vtes_tran[sat_ind]*1e6, Psat/np.max(Ptes_tran), 'x', label='$P_{sat}$')
                pylab.xlabel('Voltage [$\mu$V]')
                pylab.ylabel('Normalized Current')
                pylab.legend()
                pylab.title('TES IV and Calculated Power at Tbath = ' + str(round(Tb_toquote*1000, 1)) + 'mK')
                if save_figs: pylab.savefig(bolotest_dir + 'Plots/psat_calc/' + tesid + '_psatcalc' + fn_comments + '.png', dpi=300)


            ivs[tesid]['fitGexplicit'] = fitGexplicit
            ivs[tesid]['G@Tc [pW/K]'] = GTc_toquote*1e12 
            ivs[tesid]['G_err@Tc [pW/K]'] = GTcerr_toquote*1e12
            ivs[tesid]['G@170mK [pW/K]'] = tes.scale_G(.170, GTc_toquote, Tc_toquote, ns[qind])*1e12  
            ivs[tesid]['G_err@170mK [pW/K]'] = tes.sigma_GscaledT(.170, GTc_toquote, Tc_toquote, ns[qind], GTcerr_toquote, Tcerr_toquote, ns_err[qind])*1e12  
            ivs[tesid]['k'] = Ks[qind] 
            ivs[tesid]['k_err'] = Ks_err[qind] 
            ivs[tesid]['n'] = ns[qind] 
            ivs[tesid]['n_err'] = ns_err[qind] 
            ivs[tesid]['Tc [mK]'] = Tc_toquote*1e3
            ivs[tesid]['Tc_err [mK]'] = Tcerr_toquote*1e3
            ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat*1e12
            ivs[tesid]['Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat_err*1e12
            ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc'] =  Psat_calc*1e12

        return ivs