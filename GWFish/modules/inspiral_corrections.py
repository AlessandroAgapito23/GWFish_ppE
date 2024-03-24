#Python libraries

import os
import logging
import requests 
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
import scipy.optimize as optimize

#GWFish libraries

import GWFish as gw 
import GWFish.modules.constants as cst
import GWFish.modules.auxiliary as aux
import GWFish.modules.fft as fft
import GWFish.modules.waveforms as wf
from GWFish.modules.waveforms import Waveform

#class which inherits the Waveform class in waveforms.py
class Inspiral_corr(Waveform):

     def _set_default_gw_params(self):
        self.gw_params = {
            'mass_1': 0., 'mass_2': 0., 'luminosity_distance': 0., 
            'redshift': 0., 'theta_jn': 0., 'phase': 0., 'geocent_time': 0., 
            'a_1': 0., 'a_2': 0.,'cut': 4.,
            #ppE parameters
            'beta':0., 'PN':0.,
            #gIMR
            'delta_phi_0':0.,
            'delta_phi_1':0.,
            'delta_phi_2':0.,
            'delta_phi_3':0.,
            'delta_phi_4':0.,
            'delta_phi_5':0.,
            'delta_phi_6':0.,
            'delta_phi_7':0.,
            'delta_phi_8':0.,
            'delta_phi_9':0.,
            #quadrupole deviations
            'k_1':0., 'k_2':0.,
            #octupole deviations
            'lambda_1':0., 'lambda_2':0.
        }

     def update_gw_params(self, new_gw_params):
        self.gw_params.update(new_gw_params)
        self._frequency_domain_strain = None
        self._time_domain_strain = None
          
     def get_phase_corr(self):

        # We have to add delta_phi_i as in gIMRPhenomD (arXiv:1603.08955)
        # phi ---> phi*(1+delta_phi_i)
        # phi is a combination of phi_i, i=0,....,7 and i=2PN
        # We want to modify phi for each b one by one and b = i-5 
        # beta is a function of delta_phi_i, phi_i and eta

        #PPE phase deviations
        PN = self.gw_params['PN']
        beta = self.gw_params['beta']
        
        #gIMR phase deviations
        delta_phi_0 = self.gw_params['delta_phi_0']
        delta_phi_1 = self.gw_params['delta_phi_1']
        delta_phi_2 = self.gw_params['delta_phi_2']
        delta_phi_3 = self.gw_params['delta_phi_3']
        delta_phi_4 = self.gw_params['delta_phi_4']
        delta_phi_5 = self.gw_params['delta_phi_5']
        delta_phi_6 = self.gw_params['delta_phi_6']
        delta_phi_7 = self.gw_params['delta_phi_7']
        delta_phi_8 = self.gw_params['delta_phi_8']
        delta_phi_9 = self.gw_params['delta_phi_9']
     
        return PN, beta, delta_phi_0, delta_phi_1, delta_phi_2, delta_phi_3, delta_phi_4,\
        delta_phi_5, delta_phi_6, delta_phi_7, delta_phi_8, delta_phi_9

     def get_mult_corr(self):
          
        #quadrupole deviations
        k_1 = self.gw_params['k_1']
        k_2 = self.gw_params['k_2']
        lambda_1 = self.gw_params['lambda_1']
        lambda_2 = self.gw_params['lambda_2']
          
        return k_1, k_2, lambda_1, lambda_2
     
         
################################################################################
################################ TAYLORF2_PPE ##################################
########################## with spin corrections ###############################

class TaylorF2_PPE(Inspiral_corr):

    """ GWFish implementation of TaylorF2_PPE """
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._maxn = None
        self.psi = None
        if self.name != 'TaylorF2_PPE':
            logging.warning('Different waveform name passed to TaylorF2_PPE: '+ self.name)

    @property
    def maxn(self):
        if self._maxn is None:
            if 'maxn' in self.data_params:
                self._maxn = self.data_params['maxn']
            else:
                self._maxn = 8
            if type(self._maxn) is not int:
                return ValueError('maxn must be integer')
        return self._maxn
            
    
    def calculate_phase(self): 

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        f_isco = aux.fisco(self.gw_params)  #inner stable circular orbit 
        ones = np.ones((len(ff), 1)) 

        PN, beta, delta_phi_0, delta_phi_1, delta_phi_2, delta_phi_3, delta_phi_4,\
        delta_phi_5, delta_phi_6, delta_phi_7, delta_phi_8, delta_phi_9 = Inspiral_corr.get_phase_corr(self)
        
        #f_cut = cut_order * f_isco
        cut = self.gw_params['cut']

        ################################################################################ 
        ############################## PHASE CORRECTIONS ###############################
        ############################# PN expansion of phase ############################

        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)
        phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_5_l, phi_6, phi_6_l, phi_7 = wf.TaylorF2.EI_phase_coeff(self)

        psi_gIMR = 3./(128.*eta)*(delta_phi_0*(np.pi*ff)**(-5./3.) +\
                delta_phi_1*(np.pi*ff)**(-4./3.)+\
                phi_2*delta_phi_2*(np.pi*ff)**(-1.) +\
                phi_3*delta_phi_3*(np.pi*ff)**(-2./3.) +\
                phi_4*delta_phi_4*(np.pi*ff)**(-1./3.) +\
                phi_5*delta_phi_5 + phi_5_l*delta_phi_8*np.log(np.pi*ff) +\
                (phi_6*delta_phi_6 + phi_6_l*delta_phi_9*np.log(np.pi*ff))*((np.pi*ff)**(1./3.)) +\
                phi_7*delta_phi_7*(np.pi*ff)**(2./3.)) 
        
        psi_ppe = eta**((2*PN-5.)/5.)*beta*(np.pi*ff)**((2*PN-5.)/3.)  #ppe correction at every b order

        psi_EI = psi_TF2 + psi_ppe + psi_gIMR

        ################################################################################ 
        # Evaluate PHASE and DERIVATIVE at the INTERFACE between ins and int >>>>>>>>>>>
        ################################################################################ 

        f1 = 0.018

        psi_gIMR_f1 = 3./(128.*eta)*(delta_phi_0*(np.pi*f1)**(-5./3.) +\
                    delta_phi_1*(np.pi*f1)**(-4./3.)+\
                    phi_2*delta_phi_2*(np.pi*f1)**(-1.) +\
                    phi_3*delta_phi_3*(np.pi*f1)**(-2./3.) +\
                    phi_4*delta_phi_4*(np.pi*f1)**(-1./3.) +\
                    phi_5*delta_phi_5 + phi_5_l*delta_phi_8*np.log(np.pi*f1) +\
                    (phi_6*delta_phi_6 + phi_6_l*delta_phi_9*np.log(np.pi*f1))*((np.pi*f1)**(1./3.)) +\
                    phi_7*delta_phi_7*(np.pi*f1)**(2./3.))
                
        psi_ppe_f1 = eta**((2*PN-5.)/5.)*beta*(np.pi*f1)**((2*PN-5.)/3.)

        psi_EI_f1 = psi_TF2_f1 + psi_ppe_f1 + psi_gIMR_f1
        

        # Analytical derivative 
        psi_gIMR_prime = 3./(128.*eta)*((np.pi)**(-5./3.)*(-5./3.*ff**(-8./3.)) +\
                        delta_phi_1*(np.pi)**(-4./3.)*(-4./3.*ff**(-7./3.)) +\
                        phi_2*delta_phi_2*(np.pi)**(-1.)*(-1.*ff**(-2.)) +\
                        phi_3*delta_phi_3*(np.pi)**(-2./3.)*(-2./3.*ff**(-5./3.)) +\
                        phi_4*delta_phi_4*(np.pi)**(-1./3.)*(-1./3.*ff**(-4./3.)) +\
                        phi_5_l*delta_phi_8*ff**(-1.) +\
                        phi_6*delta_phi_6*(np.pi)**(1./3.)*(1./3.*ff**(-2./3.)) +\
                        phi_6_l*delta_phi_9*(((np.pi*ff)**(1./3.))*(ff**(-1.)) +\
                                             np.log(np.pi*ff)*(np.pi)**(1./3.)*(1./3.*ff**(-2./3.))) +\
                        phi_7*delta_phi_7*(np.pi)**(2./3.)*(2./3.*ff**(-1./3.)))

        psi_gIMR_prime_f1 = 3./(128.*eta)*((np.pi)**(-5./3.)*(-5./3.*f1**(-8./3.)) +\
                        delta_phi_1*(np.pi)**(-4./3.)*(-4./3.*f1**(-7./3.)) +\
                        phi_2*delta_phi_2*(np.pi)**(-1.)*(-1.*f1**(-2.)) +\
                        phi_3*delta_phi_3*(np.pi)**(-2./3.)*(-2./3.*f1**(-5./3.)) +\
                        phi_4*delta_phi_4*(np.pi)**(-1./3.)*(-1./3.*f1**(-4./3.)) +\
                        phi_5_l*delta_phi_8*1./3.*f1**(-1.) +\
                        phi_6*delta_phi_6*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.)) +\
                        phi_6_l*delta_phi_9*(((np.pi*f1)**(1./3.))*(f1**(-1.)) +\
                                             np.log(np.pi*f1)*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.))) +\
                        phi_7*delta_phi_7*(np.pi)**(2./3.)*(2./3.*f1**(-1./3.)))

        psi_ppe_prime = eta**((2*PN-5.)/5.)*beta*((2*PN-5.)/3.)*(np.pi*ff)**((2*PN-8.)/3.)
                                           
        psi_ppe_prime_f1 = eta**((2*PN-5.)/5.)*beta*((2*PN-5.)/3.)*(np.pi*f1)**((2*PN-8.)/3.)
        
        psi_EI_prime = psi_TF2_prime + psi_gIMR_prime + psi_ppe_prime
        psi_EI_prime_f1 = psi_TF2_prime_f1 + psi_gIMR_prime_f1 + psi_ppe_prime_f1

        return psi_EI, psi_EI_prime, psi_EI_f1, psi_EI_prime_f1

    def calculate_frequency_domain_strain(self):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        cut = self.gw_params['cut']
        f_isco = aux.fisco(self.gw_params)

        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_PPE.calculate_phase(self)
        hp, hc = wf.TaylorF2.calculate_amplitude(self)
         
        ############################### PHASE OUTPUT ###############################

        phase = np.exp(1.j * psi)

        ############################## STRAIN OUTPUT ###############################
        
        polarizations = np.hstack((hp * phase, hc * 1.j * phase))

        # Very crude high-f cut-off which can be an input parameter 'cut', default = 4*f_isco
        f_cut = cut*f_isco*cst.G*M/cst.c**3
 
        polarizations[np.where(ff[:,0] > f_cut), :] = 0.j

        self._frequency_domain_strain = polarizations

        ############################################################################
        
    ################################################################################
    ############################# Amplitude & phase plot ###########################
    ################################################################################
        
    def plot (self, output_folder='./'):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)
        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_PPE.calculate_phase(self)
        
        phase = psi
        delta_phase = psi - psi_TF2
        
        plt.figure()
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 0]), label=r'$h_+$')
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 1]), label=r'$h_\times$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Fourier amplitude [$Hz^{-1}$]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        #plt.axis(axis)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder + 'amp_tot_TF2_PPE.png')
        plt.close()


        plt.figure()
        plt.semilogx(ff, phase)
        plt.xlabel('Dimensionless frequency')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + 'phase_tot_TF2_PPE.png')
        plt.close()

        
        plt.figure()
        plt.semilogx(ff, delta_phase)
        plt.xlabel('Dimensionless frequency')
        plt.ylabel('Phase difference [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + 'delta_phase_tot_PPE.png')
        plt.close()


################################################################################
################################ TAYLORF2_mult #################################
########################## with multipolar corrections #########################

class TaylorF2_mult(Inspiral_corr):

    """ GWFish implementation of TaylorF2_k """
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._maxn = None
        self.psi = None
        if self.name != 'TaylorF2_mult':
            logging.warning('Different waveform name passed to TaylorF2_mult: '+ self.name)

    @property
    def maxn(self):
        if self._maxn is None:
            if 'maxn' in self.data_params:
                self._maxn = self.data_params['maxn']
            else:
                self._maxn = 8
            if type(self._maxn) is not int:
                return ValueError('maxn must be integer')
        return self._maxn

    def INS_mult_coeff(self):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        k_1, k_2, lambda_1, lambda_2 = Inspiral_corr.get_mult_corr(self)

        P4 = ((-50.)*((1. - 2.*eta) * k_1 + delta_mass * k_2))*(chi_s**2 + chi_a**2) +\
             ((-100.)*((1. - 2.*eta) * k_2 + delta_mass * k_1))*chi_s*chi_a
         
        P6 = ((75515./288. - 232415./504.*eta + 1255./9.*eta2)*chi_s**2 +\
              (75515./288. - 263245./252.*eta - 480.*eta2)*chi_a**2)+\
              ((26015./28. - 1495./6.*eta)*delta_mass * k_2 +\
              (26015./28. - 44255./21.*eta - 240.*eta2)* k_1)*(chi_s**2 + chi_a**2) +\
              ((75515./144. - 8225./18.*eta)*delta_mass +\
              (26015./14. - 1495./3.*eta)*delta_mass * k_1 +\
              (26015./14. - 88510./21.*eta - 480.*eta2)*k_2)*chi_s*chi_a
         
        P7 = (14585./24. - 475./6.*eta + 100./3.*eta2)*chi_s**3 +\
             (25145./24. -2820.*eta)*delta_mass*chi_a**3 +\
             (14585/.8 - 215./2*eta)*delta_mass*chi_s**2*chi_a +\
             (14585./8. - 7270.*eta + 80*eta2)*chi_s*chi_a**2 +\
             ((3110./3. - 10250./3.*eta + 40*eta2)*k_1 +\
              ((3110./3. - 4030./3.*eta)*k_2 - 440.*(1 - eta)*lambda_2 - 440.*(1 - 3*eta)*lambda_1)*delta_mass)*chi_s**3 +\
              ((3110./3. - 8470./3.*eta)*k_2 - 440.*(1 - 3*eta)*lambda_2 +\
             ((3110./3. - 750.*eta)*k_1 - 440.*(1 - eta)*lambda_1)*delta_mass)*chi_a**3 +\
             ((3110./3. - 28970./3.*eta + 80.*eta2)*k_2 - 1320.*(1 - eta)*lambda_2 +\
              ((3110./3. - 10310./3.*eta)*k_1 - 1320.*(1 - eta)*lambda_1)*delta_mass)*chi_s**2*chi_a +\
             ((3110./3. - 27190./3.*eta + 40.*eta2)*k_1 - 1320.*(1 - 3*eta)*lambda_1 +\
              ((3110./3. - 8530./3.*eta)*k_2 - 1320.*(1 - eta)*lambda_2)*delta_mass)*chi_s*chi_a**2

        return P4, P6, P7
            
    
    def calculate_phase(self): 

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        f_isco = aux.fisco(self.gw_params)  #inner stable circular orbit 
        ones = np.ones((len(ff), 1)) 
        
        #f_cut = cut_order * f_isco
        cut = self.gw_params['cut']

        ################################################################################ 
        ############################## PHASE CORRECTIONS ###############################
        ############################# multipolar deviations ############################
        # with quadratic spin corrections at 3PN and cubic spin corrections at 3.5PN

        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)

        phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_5_l, phi_6, phi_6_l, phi_7 = wf.TaylorF2.EI_phase_coeff(self)
        P4, P6, P7 = Inspiral_corr.INS_mult_coeff(self)

        psi_mult = 3./(128.*eta)*(P4*(np.pi*ff)**(-1./3.) +\
                                  P6*(np.pi*ff)**(1./3.) +\
                                  P7*(np.pi*ff)**(2./3.))

        psi_EI = psi_TF2 + psi_mult

        ################################################################################ 
        # Evaluate PHASE and DERIVATIVE at the INTERFACE between ins and int >>>>>>>>>>>
        ################################################################################ 

        f1 = 0.018

        psi_mult = 3./(128.*eta)*(P4*(np.pi*f1)**(-1./3.) +\
                                  P6*(np.pi*f1)**(1./3.) +\
                                  P7*(np.pi*f1)**(2./3.))
                
        psi_EI_f1 = psi_TF2_f1 + psi_mult_f1        

        # Analytical derivative 
        psi_mult_prime = 3./(128.*eta)*(P4*(np.pi)**(-1./3.)*(-1./3.*ff**(-4./3.)) +\
                                        P6*(np.pi)**(1./3.)*(1./3.*ff**(-2./3.)) +\
                                        P7*((np.pi)**(2./3.)*(2./3.*ff**(-1./3.))))

        psi_mult_prime = 3./(128.*eta)*(P4*(np.pi)**(-1./3.)*(-1./3.*f1**(-4./3.)) +\
                                        P6*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.)) +\
                                        P7*((np.pi)**(2./3.)*(2./3.*f1**(-1./3.))))
         
         
        psi_EI_prime = psi_TF2_prime + psi_mult_prime
        psi_EI_prime_f1 = psi_TF2_prime_f1 + psi_k_mult_f1
         
        return psi_EI, psi_EI_prime, psi_EI_f1, psi_EI_prime_f1

    def calculate_frequency_domain_strain(self):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        r = self.gw_params['luminosity_distance'] * cst.Mpc
        iota = self.gw_params['theta_jn']
        cut = self.gw_params['cut']
        f_isco = aux.fisco(self.gw_params)

        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_mult.calculate_phase(self)

        A0 = 1./(np.pi**(2./3.))*(5./24.)**(0.5)*cst.c/r*Mc**(5./6.)*(ff*cst.c**3/(cst.G*M))**(-7./6.)
         
        a_0, a_1, a_2, a_2, a_3, a_4, a_5, a_6 = wf.IMRPhenomD.INS_amp_coeff(self)
        amp_PN = a_0 +\
                 a_2*(np.pi*ff)**(2./3.) +\
                 a_3*(np.pi*ff) +\
                 a_4*(np.pi*ff)**(4./3.) +\
                 a_5*(np.pi*ff)**(5./3.) +\
                 a_6*(np.pi*ff)**2.
         
        amp_tot = amp_PN*A0
         
        hp = amp_tot*0.5*(1 + np.cos(iota)**2.)
        hc = amp_tot*np.cos(iota)
         
        ############################### PHASE OUTPUT ###############################

        phase = np.exp(1.j * psi)

        ############################## STRAIN OUTPUT ###############################
        
        polarizations = np.hstack((hp * phase, hc * 1.j * phase))

        # Very crude high-f cut-off which can be an input parameter 'cut', default = 4*f_isco
        f_cut = cut*f_isco*cst.G*M/cst.c**3
 
        polarizations[np.where(ff[:,0] > f_cut), :] = 0.j

        self._frequency_domain_strain = polarizations

        ############################################################################

    ################################################################################
    ############################# Amplitude & phase plot ###########################
    ################################################################################
        
    def plot (self, output_folder='./'):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)
        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_mult.calculate_phase(self)
        
        phase = psi
        delta_phase = psi - psi_TF2
        
        plt.figure()
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 0]), label=r'$h_+$')
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 1]), label=r'$h_\times$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Fourier amplitude [$Hz^{-1}$]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        #plt.axis(axis)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder + 'amp_tot_TF2_mult.png')
        plt.close()


        plt.figure()
        plt.semilogx(ff, phase)
        plt.xlabel('Dimensionless frequency')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + 'phase_tot_TF2_mult.png')
        plt.close()

        
        plt.figure()
        plt.semilogx(ff, delta_phase)
        plt.xlabel('Dimensionless frequency')
        plt.ylabel('Phase difference [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + 'delta_phase_tot_mult.png')
        plt.close()


################################################################################
############################## IMRPhenomD_PPE ##################################
################################################################################

class IMRPhenomD_PPE(Inspiral_corr):
    
    """ GWFish implementation of IMRPhenomD_PPE """
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._maxn = None
        self.psi = None
        if self.name != 'IMRPhenomD_PPE':
            logging.warning('Different waveform name passed to IMRPhenomD_PPE: '+\
                             self.name)

    # Here we add the phase deviations, which satisfy the continuity conditions of the phase and its derivative at the inferface

    def calculate_phase(self): 

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)
        f_isco = aux.fisco(self.gw_params)  #inner stable circular orbit 
        ones = np.ones((len(ff), 1)) 

        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)
        phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_5_l, phi_6, phi_6_l, phi_7 = wf.TaylorF2.EI_phase_coeff(self) 
        PN, beta, delta_phi_0, delta_phi_1, delta_phi_2, delta_phi_3, delta_phi_4,\
        delta_phi_5, delta_phi_6, delta_phi_7, delta_phi_8, delta_phi_9 = Inspiral_corr.get_phase_corr(self)

        psi_gIMR = 3./(128.*eta)*(delta_phi_0*(np.pi*ff)**(-5./3.) +\
                delta_phi_1*(np.pi*ff)**(-4./3.)+\
                phi_2*delta_phi_2*(np.pi*ff)**(-1.) +\
                phi_3*delta_phi_3*(np.pi*ff)**(-2./3.) +\
                phi_4*delta_phi_4*(np.pi*ff)**(-1./3.) +\
                phi_5*delta_phi_5 + phi_5_l*delta_phi_8*np.log(np.pi*ff) +\
                (phi_6*delta_phi_6 + phi_6_l*delta_phi_9*np.log(np.pi*ff))*((np.pi*ff)**(1./3.)) +\
                phi_7*delta_phi_7*(np.pi*ff)**(2./3.)) 
        
        psi_ppe = eta**((2*PN-5.)/5.)*beta*(np.pi*ff)**((2*PN-5.)/3.)  #ppe correction at every b order

        psi_EI = psi_TF2 + psi_ppe + psi_gIMR

        sigma2, sigma3, sigma4 = wf.IMRPhenomD.LI_phase_coeff(self)

        psi_late_ins = + 1./eta*(3./4.*sigma2*ff**(4./3.) + 3./5.*sigma3*ff**(5./3.) + 1./2.*sigma4*ff**2)

        ################################################################################ 
        # Evaluate PHASE and DERIVATIVE at the INTERFACE between ins and int >>>>>>>>>>>
        ################################################################################ 

        f1 = 0.018
            
        psi_gIMR_f1 = 3./(128.*eta)*(delta_phi_0*(np.pi*f1)**(-5./3.) +\
                    delta_phi_1*(np.pi*f1)**(-4./3.)+\
                    phi_2*delta_phi_2*(np.pi*f1)**(-1.) +\
                    phi_3*delta_phi_3*(np.pi*f1)**(-2./3.) +\
                    phi_4*delta_phi_4*(np.pi*f1)**(-1./3.) +\
                    phi_5*delta_phi_5 + phi_5_l*delta_phi_8*np.log(np.pi*f1) +\
                    (phi_6*delta_phi_6 + phi_6_l*delta_phi_9*np.log(np.pi*f1))*((np.pi*f1)**(1./3.)) +\
                    phi_7*delta_phi_7*(np.pi*f1)**(2./3.))
                
        psi_ppe_f1 = eta**((2*PN-5.)/5.)*beta*(np.pi*f1)**((2*PN-5.)/3.)
         
        psi_EI_f1 = psi_TF2_f1 + psi_ppe_f1 + psi_gIMR_f1

        psi_gIMR_prime_f1 = 3./(128.*eta)*((np.pi)**(-5./3.)*(-5./3.*f1**(-8./3.)) +\
                        delta_phi_1*(np.pi)**(-4./3.)*(-4./3.*f1**(-7./3.)) +\
                        phi_2*delta_phi_2*(np.pi)**(-1.)*(-1.*f1**(-2.)) +\
                        phi_3*delta_phi_3*(np.pi)**(-2./3.)*(-2./3.*f1**(-5./3.)) +\
                        phi_4*delta_phi_4*(np.pi)**(-1./3.)*(-1./3.*f1**(-4./3.)) +\
                        phi_5_l*delta_phi_8*f1**(-1.) +\
                        phi_6*delta_phi_6*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.)) +\
                        phi_6_l*delta_phi_9*(((np.pi*f1)**(1./3.))*(f1**(-1.)) +\
                                             np.log(np.pi*f1)*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.))) +\
                        phi_7*delta_phi_7*(np.pi)**(2./3.)*(2./3.*f1**(-1./3.)))

        psi_ppe_prime_f1 = eta**((2*PN-5.)/5.)*beta*((2*PN-5.)/3.)*(np.pi*f1)**((2*PN-8.)/3.)

        psi_EI_prime_f1 = psi_TF2_prime_f1 + psi_ppe_prime_f1 + psi_gIMR_prime_f1
         
        psi_late_ins_f1 = 1./eta*(3./4.*sigma2*f1**(4./3.) + 3./5.*sigma3*f1**(5./3.) + 1./2.*sigma4*f1**2)
        psi_late_ins_prime = 1./eta*(sigma2*ff**(1./3.) + sigma3*ff**(2./3.) + sigma4*ff)
        psi_late_ins_prime_f1 = 1./eta*(sigma2*f1**(1./3.) + sigma3*f1**(2./3.) + sigma4*f1)

        #sigma1 = eta*psi_EI_prime_f1 - psi_late_ins_prime_f1
        #sigma0 = eta*psi_EI_f1 - psi_late_ins_f1

        psi_late_ins = 1./eta*(3./4.*sigma2*ff**(4./3.) + 3./5.*sigma3*ff**(5./3.) + 1./2.*sigma4*ff**2)
        psi_late_ins_f1 = 1./eta*(3./4.*sigma2*f1**(4./3.) + 3./5.*sigma3*f1**(5./3.) + 1./2.*sigma4*f1**2)
        psi_late_ins_prime_f1 = 1./eta*(sigma2*f1**(1./3.) + sigma3*f1**(2./3.) + sigma4*f1)
        
        #Total INSPIRAL PART OF THE PHASE (and its DERIVATIVE), with also late inspiral terms
        ################################################################################ 
        
        psi_ins = psi_EI + psi_late_ins
        psi_ins_f1 = psi_EI_f1 + psi_late_ins_f1
        psi_ins_prime_f1 = psi_EI_prime_f1 + psi_late_ins_prime_f1
        
        ####################### INS-INT PHASE CONTINUITY CONDITIONS ###################
        # Impose C1 conditions at the interface (same conditions as in IMRPhenomD but with different psi_ins & psi_ins_prime)

        beta2, beta3 = wf.IMRPhenomD.INT_phase_coeff(self)

        beta1 = eta*psi_ins_prime_f1 - beta2*f1**(-1.) - beta3*f1**(-4.)  # psi_ins_prime_f1 = psi_int_prime_f1
        beta0 = eta*psi_ins_f1 - beta1*f1 - beta2*np.log(f1) + beta3/3.*f1**(-3.) #psi_ins_f1 = psi_int_f1
      
        # Evaluate full psi intermediate and its analytical derivative
        psi_int = 1./eta*(beta0 + beta1*ff + beta2*np.log(ff) - 1./3.*beta3*ff**(-3.))
        psi_int_prime = 1./eta*(beta1 + beta2*ff**(-1.) + beta3*ff**(-4.))

        # Frequency at the interface between intermediate and merger-ringdown phases
        ff_RD, ff_damp = wf.IMRPhenomD.RD_damping(self)
        f2 = 0.5*ff_RD

        psi_int_f2 = 1./eta*(beta0 + beta1*f2 + beta2*np.log(f2) - 1./3.*beta3*f2**(-3.))
        psi_int_prime_f2 = 1./eta*(beta1 + beta2*f2**(-1.) + beta3*f2**(-4.))
        
        ####################### INT-MERG PHASE CONTINUITY CONDITIONS ###################

        alpha2, alpha3, alpha4, alpha5 = wf.IMRPhenomD.MR_phase_coeff(self)
        
        alpha1 = eta*psi_int_prime_f2 - alpha2*f2**(-2.) - alpha3*f2**(-1./4.) -\
                (alpha4*ff_damp)/(ff_damp**2. + (f2 - alpha5*ff_RD)**2.) # psi_int_prime_f2 = psi_MR_prime_f2
        alpha0 = eta*psi_int_f2 - alpha1*f2 + alpha2*f2**(-1.) -\
                4./3.*alpha3*f2**(3./4.) - alpha4*np.arctan((f2 - alpha5*ff_RD)/ff_damp) #psi_int_f2 = psi_MR_f2

        # Evaluate full merger-ringdown phase and its analytical derivative
        psi_MR = 1./eta*(alpha0 + alpha1*ff - alpha2*ff**(-1.) + 4./3.*alpha3*ff**(3./4.) +\
                         alpha4*np.arctan((ff - alpha5*ff_RD)/ff_damp))
        psi_MR_prime = 1./eta*(alpha1 + alpha2*ff**(-2.) + alpha3*ff**(-1./4.) + alpha4*ff_damp/(ff_damp**2. +\
                          (ff - alpha5*ff_RD)**2.))

        # Conjunction functions
        ff1 = 0.018*ones
        ff2 = 0.5*ff_RD*ones

        theta_minus1 = 0.5*(1*ones - wf.step_function(ff,ff1))
        theta_minus2 = 0.5*(1*ones - wf.step_function(ff,ff2))
    
        theta_plus1 = 0.5*(1*ones + wf.step_function(ff,ff1))
        theta_plus2 = 0.5*(1*ones + wf.step_function(ff,ff2))

        ########################### PHASE COMPONENTS ############################
        ###################### written continuosly in frequency #################

        psi_ins = psi_ins*theta_minus1
        psi_int = theta_plus1*psi_int*theta_minus2
        psi_MR = psi_MR*theta_plus2

        psi_tot = psi_ins + psi_int + psi_MR

        return psi_tot

        
    def calculate_frequency_domain_strain(self): 

        psi = IMRPhenomD_PPE.calculate_phase(self)
        hp, hc = wf.IMRPhenomD.calculate_amplitude(self)
         
        ########################### PHASE OUTPUT ###############################
         
        phase = np.exp(1.j * psi)
 
        ########################################################################

        polarizations = np.hstack((hp * phase, hc * 1.j * phase))

        ############################### OUTPUT #################################

        self._frequency_domain_strain = polarizations

        ########################################################################
       
#GWFISH

