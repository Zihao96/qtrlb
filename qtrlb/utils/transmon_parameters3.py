# -*- coding: utf-8 -*-
"""
Created on Nov 11 2022
Updated on Mar 16 2023

@author: Z, MB
"""

import numpy as np
import scqubits as scq
from scipy.optimize import minimize
from scipy.constants import pi, physical_constants

e = physical_constants['elementary charge'][0] # [Coulomb]
GHz = 1. # Use GHz for all frequency, anharmonicity and energy, unless specified.
# Anharmonicity should be negative and called "alpha" below.


def EJEC_to_falpha(EJ, EC, ng=0.0, ncut=50, truncated_dim=50):
    tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut, truncated_dim=truncated_dim)
    energies = tmon.eigenvals(evals_count=3)
    f01 = energies[1] - energies[0]
    f12 = energies[2] - energies[1]
    alpha = f12 - f01
    return f01, alpha


def falpha_to_EJEC(f01, alpha, method='BFGS', options={'gtol': 1e-07}):
    """
    This function gives you transmon EJ&EC based on desired f01 and anharmonicity:
    1) First it 'guesses' EJ and EC from input f01 and alpha with approximated formula's -> EJ_guess & EC_guess
    2) These guesses are start of numerical optimization with the exact transmon solution -> EJ & EC
    """
    
# =============================================================================
#   I think minimize might be more suitable for this problem than least_square fit.
#   All such optimization from scipy.optimize take functions that should have only one argument.
#   If there is more arguments like EJ, EC, we need to pack it into this way below.
#   Reference about minimization method:
#   https://scipy-lectures.org/advanced/mathematical_optimization/index.html#practical-guide-to-optimization-with-scipy
#   https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs
#   I choose BFGS rather than L-BFGS-B is because we only have two variables. It's not heavy on memory requirement.
#   I've tested all transmon frequency 2-6GHz with 10MHz step and all anharmonicity -300MHz to -100MHz with 1MHz step.
#   The discrepancy is below 1kHz for all frequency and anharmonicity. -Z
# =============================================================================
    
    EC_guess = -alpha if alpha < 0 else alpha
    EJ_guess = (f01 + EC_guess)**2 / (8*EC_guess)

    result = minimize(lambda params: ( (EJEC_to_falpha(EJ=params[0], EC=params[1])[0] - f01)**2 
                                      + 10 * (EJEC_to_falpha(EJ=params[0], EC=params[1])[1] - alpha)**2 ),
                      x0=(EJ_guess, EC_guess),
                      method=method,
                      options=options
                      )
    EJ, EC = result.x
    if result.fun > 1e-6:
        print('The estimated frequency of such EJEC may exceed 1MHz !!!')
    return EJ, EC
                     
                      
def round_EJ_jjlength(EJ, jj_length_step=0.005, jj_width=0.2, Jc=1):
    """
    This function calculates the closest EJ we can get in fab, based on MIT LL limitations.
        (So we will get discrete critical current and junction dimensions.)
    The main frequency discrepancy is from this function rather than from falpha_to_EJEC.
    
    EJ: [GHz]
    jj_length_step: [um], smallest allowed length dimension.
    jj_width: [um]
    Jc: [uA/um^2], critical current density.
    """
    Ic = EJ * 4*pi*e * 1e15 # [uA]. The beauty of set h=1. -Z
    jj_length = Ic / (Jc * jj_width)
    jj_length_rounded = jj_length_step * round(jj_length/jj_length_step)
    EJ_rounded = (jj_length_rounded * jj_width * Jc) / (4*pi*e * 1e15) # [GHZ]
    if jj_length_rounded < 0.15 or jj_length_rounded > 3:
        print('Junction length is smaller than 150nm or larger than 3um !!!')
    return EJ_rounded, jj_length_rounded


def get_parameters_dict(f01, alpha, fr, chi, jj_length_step=0.005, jj_width=0.2, Jc=1):
    """
    Generate useful transmon parameters dictionary based on the desired frequency and actual limitation.
    
    All frequencies/energies: [GHz]
    jj_length_step: [um], smallest allowed length dimension.
    jj_width: [um]
    Jc: [uA/um^2], critical current density.
    """
    EJ, EC = falpha_to_EJEC(f01, alpha)
    EJ, jj_length = round_EJ_jjlength(EJ, jj_length_step=jj_length_step, jj_width=jj_width, Jc=Jc)
    
    tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.5, ncut=50, truncated_dim=50)
    energies = tmon.eigenvals(evals_count=3)
    f01 = energies[1] - energies[0]
    f12 = energies[2] - energies[1]
    alpha = f12 - f01
    
    specdata = tmon.get_dispersion_vs_paramvals('ng', 'EC', [EC], transitions=[(i,i+1) for i in range(5)])
    charge_disp = specdata.dispersion.flatten()
    
    Ic_nA = EJ * 4*pi*e * 1e18 # [nA]
    Ctot_fF = e**2 / (2*EC*physical_constants['Planck constant'][0]) * 1e6 # [fF]
    delta = f01 - fr
    g = np.sqrt(chi * delta * (delta+alpha) / alpha)
    Cg_fF = Ctot_fF * g / fr * (2*EC/EJ)**0.25 * (25.8e3/50/pi)**0.5
    
    parameters_dict = {'f01_GHz': f01,
                       'alpha_MHz': alpha*1e3,
                       'EC_GHz': EC,
                       'EJ_GHz': EJ,
                       'EJ/EC': EJ/EC,
                       'charge_disp01_MHz':charge_disp[0]*1e3,
                       'charge_disp12_MHz':charge_disp[1]*1e3,
                       'charge_disp23_MHz':charge_disp[2]*1e3,
                       'charge_disp34_MHz':charge_disp[3]*1e3,
                       'charge_disp45_MHz':charge_disp[4]*1e3,
                       'Ic_nA': Ic_nA,
                       'Jc_uA/um^2': Jc,
                       'jj_width_nm': jj_width*1e3,
                       'jj_length_nm': jj_length*1e3,
                       'Ctot_fF': Ctot_fF,
                       'fr_GHz':fr,
                       'Cg_fF':Cg_fF,
                       'g_qr_MHz':g*1e3,
                       'chi_qr_kHz':chi*1e6
                        }
    return parameters_dict


def generate_parameters_table(qubits_dict, chi_zz=0):
    """
    Generate the parameters table for convenience. Need to make it better.
    """
    names = [' ']
    f01s = ['f01 (GHz)']
    alphas = ['alpha (MHz)']
    EJ_GHz = ['EJ (GHz)']
    EC_GHz = ['EC (GHz)']
    EJEC = ['EJ/EC']
    charge_disp01_MHz = ['charge_disp01_MHz']
    charge_disp12_MHz = ['charge_disp12_MHz']
    charge_disp23_MHz = ['charge_disp23_MHz']
    charge_disp34_MHz = ['charge_disp34_MHz']
    charge_disp45_MHz = ['charge_disp45_MHz']
    Ic = ['Ic (nA)']
    Jc = ['Jc (uA/um^2)']
    jj_width_nm = ['JJ width (nm)']
    jj_length_nm = ['JJ length (nm)']
    Ctot = ['Ctot (fF)']
    fr = ['fr (GHz)']
    Cg = ['Cg (fF)']
    g_qr = ['g_qr (MHz)']
    chi_qr = ['chi_qr (kHz)']
    
    for qubit, params in qubits_dict.items():
        params_dict = get_parameters_dict(params['f01'], params['alpha'], params['fr'], params['chi'])
        names.append(f'Q{qubit}')
        f01s.append(round(params_dict['f01_GHz'],3))
        alphas.append(round(params_dict['alpha_MHz']))
        EJ_GHz.append(round(params_dict['EJ_GHz'],3))
        EC_GHz.append(round(params_dict['EC_GHz'],3))
        EJEC.append(round(params_dict['EJ/EC']))
        charge_disp01_MHz.append(round(params_dict['charge_disp01_MHz'],3))
        charge_disp12_MHz.append(round(params_dict['charge_disp12_MHz'],3))
        charge_disp23_MHz.append(round(params_dict['charge_disp23_MHz'],3))
        charge_disp34_MHz.append(round(params_dict['charge_disp34_MHz'],3))
        charge_disp45_MHz.append(round(params_dict['charge_disp45_MHz'],3))
        Ic.append(round(params_dict['Ic_nA'],2))
        Jc.append(params_dict['Jc_uA/um^2'])
        jj_width_nm.append(params_dict['jj_width_nm'])
        jj_length_nm.append(params_dict['jj_length_nm'])
        Ctot.append(round(params_dict['Ctot_fF'],2))
        fr.append(params_dict['fr_GHz'])
        Cg.append(round(params_dict['Cg_fF'],3))
        g_qr.append(round(params_dict['g_qr_MHz'],2))
        chi_qr.append(params_dict['chi_qr_kHz'])
    
    table = [names, f01s, alphas, EJ_GHz, EC_GHz, EJEC, charge_disp01_MHz, charge_disp12_MHz, 
             charge_disp23_MHz, charge_disp34_MHz, charge_disp45_MHz, Ic, Jc, 
             jj_width_nm, jj_length_nm, Ctot, fr, Cg, g_qr, chi_qr]
    return table


def get_bare_frequency(f01_m, alpha_m, fr_0, chi):
    """
    Calculate uncoupled bare frequency of qudit and resonator from measurement results.
    All quantities should keep same unit.
    The returned g01 is not the original g in tridiagonal JC Hamiltonian. Their relation is g01=g<0|n|1>.
    
    Arguments:
    f01_m : measured qubit base transition frequency.
    alpha_m : measured qubit anharmonicity. Usually negative.
    fr_0 : measured resonator frequency when qubit is in ground state |0>.
    chi : half of the measured dispersive shift. 2*chi = fr_1 - fr_0. Usually negative.
    """
    
    alpha_b = alpha_m + 2*chi
    lambshift = chi * (f01_m - fr_0 + alpha_b) / (2*chi + alpha_b)
    f01_b = f01_m - lambshift
    fr_b = fr_0 + lambshift
    g01 = np.sqrt(lambshift * (f01_m - fr_0 - 2*lambshift))
    return f01_b, alpha_b, fr_b, g01


def calculate_detuning_coupling_matrix(EJ, EC, ng, fr, g01, matrix_size=10, evals_count=10):
    """
    Calculate |Δ_ij|/g_ij in Table 1 of supplementary material in https://doi.org/10.1103/PhysRevLett.114.010501
    The EJEC and fr should be uncoupled bare frequency.
    g01 is not the original g in tridiagonal JC Hamiltonian. Their relation is g01=g<0|n|1>.
    So g_ij = g01 * <i|n|j>/<0|n|1>
    """
    
    Delta = np.zeros(shape=(matrix_size, matrix_size))
    g = np.zeros(shape=(matrix_size, matrix_size))

    q0 = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=30, truncated_dim=30)
    energies = q0.eigenvals(evals_count=evals_count)
    n_matrix_elements = q0.matrixelement_table(operator='n_operator', evals_count=evals_count)
    
    for i in range(matrix_size):
        for j in range(matrix_size):
            Delta[i][j] = np.abs(energies[j] - energies[i]) - fr
            g[i][j] = g01 * n_matrix_elements[i][j] / n_matrix_elements[0][1]

    ratio_matrix = np.abs(Delta/g)
    ratio_matrix = np.where(ratio_matrix<1000, ratio_matrix, 0)

    return ratio_matrix


def calculate_qubits_coupling(f01_1, alpha_1, f01_2, alpha_2, chi_zz):
    """
    From bare frequencies of qubits pairs and their χ_zz to calculate the original 
    coupling strenght g_qq and intra-transmon capacitance Cqq.
    All frequencies are in unit of [GHz]
    Ref: CircuitQED Eq.(149); Quantum Engineer's guide Eq.(104).
    Notice the χ_zz here is already the total frequency shift, not half of it.
    """
    if chi_zz < 0:
        print('chi_zz should better be positive!!!')
        
    # Determine which is control(c) which is target(t).
    if f01_1 < f01_2:
        f01_c = f01_1
        alpha_c = alpha_1
        f01_t = f01_2
        alpha_t = alpha_2
    else:
        f01_c = f01_2
        alpha_c = alpha_2
        f01_t = f01_1
        alpha_t = alpha_1
        
    delta_qq = f01_c - f01_t
    EJ_c, EC_c = falpha_to_EJEC(f01_c, alpha_c)
    EJ_t, EC_t = falpha_to_EJEC(f01_t, alpha_t)
    g_qq = np.sqrt(chi_zz * (delta_qq+EC_t) * (delta_qq-EC_c) / (-EC_c-EC_t))
    
    C_c = e**2 / (2*EC_c*physical_constants['Planck constant'][0]) * 1e6 # [fF]
    C_t = e**2 / (2*EC_t*physical_constants['Planck constant'][0]) * 1e6 # [fF]
    
    # You need to solve the quadratic equation:)
    R = (4*g_qq**2) / (f01_c * f01_t)
    Cqq = (-R*(C_c+C_t) - np.sqrt(R**2 * (C_c-C_t)**2 + 4*R*C_c*C_t) ) / (2*R-2) # [fF]

    return g_qq, Cqq
























