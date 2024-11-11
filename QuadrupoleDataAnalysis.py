# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:42:05 2024

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.optimize import curve_fit
import scipy.optimize as opt

pos_current = np.arange(-24,26,2)

B_z_0A = [3.15,2.87,2.61,2.34,2.07,1.81,1.54,1.29,1.03,0.77,0.51,0.25,0.00,-0.26,-0.51,-0.77,-1.03,-1.27,-1.53,-1.79,-2.04,-2.28,-2.52,-2.76,-2.99]
B_z_05A = [-70.6,-65.0,-59.4,-53.5,-47.5,-41.7,-34.6,-29.7,-23.8,-17.76,-11.87,-5.94,-0.05,6.01,12.01,17.86,23.85,29.80,35.8,41.8,47.8,53.9,59.7,65.6,71.2]
B_z_1A = [-138.8,-127.8,-116.6,-105.1,-93.2,-81.8,-70.0,-58.5,-46.8,-34.9,-23.2,-11.61,0.00,11.92,23.62,35.2,47.0,58.7,70.6,82.3,94.2,105.8,117.4,129.0,140.1]
B_z_15A = [-202.6,-185.7,-168.6,-152.0,-134.8,-118.3,-101.3,-84.3,-67.7,-50.6,-33.9,-16.89,0.05,16.79,34.0,50.6,67.4,84.6,101.6,118.5,135.1,152.6,169.4,186.6,203.7]
B_z_2A = [-266.7,-244.5,-222.2,-200.1,-177.4,-155.9,-133.5,-110.9,-89.1,-66.6,-44.6,-22.5,0.06,22.11,44.8,66.5,88.9,111.2,133.6,156.1,178.2,201.0,223.2,245.8,268.3]
B_z_25A = [-331.4,-303.3,-276.2,-248.4,-220.3,-192.8,-165.0,-137.8,-110.1,-82.3,-54.9,-27.1,0.0,27.7,55.6,83.1,110.8,138.1,166.2,194.1,222.0,250.1,277.8,306.5,334.6]

B_x_0A = [-1.47,-1.35,-1.24,-1.11,-0.99,-0.86,-0.73,-0.61,-0.48,-0.36,-0.25,-0.12,0.00,0.11,0.22,0.32,0.45,0.55,0.66,0.76,0.87,0.98,1.08,1.18,1.28]
B_x_05A = [70.7,64.9,59.4,53.5,47.4,41.8,35.5,29.7,23.8,17.63,12.04,5.01,0.00,-5.84,-11.95,-17.62,-23.74,-29.58,-35.5,-41.6,-47.3,-53.3,-59.2,-64.7,-70.4]
B_x_1A = [138.0,127.9,116.3,104.6,93.5,81.1,70.0,58.1,46.2,35.1,22.8,11.70,-0.12,-12.12,-23.05,-35.2,-46.4,-58.1,-70.0,-81.3,-93.2,-104.6,-116.1,-127.7,-138.1]
B_x_15A = [198.2,183.8,167.1,150.5,134.7,116.7,101.0,83.6,66.7,50.8,33.0,17.07,0.00,-16.95,-32.9,-50.4,-66.5,-83.5,-100.4,-116.7,-133.7,-150.1,-166.6,-183.1,-198.1]
B_x_2A = [260.6,241.6,219.7,197.6,177.1,153.3,132.6,109.9,87.3,66.8,43.5,22.4,0.00,-22.37,-43.3,-66.2,-87.4,-109.6,-132.0,-153.2,-175.9,-197.3,-218.9,-240.6,-260.1]
B_x_25A = [325.0,302.0,274.1,246.9,220.5,191.2,165.7,136.8,109.1,82.8,53.9,28.2,-0.20,-27.70,-54.0,-82.6,-108.5,-136.7,-164.6,-191.0,-219.4,-245.5,-273.1,-299.9,-324.2]

B_z_length = [141.7,141.2,140.7,140.6,140.0,139.8,139.1,138.4,138.0,137.4,136.9,136.5,135.9,135.2,134.3,133.3,132.4,130.8,129.6,127.6,125.6,123.1,119.8,116.3,112.9,108.7,104.3,99.8,94.7,89.8,85.0,80.1,75.1,70.3,49.3,33.3,22.2,14.7,9.7,6.5,4.4,3.1,2.2,1.6,1.2,0.9,0.7,0.6,0.5]
pos_length = [16,14,12,10,8,6,4,2,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-30,-35,-40,-45,-50,-55,-60,-65,-70,-75,-80,-85,-90,-95,-100]

current_am = [0, 0.021, 0.039, 0.061, 0.082, 0.101, 0.12, 0.14]
aligned_x = np.array([0.75, 0.72, 0.68, 0.60, 0.57, 0.50, 0.45, 0.43]) / 10**3 #jetzt in [m]
misaligned_x = np.array([0.84, 0.69, 0.51, 0.38, 0.27, 0.13, -0.05, -0.20]) / 10**3 #jetzt in [m]
aligned_y = np.array([-3.62, -3.80, -3.89, -4.15, -4.42, -4.48, -4.83, -4.98]) / 10**3 #jetzt in [m]
misaligned_y = np.array([-3.64, -3.83, -3.99, -4.13, -4.30, -4.38, -5.06, -4.85]) / 10**3 #jetzt in [m]
errors_am = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]) / 10**3 #jetzt in [m]

current_bs = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
s_i = np.array([1.32828, 0.97648, 0.75455, 0.67524, 0.88940, 1.01929, 1.39344]) / 10**3 #[m]
matrix = np.array([[1.7903448898864113*10**(-5), -3.2379715866128857*10**(-5)], [-3.2379715866128857*10**(-5), 5.939762497451817*10**(-5)]])  #sollte m^2, m*rad, rad^2 sein
L_drift = 0.75 #[m]

def plotsBz(x_Achse, y_Achse, colour, name):
    plt.plot(x_Achse, y_Achse, colour, label=name)
    plt.title("Magnetic Field z Component Measurement for all currents I")
    plt.ylabel("Magnetic Field Components B_z [G]")
    plt.xlabel("Position x [mm]")
    plt.grid(True)
    #plt.errorbar(x_Achse, y_Achse, yerr=error(y_Achse), ls="none")
    plt.legend()
    
def plotsBx(x_Achse, y_Achse, colour, name):
    plt.plot(x_Achse, y_Achse, colour, label=name)
    plt.title("Magnetic Field x Component Measurement for all currents I")
    plt.ylabel("Magnetic Field Components B_x [G]")
    plt.xlabel("Position z [mm]")
    plt.grid(True)
    #plt.errorbar(x_Achse, y_Achse, yerr=error(y_Achse), ls="none")
    plt.legend()
    
def plotlength(x_Achse, y_Achse, colour, name):
    plt.plot(x_Achse, y_Achse, colour, label=name, markersize=4)
    plt.title("Magnetic Field Measurement for I=2,5A at x=10mm and z=0mm ")
    plt.ylabel("Magnetic Field Components B_z [G]")
    plt.xlabel("Position  y [mm]")
    plt.grid(True)
    #plt.errorbar(x_Achse, y_Achse, yerr=error(y_Achse), ls="none")
    #plt.legend()
    plt.show()

def plot_Allg(x_Achse, y_Achse, y_error, title, xlabel, ylabel, name, ylim):
    plt.plot(x_Achse, y_Achse, label=name, marker="o", markersize=4, linestyle="none")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.errorbar(x_Achse, y_Achse, yerr=y_error, fmt="none", capsize=4)
    plt.grid(True)
    plt.ylim(ylim)
    plt.xticks(x_Achse)
    plt.show()


def plot_Allg_fit(x, y, y_error, title, xlabel, ylabel, filename, mode, fit_params=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=y_error, fmt='o', label='Data', capsize=5)
    #plt.plot(current_am, misaligned_x, marker="o", markersize=4, linestyle="none")

   
    # Add fit line if fit parameters are provided
    if mode == 1:
        if fit_params is not None:
            m, b = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = m*x_fit + b
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x + {:.2f}'.format(m, b), color='red')
            
    elif mode == 2:
        if fit_params is not None:
            a, b, c = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = a * x_fit**2 + b * x_fit + c
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x² + {:.2f}x + {:.2f}'.format(a, b, c), color='red')
    
    elif mode == 3:
        if fit_params is not None:
            a, b, c, d = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit + d
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x² + {:.2f}x + {:.2f}'.format(a, b, c, d), color='red')
    
    else:
        if fit_params is not None:
            m, b = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = m / x_fit + b
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x + {:.2f}'.format(m, b), color='red')
                
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xticks(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend()
    plt.grid(True)
    #plt.savefig(f"{filename}.png")
    plt.show()
    
    
def plot_Allg_noerror(x_Achse, y_Achse, title, xlabel, ylabel, name, ylim):
    plt.plot(x_Achse, y_Achse, label=name, marker="o", markersize=4, linestyle="none")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.errorbar(x_Achse, y_Achse, fmt="none", capsize=4)
    plt.grid(True)
    plt.xticks(x_Achse)
    plt.show()

 
def plot_Allg_noerror_fit(x, y, title, xlabel, ylabel, filename, mode, fit_params=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Data')
    
    # Add fit line if fit parameters are provided
    if mode == 1:
        if fit_params is not None:
            m, b = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = m*x_fit + b
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x + {:.2f}'.format(m, b), color='red')
            
    elif mode == 2:
        if fit_params is not None:
            a, b, c = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = a * x_fit**2 + b * x_fit + c
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x² + {:.2f}x + {:.2f}'.format(a, b, c), color='red')
    
    elif mode == 3:
        if fit_params is not None:
            a, b, c, d = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit + d
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x² + {:.2f}x + {:.2f}'.format(a, b, c, d), color='red')
    
    else:
        if fit_params is not None:
            m, b = fit_params
            x_fit = np.linspace(min(x), max(x), 100)  # Create a smooth line
            y_fit = m / x_fit + b
            plt.plot(x_fit, y_fit, '--', label='Fit: y = {:.2f}x + {:.2f}'.format(m, b), color='red')
                
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xticks(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend()
    plt.grid(True)
    #plt.savefig(f"{filename}.png")
    plt.show()


def error(Achse):
    err = []
    for i in range(len(Achse)):
        err.append(abs(0.01*Achse[i] + 0.1))
        #err[i] = err[i] + 0.1
    return err

def line_model(params, x):
    x = np.array(x)  # Stelle sicher, dass x ein numpy Array ist
    y = params[0] * x + params[1]
    return y

def get_chisq(params, data_x, data_y):
    model = line_model(params, data_x)
    dY = data_y - model
    return np.sum((dY / error(data_y))**2)

def get_parameter_errors(params, data_x, data_y, chi2_min, delta_chi2=1.0):
    # Funktion zur Anpassung von Parametern, um Fehler zu berechnen
    errors = []
    
    for i in range(len(params)):
        def chi2_for_param_shift(param_shift):
            # Verändere nur den aktuellen Parameter, die anderen bleiben konstant
            new_params = params.copy()
            if isinstance(param_shift, np.ndarray) and param_shift.size == 1:
                param_shift = param_shift.item()  # Converts single-element array to a scalar

            new_params[i] += param_shift
            return get_chisq(new_params, data_x, data_y) - chi2_min
        
        # Suche nach der Änderung des Parameters, die den Chi2-Wert um delta_chi2 (z.B. 1) erhöht
        error = fmin(lambda shift: abs(chi2_for_param_shift(shift) - delta_chi2), 0.1, disp=False)[0]
        errors.append(error)
        
    return np.array(errors)

# Quadratische Funktion
def quadratic(x, a, b, c):
    x = np.array(x)
    return a * x**2 + b * x + c

# Chi-Quadrat-Funktion
def chi_square(observed, expected, errors):
    return np.sum(((observed - expected) / errors) ** 2)

def cubic_function(x, a, b, c, d):
    x = np.array(x)
    return a * x**3 + b * x**2 + c * x + d

def cubic_fit(x, y):
    popt, _ = curve_fit(cubic_function, x, y)
    return popt

# Definiere die Funktion der Form m/x + b
def linear_reciprocal(x, m, b):
    x = np.array(x)
    return m / x + b

# Funktion für den Fit
def linear_reciprocal_fit(x, y):
    popt, _ = curve_fit(linear_reciprocal, x, y)
    return popt

# Funktion für den Chi-Quadrat-Fit
def chi_square_fit(x, y, y_err):
    # Curve Fit durchführen
    popt, pcov = curve_fit(cubic_function, x, y, sigma=y_err, absolute_sigma=True)
    
    # Berechnung von Chi-Quadrat
    residuals = y - cubic_function(x, *popt)
    chi2 = np.sum((residuals / y_err) ** 2)
    dof = len(x) - len(popt)  # Freiheitsgrade
    reduced_chi2 = chi2 / dof
    
    # Unsicherheiten der Fitparameter (sqrt der Diagonale der Kovarianzmatrix)
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr, chi2, reduced_chi2

def effective_length_with_error(B_z_length, pos_length):
    # 1% Fehleranteil
    error_percentage = 0.01
    pos_length = np.array(pos_length)  / 10**3
    # Berechne das Integral über B_z_length
    integral = abs(np.trapz(B_z_length, pos_length))
    
    # Finde den höchsten Wert in B_z_length
    max_B_z = max(B_z_length)
    
    # Normiertes Ergebnis
    result = ((integral / max_B_z) * 2)
    
    # Fehler in integral und max_B_z
    delta_integral = integral * error_percentage
    delta_max_B_z = max_B_z * error_percentage
    
    # Fehlerfortpflanzung für result
    delta_result = np.sqrt((2 / max_B_z * delta_integral)**2 + 
                           (-2 * integral / max_B_z**2 * delta_max_B_z)**2)
    
    return result, delta_result

def calculate_f(s, matrix, delta_s, delta_length_eff):
    # Extrahiere die Mittelwerte aus der Matrix
    x2_mean = matrix[0, 0]      # <x^2>
    xx_prime_mean = matrix[0, 1] # <xx'>
    x_prime2_mean = matrix[1, 1] # <x'^2>
    
    focal_length = []
    uncertainties = []
    
    for i in range(len(s)):
        # Berechnung von epsilon
        epsilon = np.sqrt(x2_mean * x_prime2_mean - xx_prime_mean**2)
        sqrt_value = np.sqrt(abs(-L_drift**4 * epsilon**2 + L_drift**2 * x2_mean * (s[i])**2)) #abs!!!!!!!!!
    
        # Zähler der Formel für f
        if i < 4:
            numerator = (L_drift * (x2_mean + L_drift * xx_prime_mean) + 
                         sqrt_value) 
        else:
            numerator = (L_drift * (x2_mean + L_drift * xx_prime_mean) -
                         sqrt_value) 

        # Nenner der Formel für f
        denominator = (x2_mean + L_drift * (L_drift * x_prime2_mean + 2 * xx_prime_mean) - (s[i])**2)

        # Berechnung des Ergebnisses
        f = numerator / denominator
        focal_length.append(f)
        
        # Berechne partielle Ableitungen nach s
        #df_ds = (-L_drift**2 * x2_mean * s[i] / sqrt_value - 2 * s[i]) / denominator
        df_ds = ((L_drift**2 * x2_mean * s[i] / sqrt_value) + (numerator * 2 * s[i] / denominator)) / denominator
        # Berechne die Unsicherheit in f
        delta_f = np.sqrt((df_ds * delta_s)**2) #+ (df_dlength_eff * delta_length_eff)**2)
        uncertainties.append(delta_f)

    return focal_length, uncertainties
    
def energy(focal_length, current_bs, steigung, absatz):
    energy_beam = []
    energy_beam_c = []
    for i in range(len(focal_length)):
        g = steigung * current_bs[i] + absatz
        p_c =  1.602*10**(-19) * g * focal_length[i] * length_eff
        p = (p_c * 3 * 10**8) / (1.602 * 10**(-19))     #* 3 * 10**8
        energy_beam.append(p)
        energy_beam_c.append(p_c)
    #energy = np.mean(energy_beam)
    #energy_error = np.std(energy_beam)
    return energy_beam, energy_beam_c

def offset_sigma(a, p, L_d, L_eff, Kappa, a_error, p_error, L_drift_error, L_eff_error, Kappa_error):
    
    offset_value = (a * p) / (L_d * L_eff * 1.602*10**(-19) * Kappa)
    
    offset_error = np.sqrt(
        (p / (L_d * L_eff * Kappa * 1.602*10**(-19)) * a_error) ** 2 +
        (a / (L_d * L_eff * Kappa * 1.602*10**(-19)) * p_error) ** 2 +
        ((-a * p) / (L_d**2 * L_eff * Kappa * 1.602*10**(-19)) * L_drift_error) ** 2 +
        ((-a * p) / (L_d * L_eff**2 * Kappa * 1.602*10**(-19)) * L_eff_error) ** 2 +
        ((-a * p) / (L_d * L_eff * Kappa**2 * 1.602*10**(-19)) * Kappa_error) ** 2)
    
    return offset_value, offset_error
    
def get_k(f, l_eff, f_error, l_eff_error):
    k = []
    dk = []
    for i in range(len(f)):
        k_val = 1 / (f[i] * l_eff)
        k.append(k_val)
        
        dk_df = -(1 / (f[i]**2 * l_eff))
        dk_dl = -(1 / (f[i] * l_eff**2))
    
        dk_val = np.sqrt((dk_df * f_error[i])**2 + (dk_dl * l_eff_error))
        dk.append(dk_val)
        
    return k, dk


plotsBz(pos_current, B_z_0A, "bo", "I=0A")
plotsBz(pos_current, B_z_05A, "ro", "I=0,5A")
plotsBz(pos_current, B_z_1A, "go", "I=1A")
plotsBz(pos_current, B_z_15A, "co", "I=1,5A")
plotsBz(pos_current, B_z_2A, "mo", "I=2A")
plotsBz(pos_current, B_z_25A, "yo", "I=2,5A")
plt.show()
plotsBx(pos_current, B_x_0A, "bo", "I=0A")
plotsBx(pos_current, B_x_05A, "ro", "I=0,5A")
plotsBx(pos_current, B_x_1A, "go", "I=1A")
plotsBx(pos_current, B_x_15A, "co", "I=1,5A")
plotsBx(pos_current, B_x_2A, "mo", "I=2A")
plotsBx(pos_current, B_x_25A, "yo", "I=2,5A")
plt.show()
plotlength(pos_length, B_z_length, "bo", "Jo :)")

###################################

gradB_z = []
gradB_z_errors = []

#error = fmin(lambda shift: abs(chi2_for_param_shift(shift) - delta_chi2), params[i] * 0.1, disp=False)[0] ??

guess_z_0A = 0, 0
chi2_z_0A = fmin(get_chisq, guess_z_0A, args=(pos_current,B_z_0A,))
gradB_z.append(chi2_z_0A[0])
chi2_min_z_0A = get_chisq(chi2_z_0A, pos_current, B_z_0A)
errors_z_0A = get_parameter_errors(chi2_z_0A, pos_current, B_z_0A, chi2_min_z_0A)
gradB_z_errors.append(errors_z_0A[0])

guess_z_05A = 3, 0
chi2_z_05A = fmin(get_chisq, guess_z_05A, args=(pos_current,B_z_05A,))
gradB_z.append(chi2_z_05A[0])
chi2_min_z_05A = get_chisq(chi2_z_05A, pos_current, B_z_05A)
errors_z_05A = get_parameter_errors(chi2_z_05A, pos_current, B_z_05A, chi2_min_z_05A)
gradB_z_errors.append(errors_z_05A[0])

guess_z_1A = 5.8, 0
chi2_z_1A = fmin(get_chisq, guess_z_1A, args=(pos_current,B_z_1A,))
gradB_z.append(chi2_z_1A[0])
chi2_min_z_1A = get_chisq(chi2_z_1A, pos_current, B_z_1A)
errors_z_1A = get_parameter_errors(chi2_z_1A, pos_current, B_z_1A, chi2_min_z_1A)
gradB_z_errors.append(errors_z_1A[0])

guess_z_15A = 8.5, 0
chi2_z_15A = fmin(get_chisq, guess_z_15A, args=(pos_current,B_z_15A,))
gradB_z.append(chi2_z_15A[0])
chi2_min_z_15A = get_chisq(chi2_z_15A, pos_current, B_z_15A)
errors_z_15A = get_parameter_errors(chi2_z_15A, pos_current, B_z_15A, chi2_min_z_15A)
gradB_z_errors.append(errors_z_15A[0])

guess_z_2A = 11.1, 0
chi2_z_2A = fmin(get_chisq, guess_z_2A, args=(pos_current,B_z_2A,))
gradB_z.append(chi2_z_2A[0])
chi2_min_z_2A = get_chisq(chi2_z_2A, pos_current, B_z_2A)
errors_z_2A = get_parameter_errors(chi2_z_2A, pos_current, B_z_2A, chi2_min_z_2A)
gradB_z_errors.append(errors_z_2A[0])

guess_z_25A = (13, 0)
chi2_z_25A = fmin(get_chisq, guess_z_25A, args=(pos_current, B_z_25A,))
gradB_z.append(chi2_z_25A[0])
chi2_min_z_25A = get_chisq(chi2_z_25A, pos_current, B_z_25A)
errors_z_25A = get_parameter_errors(chi2_z_25A, pos_current, B_z_25A, chi2_min_z_25A)
gradB_z_errors.append(errors_z_25A[0])

##########################################

gradB_x = []
gradB_x_errors = []

guess_x_0A = 0, 0
chi2_x_0A = fmin(get_chisq, guess_x_0A, args=(pos_current,B_x_0A,))
gradB_x.append(chi2_x_0A[0])
chi2_min_x_0A = get_chisq(chi2_x_0A, pos_current, B_x_0A)
errors_x_0A = get_parameter_errors(chi2_x_0A, pos_current, B_x_0A, chi2_min_x_0A)
gradB_x_errors.append(errors_x_0A[0])

guess_x_05A = 3, 0
chi2_x_05A = fmin(get_chisq, guess_x_05A, args=(pos_current,B_x_05A,))
gradB_x.append(chi2_x_05A[0])
chi2_min_x_05A = get_chisq(chi2_x_05A, pos_current, B_x_05A)
errors_x_05A = get_parameter_errors(chi2_x_05A, pos_current, B_x_05A, chi2_min_x_05A)
gradB_x_errors.append(errors_x_05A[0])

guess_x_1A = 5.8, 0
chi2_x_1A = fmin(get_chisq, guess_x_1A, args=(pos_current,B_x_1A,))
gradB_x.append(chi2_x_1A[0])
chi2_min_x_1A = get_chisq(chi2_x_1A, pos_current, B_x_1A)
errors_x_1A = get_parameter_errors(chi2_x_1A, pos_current, B_x_1A, chi2_min_x_1A)
gradB_x_errors.append(errors_x_1A[0])

guess_x_15A = 8.5, 0
chi2_x_15A = fmin(get_chisq, guess_x_15A, args=(pos_current,B_x_15A,))
gradB_x.append(chi2_x_15A[0])
chi2_min_x_15A = get_chisq(chi2_x_15A, pos_current, B_x_15A)
errors_x_15A = get_parameter_errors(chi2_x_15A, pos_current, B_x_15A, chi2_min_x_15A)
gradB_x_errors.append(errors_x_15A[0])

guess_x_2A = 11.1, 0
chi2_x_2A = fmin(get_chisq, guess_x_2A, args=(pos_current,B_x_2A,))
gradB_x.append(chi2_x_2A[0])
chi2_min_x_2A = get_chisq(chi2_x_2A, pos_current, B_x_2A)
errors_x_2A = get_parameter_errors(chi2_x_2A, pos_current, B_x_2A, chi2_min_x_2A)
gradB_x_errors.append(errors_x_2A[0])

guess_x_25A = (13, 0)
chi2_x_25A = fmin(get_chisq, guess_x_25A, args=(pos_current, B_x_25A,))
gradB_x.append(chi2_x_25A[0])
chi2_min_x_25A = get_chisq(chi2_x_25A, pos_current, B_x_25A)
errors_x_25A = get_parameter_errors(chi2_x_25A, pos_current, B_x_25A, chi2_min_x_25A)
gradB_x_errors.append(errors_x_25A[0])

#########################################

current = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
gradB_z = np.array(gradB_z) * 0.1
gradB_z_errors = np.array(gradB_z_errors) * 0.1
gradB_x = np.array(gradB_x) * 0.1
gradB_x_errors = np.array(gradB_x_errors) * 0.1


guess_gradB_z = (6, 0)
chi2_gradB_z = fmin(get_chisq, guess_gradB_z, args=(current, gradB_z))
chi2_min_gradB_z = get_chisq(chi2_gradB_z, current, gradB_z)
errors_gradB_z = get_parameter_errors(chi2_gradB_z, current, gradB_z, chi2_min_gradB_z)

guess_gradB_x = (6, 0)
chi2_gradB_x = fmin(get_chisq, guess_gradB_x, args=(current, gradB_x))
chi2_min_gradB_x = get_chisq(chi2_gradB_x, current, gradB_x)
errors_gradB_x = get_parameter_errors(chi2_gradB_x, current, gradB_x, chi2_min_gradB_x)

########################################

guess_a_x = (1, 0)

chi2_a_x = fmin(get_chisq, guess_a_x, args=(current_am, aligned_x * 10**3))
params_a_x = chi2_a_x
chi2_min_a_x = get_chisq(chi2_a_x, current_am, aligned_x * 10**3)
errors_a_x = get_parameter_errors(chi2_a_x, current_am, aligned_x * 10**3, chi2_min_a_x)
params_errors_a_x = errors_a_x

guess_m_x = (10, 0)

chi2_m_x = fmin(get_chisq, guess_m_x, args=(current_am, misaligned_x * 10**3))
params_m_x = chi2_m_x
chi2_min_m_x = get_chisq(chi2_m_x, current_am, misaligned_x * 10**3)
errors_m_x = get_parameter_errors(chi2_m_x, current_am, misaligned_x * 10**3, chi2_min_m_x)
params_errors_m_x = errors_m_x

guess_a_y = (10, 0)

chi2_a_y = fmin(get_chisq, guess_a_y, args=(current_am, aligned_y * 10**3))
params_a_y = chi2_a_y
chi2_min_a_y = get_chisq(chi2_a_y, current_am, aligned_y * 10**3)
errors_a_y = get_parameter_errors(chi2_a_y, current_am, aligned_y * 10**3, chi2_min_a_y)
params_errors_a_y = errors_a_y

guess_m_y = (10, 0)

chi2_m_y = fmin(get_chisq, guess_m_y, args=(current_am, misaligned_y * 10**3))
params_m_y = chi2_m_y
chi2_min_m_y = get_chisq(chi2_m_y, current_am, misaligned_y * 10**3)
errors_m_y = get_parameter_errors(chi2_m_y, current_am, misaligned_y * 10**3, chi2_min_m_y)
params_errors_m_y = errors_m_y

############################

plot_Allg_fit(current_am, misaligned_x * 10**3, errors_am * 10**3, "Horizontal Beam Position for the Misaligned Quadrupole", "Current I [A]", "horizonatl position x [mm]", "test", 1, params_m_x)#, ylim=(-0.3, 1))#, params_m_x) für errors dann errors_am benutzen
plot_Allg_fit(current_am, misaligned_y * 10**3, errors_am * 10**3, "Vertical Beam Position for the Misaligned Quadrupole", "Current I [A]", "vertical position z [mm]", "test", 1, params_m_y)#, ylim=(-5.3, -3.5))#, params_m_y) für errors dann errors_am benutzen
plot_Allg_fit(current_am, aligned_x * 10**3, errors_am * 10**3, "Horizontal Beam Position for the Aligned Quadrupole", "Current I [A]", "horizontal position x [mm]", "test", 1, params_a_x)#, ylim=(-0.3, 1))#, params_a_x) für errors dann errors_am benutzen
plot_Allg_fit(current_am, aligned_y * 10**3, errors_am * 10**3, "Vertical Beam Position for the Aligned Quadrupole", "Current I [A]", "vertical position z [mm]", "test", 1, params_a_y)#, ylim=(-5.3, -3.5))#, params_a_y) für errors dann errors_am benutzen

#plt.plot(current_am, misaligned_x, marker="o", markersize=4, linestyle="none")
#plt.show()

############################

#plot_Allg_noerror(current, gradB_z, "Vertical Gradient of the Quadrupole","Current I [A]" ,"Verticaal Gradient g_z [T/m]", "test", (1.5, 0))
#plot_Allg_noerror(current, gradB_x, "Horizontal Gradient of the Quadrupole","Current I [A]" ,"Horizontal Gradient g_x [T/m]", "test", (-1.5, 0))

plot_Allg_noerror_fit(current, gradB_z, "Vertical Gradient of the Quadrupole",
                  "Current I [A]", "Vertical Gradient g_z [T/m]", "test_vertical", 1, fit_params=chi2_gradB_z)

plot_Allg_noerror_fit(current, gradB_x, "Horizontal Gradient of the Quadrupole",
                  "Current I [A]", "Horizontal Gradient g_x [T/m]", "test_horizontal", 1, fit_params=chi2_gradB_x) 
#plot_Allg_noerror_fit()
###########################

length_eff, length_eff_error = effective_length_with_error(B_z_length, pos_length)

focal_length, focal_length_error  = calculate_f(s_i, matrix, 0, length_eff_error) #std_s austauschen
#focal_length_error = np.array(focal_length_error) * 0
#plot_Allg(current_bs, focal_length, focal_length_error, "Focal Length of the Quadrupole", "Current I[A]", "Focal Length f[m]", "test", (-2.2, -0.8)) #(-2.2, -0.8)

#plt.show()
#plt.plot(current_bs, focal_length)

k, k_error = get_k(focal_length, length_eff, focal_length_error, length_eff_error)
#plot_Allg(current_bs, k, k_error, "Strength of the Quadrupole", "Current I[A]", "Quadrupole Strength k[1/m^2]", "test", (-11, -5))

plot_Allg_noerror(current_bs, s_i * 10**3, "Beam Spread for different Currents", "Current I[A]", "Beam Spread s[mm]", "test", (0.5, 1.5))

##########################

p, p_c = energy((np.array(focal_length)), current_bs, chi2_gradB_z[0], chi2_gradB_z[1])
print(p)
print("p mean:", np.mean(p)) #sollte in eV sein (bitte bitte hoffentlich) 
print("p standardabweichung:", np.std(p))
print("relative error:", np.std(p)/np.mean(p))

p_c = np.array(p_c) / (1.602 * 10**(-19))
print(p_c)
print("p_c mean:", np.mean(p_c)) #sollte in eV sein (bitte bitte hoffentlich) 
print("p_c standardabweichung:", np.std(p_c))
print("relative error:", np.std(p_c)/np.mean(p_c))

offset_mx, offset_mx_error = offset_sigma(params_m_x[0], 3.6*10**(6)*1.602*10**(-19)/(3*10**8), L_drift, length_eff, chi2_gradB_z[0], errors_m_x[0], 0, 0, length_eff_error, errors_gradB_z[0])
print("_______")
print("offset mx:", offset_mx)
print("offset mx error:", offset_mx_error)
print("relative error:", offset_mx_error/offset_mx)

offset_ax, offset_ax_error = offset_sigma(params_a_x[0], 3.6*10**(6)*1.602*10**(-19)/(3*10**8), L_drift, length_eff, chi2_gradB_z[0], errors_a_x[0], 0, 0, length_eff_error, errors_gradB_z[0])
print("_______")
print("offset ax:", offset_ax)
print("offset ax error:", offset_ax_error)
print("relative error:", offset_ax_error/offset_ax)

offset_my, offset_my_error = offset_sigma(params_m_y[0], 3.6*10**(6)*1.602*10**(-19)/(3*10**8), L_drift, length_eff, chi2_gradB_z[0], errors_m_y[0], 0, 0, length_eff_error, errors_gradB_z[0])
print("_______")
print("offset my:", offset_my)
print("offset my error:", offset_my_error)
print("relative error:", offset_my_error/offset_my)

offset_ay, offset_ay_error = offset_sigma(params_a_y[0], 3.6*10**(6)*1.602*10**(-19)/(3*10**8), L_drift, length_eff, chi2_gradB_z[0], errors_a_y[0], 0, 0, length_eff_error, errors_gradB_z[0])
print("_______")
print("offset ay:", offset_ay)
print("offset ay error:", offset_ay_error)
print("relative error:", offset_my_error/offset_ay)

###########################

# Initiale Schätzung der Parameter [a, b, c]
initial_guess = [1, 1, 1]

# Curve Fit, um optimale Parameter zu finden
f_params, f_params_covariance = opt.curve_fit(quadratic, current_bs, focal_length, sigma=focal_length_error, p0=initial_guess, absolute_sigma=True)

# Berechnete y-Werte mit den optimalen Parametern
y_fit = quadratic(current_bs, *f_params)

# Berechne das Chi-Quadrat
chi2 = chi_square(focal_length, y_fit, focal_length_error)
dof = len(current_bs) - len(f_params)  # Freiheitsgrade
chi2_reduced = chi2 / dof

# Fehler der Parameter (Standardabweichung = sqrt(Kovarianzmatrix-Diagonale))
f_param_errors = np.sqrt(np.diag(f_params_covariance))

plot_Allg_noerror_fit(current_bs, focal_length, "Focal Length of the Quadrupole", "Current I[A]", "Focal Length f[m]", "test", 2, f_params)

# Ergebnis
print("Gefundene Parameter f (a, b, c):", f_params)
print("Fehler der Parameter f (a, b, c):", f_param_errors)
print("Chi-Quadrat:", chi2)
print("Reduziertes Chi-Quadrat:", chi2_reduced)
"""
"""
# Fit anwenden
f_fit_params = cubic_fit(current_bs, focal_length)
# Ergebnisse ausgeben
print("focal_length Fit-Parameter:", f_fit_params)
plot_Allg_noerror_fit(current_bs, focal_length, "Focal Length of the Quadrupole", "Current I[A]", "Focal Length f[m]", "test", 3, f_fit_params)


# Fit anwenden
f_fit_params_rez = linear_reciprocal_fit(current_bs, focal_length)
# Ergebnisse ausgeben
print("Fit-Parameter (m, b):", f_fit_params_rez)
plot_Allg_noerror_fit(current_bs, focal_length, "Focal Length of the Quadrupole", "Current I[A]", "Focal Length f[m]", "test", 4, f_fit_params_rez)


#################################

# Initiale Schätzung der Parameter [a, b, c]
initial_guess = [1, 1, 1]

# Curve Fit, um optimale Parameter zu finden
k_params, k_params_covariance = opt.curve_fit(quadratic, current_bs, k, sigma=k_error, p0=initial_guess, absolute_sigma=True)

# Berechnete y-Werte mit den optimalen Parametern
y_fit = quadratic(current_bs, *k_params)

# Berechne das Chi-Quadrat
chi2 = chi_square(k, y_fit, k_error)
dof = len(current_bs) - len(k_params)  # Freiheitsgrade
chi2_reduced = chi2 / dof

# Fehler der Parameter (Standardabweichung = sqrt(Kovarianzmatrix-Diagonale))
k_param_errors = np.sqrt(np.diag(k_params_covariance))

plot_Allg_fit(current_bs, k, k_error, "Strength of the Quadrupole", "Current I[A]", "Quadrupole Strength k[1/m^2]", "test", 2, k_params)

print("Gefundene Parameter k (a, b, c):", k_params)
print("Fehler der Parameter k (a, b, c):", k_param_errors)
print("Chi-Quadrat:", chi2)
print("Reduziertes Chi-Quadrat:", chi2_reduced)

guess_k = (1, 0)

chi2_k = fmin(get_chisq, guess_k, args=(current_bs, k))
params_k = chi2_k
chi2_min_k = get_chisq(chi2_k, current_bs, k)
errors_k = get_parameter_errors(chi2_k, current_bs, k, chi2_min_k)
params_errors_k = errors_k
print("k fit params lin:", params_k)
print("k fit param errors lin:", params_errors_k)

plot_Allg_fit(current_bs, k, k_error, "Strength of the Quadrupole", "Current I[A]", "Quadrupole Strength k[1/m^2]", "test", 1, params_k)


