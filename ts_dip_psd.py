# pip install matplotlib numpy pandas ruptures

""" __________________________________________________________________________________________________
# GNSS TIME-SERIES DISPLACEMENT ANALYSIS (ITRF2020-like analysis linear+seasonal+psd model)

    INPUT: .pos/ .txyz2+.tenv/ .cts gnss daily solutions
        compatible data sources
           ITRF2020 psd: https://itrf.ign.fr/ftp/pub/itrf/itrf2020/ITRF2020-psd-gnss.snx
            .txyz/ .tenv NEVADA GEODETIC LABORATORY: https://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap_MAG.html
            .cts Dionysos Satellite Observatory NTUA: http://dionysos.survey.ntua.gr/dsoportal/_dataanalysis/BERN52PROC/stationlist.php#
            .pos Data Terra: https://en.poleterresolide.fr/geodesy-plotter-en/#/?bounds=7.602539062500001,25.898761936567023,49.5703125,42.35854391749705

    OUTPUT: plotting: 
                1. the per component time-series analysis containing the raw solutions (FOR ERRORBARS UNCOMMENT THEIR RESPECTIVE LINES IN PLOTTING), linear + seasonal + Jumps(if existing) fitted model. 
                   If detected jump labeled earthquake, the linear+seasonal+Jumps(if existing)+psd fitted model compared to the linear+seasonal trend
                2. the per component time-series analysis containing the residuals of the linear + seasonal + Jumps(if existing) fitted model. If detected jump labeled earthquake, the 
                   residuals of the linear+seasonal+Jumps(if existing)+psd fitted model compared to the linear+seasonal trend residuals
                 
            printing per component:
                1. if jumps detected the instructions of using the csv file, if not proceeding with linear+seasonal model
                2. if earthquake added prints its epoch and proceeds with linear+seasonal+psd model if not proceeding with linear+seasonal model
                3. the parameters of the fitted model with their respective uncertainties (intercept, velocity, annual1, annual2, semi-annual1, semi-annual2, Jumps)
                
                if there is a psd
                4. the single term exp/log selected and ssr comparison with no psd model
                5. the final model selected after testing the second term and ssr comparison with single psd model
                6. final check of comparing ssr of +psd model and of linear+sasonal model
                7. the parameters of the +psd fitted model with their respective uncertainties (constast, velocity, annual1, annual2, semi-annual1, semi-annual2, Jumps)+
                   the computed amplitude and tau with their cov and t0:earthquake epoch
                
 
    !!! WHEN RUNNING IN THE csv-popup OF AUTO JUMP DETECTION ADD 'eq' IN THE 'label' COLUMN OF THE JUMP WHICH EPOCHS CORRESPONDS TO AN EARTHQUAKE FOR PSD TO TAKE PLACE!!!
    !!! if a XXXX_PELT_jumps.csv exists in the path of your timeseries it will be used for jumps without any further auto-jump detection
    !!! UNCOMMENT LINES OF ERRORBARS TO INSPECT ERRORS
___________________________________________________________________________________________________"""

import numpy as np
import subprocess
import statistics
import sys
import datetime
from datetime import datetime as dt, timedelta
from matplotlib import pyplot as plt
import math
from pathlib import Path
from itertools import chain
import pandas as pd  
import ruptures as rpt
import matplotlib.dates as mdates
import os

# !!!!!!!!!!!!!!!!!!!!!!!!!!Path 
PATH = r"C:\Users\georg\Desktop\thesis\data\IGS_DYNG00GRC.pos"         #change path for .cts or .txyz2 or .pos file

#_____________________________________________________Model Parameters_______________________________________________________
# Coefficients of the equation being modeled 
# y = a * x + b + c1 * math.sin(w1 * x) + d1 * math.cos(w1 * x) + c2 * math.sin(w2 * x) + d2 * math.cos(w2 * x)... + J1 + ... Jn

angular_freq = [2 * math.pi * 1, 2 * math.pi * 2]
SIGMA0 = 0.001
EPSILON = 0.006694380 

# initial PSD parameters
A0_N, tau0_N = -0.1, 1      # North
A0_E, tau0_E = 0.15, 0.1       # East
A0_U, tau0_U = 0.001, 0.01   # Up

#helpers for saving terminal
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

#____________________________________________________PSD model_________________________________________________________
# post seismic deformation

def psd_term_and_derivatives(t_list, A, tau, t0, model_type):
    t_list = list(t_list)
    f    = np.zeros(len(t_list), dtype=float)
    dA   = np.zeros(len(t_list), dtype=float)
    dTau = np.zeros(len(t_list), dtype=float)

    for i, ti in enumerate(t_list):
        delta_days = (ti - t0).days #pre earthquake psd = 0
        if delta_days <= 0:
            f[i] = dA[i] = dTau[i] = 0.0 
            continue

        dt_years = delta_days / 365.25
        tau_eff  = tau if tau > 0.0 else 1e-12 #no division by 0
        
        #log model 
        if model_type == "log":
            # δlog = A * log(1 + dt/tau)
            g      = math.log(1.0 + dt_years / tau_eff)
            f[i]   = A * g
            dA[i]  = g
            dt_tau = dt_years / tau_eff
            dTau[i] = -A * dt_years / (tau_eff * tau_eff * (1.0 + dt_tau))
        
        #exp model
        elif model_type == "exp":
            # δexp = A * (1 - exp(-dt/tau))
            exp_term = math.exp(-dt_years / tau_eff)
            g        = 1.0 - exp_term
            f[i]     = A * g
            dA[i]    = g
            dTau[i]  = -A * dt_years * exp_term / (tau_eff * tau_eff)
        else:
            raise ValueError("model_type must be 'log' or 'exp' or 'auto'")

    return f, dA, dTau

def fit_joint_linear_psd(t_list, y, P_array, jump_list, angular_freq, eq_times,
    model_type="auto",           # "exp", "log", or "auto"
    tau_bounds=(0.01, 30.0),     # min yrs - max yrs iteration
    max_iter=20,                 # max iterations 
    tol=1e-6,                    # stop iteration when every param is below this value
    min_rel_improve=0.001,       # reject PSD if <value SSR improvement vs NO PSD
    verbose=True,
    A0_init=None,               
    tau0_init=None,           
):

    A0_loc  = A0_init    
    tau0_loc = tau0_init 
        
    y = np.asarray(y, float)
    P_array = np.asarray(P_array, float)
    w_sqrt = np.sqrt(P_array)
    n_obs = len(y)

    # linear and seasonal part (no PSD)
    A_lin = Design_Matrix(t_list, jump_list, angular_freq)
    m_lin = A_lin.shape[1]

    Aw = A_lin * w_sqrt[:, None]
    yw = y * w_sqrt
    theta_lin0, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    theta_lin0 = theta_lin0.flatten()

    def linear_seasonal():
        y_lin = A_lin.dot(theta_lin0)
        res_lin = y - y_lin
        ssr_lin = float(np.sum((res_lin * w_sqrt) ** 2))
        return theta_lin0, [], y_lin, y_lin, res_lin, res_lin, ssr_lin, ssr_lin

    def solve_psd_multi(term_types):
        if len(eq_times) == 0 or len(term_types) == 0:
            return linear_seasonal()

        n_eq = len(eq_times)
        n_terms = len(term_types)
        n_psd_params = 2 * n_eq * n_terms  # A, tau for each term and event
        
        psd_vec = np.zeros(n_psd_params, dtype=float)   #creating psd vector for jac
        for j in range(n_terms):
            for k in range(n_eq):
                idx = 2 * (j * n_eq + k)
                psd_vec[idx]     = A0_loc
                psd_vec[idx + 1] = tau0_loc

        param = np.concatenate([theta_lin0, psd_vec]) #connecting psd+linear_seasonal

        def model_and_jac(param_local):
            theta_lin = param_local[:m_lin]
            psd_local = param_local[m_lin:]

            # linear part
            y_lin_local = A_lin.dot(theta_lin) #linear+seasonal design matrix

            # PSD part
            y_psd = np.zeros(n_obs, dtype=float)    #total psd
            J_psd = np.zeros((n_obs, n_psd_params), dtype=float) #jac columns for psd (n_obs x n_psd_params)

            for j, mtype in enumerate(term_types):  #loop for term types
                for k, t0 in enumerate(eq_times):   #loop per earthquake
                    base_idx = 2 * (j * n_eq + k)
                    A_jk = psd_local[base_idx]      #amplitude
                    tau_jk = psd_local[base_idx + 1]#tau

                    f_k, dA_k, dTau_k = psd_term_and_derivatives(t_list, A_jk, tau_jk, t0, mtype)
                    f_k    = np.asarray(f_k, float)
                    dA_k   = np.asarray(dA_k, float)
                    dTau_k = np.asarray(dTau_k, float)

                    y_psd += f_k    
                    J_psd[:, base_idx]     = dA_k   
                    J_psd[:, base_idx + 1] = dTau_k

            y_model = y_lin_local + y_psd  #MODEL PREDICTION

            #shp Jacobian: (n_obs x (m_lin + n_psd_params)
            J_full = np.zeros((n_obs, m_lin + n_psd_params), dtype=float)
            J_full[:, :m_lin] = A_lin
            J_full[:, m_lin:] = J_psd

            return y_lin_local, y_psd, y_model, J_full

        # ITERATION
        for it in range(max_iter):
            y_lin_local, y_psd_local, y_model, J_full = model_and_jac(param) #computing Jacobian+model
            res = y - y_model #computing residuals
            
            #adding weights
            Jw = J_full * w_sqrt[:, None]
            rw = res * w_sqrt
            #N=J^T*W*J  g=J^TWr
            N = Jw.T @ Jw
            g = Jw.T @ rw

            detN = np.linalg.det(N)
            if abs(detN) < 1e-25:
                break

            delta = np.linalg.solve(N, g) #Δ= ((J^T*W*J)^-1)*(J^T*Wr)
            param_new = param + delta #new parameter vector

            # tau bounds
            tmin, tmax = tau_bounds
            for j in range(n_terms):
                for k in range(n_eq):
                    tau_idx = m_lin + 2 * (j * n_eq + k) + 1
                    tau_val = param_new[tau_idx]
                    if tau_val < tmin:
                        tau_val = tmin
                    if tau_val > tmax:
                        tau_val = tmax
                    param_new[tau_idx] = tau_val

            #iteration stop if the max update is smaller than the tolerance
            if np.linalg.norm(delta, ord=np.inf) < tol:
                param = param_new
                #if verbose:
                #    print(f"stopped at iter {it}")
                break

            param = param_new

        theta_lin = param[:m_lin]   #final lin+seasonal parameters
        psd_final = param[m_lin:]   #psd params
        y_lin_final, y_psd_final, y_full, _ = model_and_jac(param) #calling lin+seasonal, psd, full for comparisons

        #calculating residuals
        res_lin = y - y_lin_final
        res_full = y - y_full
        ssr_lin = float(np.sum((res_lin * w_sqrt) ** 2))
        ssr_full = float(np.sum((res_full * w_sqrt) ** 2))

        # build psd_params list
        psd_params = []
        for j, mtype in enumerate(term_types):
            for k, t0 in enumerate(eq_times):
                base_idx = 2 * (j * n_eq + k)
                A_jk = psd_final[base_idx]
                tau_jk = psd_final[base_idx + 1]
                psd_params.append({
                    "t0": t0,
                    "A": A_jk,
                    "tau": tau_jk,
                    "model_type": mtype,
                })

        if verbose:
            print(f"PSD iterations: {it+1}")

        
        return theta_lin, psd_params, y_lin_final, y_full, res_lin, res_full, ssr_lin, ssr_full

    # ____________________________________________________DECISION LOGIC OF PSD MODEL________________________________________
    base = linear_seasonal()
    ssr0 = base[7]

    if len(eq_times) == 0:
        if verbose and model_type == "auto":
            print("fit_joint_linear_psd(auto): no eq_times -> no PSD terms.")
        return base

    
    #if "exp" or "log" selected
    if model_type in ("exp", "log"):
        single = solve_psd_multi([model_type])
        ssr1 = single[7]
        
        rel_improve = (ssr0 - ssr1) / ssr0 if ssr0 > 0.0 else 0.0

        if verbose:
            print(f"{model_type.upper()} PSD: SSR={ssr1:.3e}, rel_improve={rel_improve:.3f}")

        # accept PSD only if it improves enough
        if (ssr1 < ssr0) and (rel_improve >= min_rel_improve):
            return single
        else:
            return base
            
    # AUTO MODE SAELECTED
    if model_type == "auto":

        # single-term EXP and single-term LOG
        exp1 = solve_psd_multi(["exp"])
        log1 = solve_psd_multi(["log"])
        ssr_exp = exp1[7]
        ssr_log = log1[7]

        #keeping the single-term with best SSR
        if ssr_exp <= ssr_log:
            best1_type, best1 = "exp", exp1
        else:
            best1_type, best1 = "log", log1
        ssr1 = best1[7]

        if ssr0 > 0.0:
            rel_improve1 = (ssr0 - ssr1) / ssr0
        else:
            rel_improve1 = 0.0
        if verbose:
            print("AUTO PSD (1 term): selected {}, SSR = {:.3e}, " " rel_improve1 = {:.6f}".format(best1_type, ssr1, rel_improve1))


        # Try second term (EXP+EXP vs LOG+LOG vs LOG+EXP disallow:EXP+LOG)
        candidates = []

        # 1-term as a candidate
        candidates.append(("1-term-" + best1_type, [best1_type], best1, ssr1))

        # EXP+EXP
        exp_exp = solve_psd_multi(["exp", "exp"])
        candidates.append(( "exp+exp", ["exp", "exp"], exp_exp, exp_exp[7]))

        # LOG+LOG
        log_log = solve_psd_multi(["log", "log"])
        candidates.append(( "log+log", ["log", "log"], log_log, log_log[7]))

        # LOG+EXP 
        log_exp = solve_psd_multi(["log", "exp"])
        candidates.append(( "log+exp", ["log", "exp"], log_exp, log_exp[7]))

        # pick candidate with smallest SSR
        name_best2, types_best2, sol_best2, ssr_best2 = min(candidates, key=lambda x: x[3])

        # improvement of best PSD model vs 1-term
        if ssr1 > 0.0:
            rel_improve2 = (ssr1 - ssr_best2) / ssr1
        else:
            rel_improve2 = 0.0

        if verbose:
            print( "PSD (final): model = {}, SSR = {:.3e}, " "rel_improve2_vs_1term = {:.6f}".format( name_best2, ssr_best2, rel_improve2))
    
        
        # If the best PSD model is worse than linear+seasonal, select linear+seasonal.
        # comment next 7 lines to force psd to happen when there is an earthquake
        if ssr_best2 >= ssr0:
            if verbose:
                print("Final check: PSD discarded (SSR_with_PSD >= SSR_no_PSD); ""keeping linear+seasonal-only model.")
            return base
        else:
            if verbose:
                print("Final check: PSD kept (SSR_with_PSD < SSR_no_PSD).")
            return sol_best2


def compute_psd_series_joint(t_list, psd_params):
    if not psd_params:
        return np.zeros(len(t_list), dtype=float) #no psd 

    total = np.zeros(len(t_list), dtype=float) #arxikopoisi
    for term in psd_params:
        f_k, _, _ = psd_term_and_derivatives( t_list, term["A"], term["tau"], term["t0"], term["model_type"])
        total += np.asarray(f_k, float)
    return total

def compute_psd_covariance( t_list, P_array, jump_list, angular_freq, psd_params,SIGMA0):
    t_list = list(t_list)
    P_array = np.asarray(P_array, float)
    w_sqrt = np.sqrt(P_array)

    # design matrix for linear+seasonal part
    A_lin = Design_Matrix(t_list, jump_list, angular_freq)
    n_obs, m_lin = A_lin.shape

    # PSD part
    n_terms = len(psd_params)
    n_psd_params = 2 * n_terms
    J_psd = np.zeros((n_obs, n_psd_params), dtype=float)

    for p, term in enumerate(psd_params):
        A_jk   = term["A"]
        tau_jk = term["tau"]
        t0     = term["t0"]
        mtype  = term["model_type"] 

        # derivatives of a and tau
        _, dA_k, dTau_k = psd_term_and_derivatives(t_list, A_jk, tau_jk, t0, mtype)
        dA_k   = np.asarray(dA_k, float)
        dTau_k = np.asarray(dTau_k, float)

        # jacobian psd terms order [A1​,τ1​,A2​,τ2​,...]
        J_psd[:, 2 * p]     = dA_k
        J_psd[:, 2 * p + 1] = dTau_k

    # full Jacobian = [ linear+seasonal+PSD ]
    J_full = np.hstack((A_lin, J_psd))

    # weighted normal matrix
    Jw = J_full * w_sqrt[:, None]
    N  = Jw.T @ Jw
    sigma0_sq = float(SIGMA0) ** 2
    cov_param = sigma0_sq * np.linalg.inv(N)

    # split linear+seasonal and psd for comparisons
    cov_lin = cov_param[:m_lin, :m_lin]
    cov_psd = cov_param[m_lin:, m_lin:]

    # variance of PSD correction δL(t)
    var_psd = np.zeros(n_obs, dtype=float)
    for i in range(n_obs):
        c_i = J_psd[i, :]
        var_psd[i] = float(c_i @ cov_psd @ c_i)

    return cov_lin, cov_psd, var_psd
    
#____________________________________________________Time Handling__________________________________________________
# List of every date between t1(start) and t2(end) with daily interval
def date_calc(t):
    t1, t2 = t
    dates = []
    i = 0
    while t1 <= t2 and i < 1e-5:
        dates.append(t1)
        t1 += datetime.timedelta(days = 1)
    return dates


#Calculating middle epoch in datetime
def middle_epoch(t):
    t1, t2 = t[0], t[-1]
    t0 = t1 + (t2 - t1)/2
    return t0

# For a time instance (middle epoch) calculate the time difference in fractional years between each epoch and the middle epoch 
# Converting the datetime objects to fractional years 
def fractionaldt(t,t0=None):
    if not t0:
        t0 = middle_epoch(t)
    dt = []
    for epoch in t:
        dt.append( (epoch-t0).days / 365.25 )
    return dt

def decimal2date(decimal_year):
    year = int(decimal_year)
    frac = decimal_year - year
    leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_in_year = 366 if leap else 365
    return dt(year, 1, 1) + timedelta(days=frac * days_in_year)

# ____________________________________________________Parser for .cts files________________________  using pandas
def parse_cts(PATH):
    columns = [ "odate", "otime", "X", "sX", "Y", "sY", "Z", "sZ", "lat", "slat", "lon", "slon", "alt", "salt", "pdate", "ptime", "com"]

    data = []
    with open(PATH, "r") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if i == 0:         #skip header
                continue

            parts = s.split()
            if len(parts) < len(columns):
                continue

            data.append(parts[:len(columns)])

    df = pd.DataFrame(data, columns=columns)

    # numeric columns
    numeric_cols = [ "X", "sX", "Y", "sY", "Z", "sZ", "lat", "slat", "lon", "slon", "alt", "salt"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # build times
    df["tv"] = pd.to_datetime(df["odate"] + " " + df["otime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["stime"] = pd.to_datetime(df["pdate"] + " " + df["ptime"], errors="coerce")

    df = df.sort_values(["tv", "stime"])

    return df
    
def _cts_df_extract(df):
    t = df["tv"].tolist()
    stime = df["stime"].tolist()
    coords = []
    
    for _, row in df.iterrows():
        c = [0.0] * 12
        c[0]  = float(row["X"])    if not pd.isna(row["X"])    else 0.0
        c[1]  = float(row["sX"])   if not pd.isna(row["sX"])   else 0.0
        c[2]  = float(row["Y"])    if not pd.isna(row["Y"])    else 0.0
        c[3]  = float(row["sY"])   if not pd.isna(row["sY"])   else 0.0
        c[4]  = float(row["Z"])    if not pd.isna(row["Z"])    else 0.0
        c[5]  = float(row["sZ"])   if not pd.isna(row["sZ"])   else 0.0
        c[6]  = float(row["lat"])  if not pd.isna(row["lat"])  else 0.0
        c[7]  = float(row["slat"]) if not pd.isna(row["slat"]) else 0.0
        c[8]  = float(row["lon"])  if not pd.isna(row["lon"])  else 0.0
        c[9]  = float(row["slon"]) if not pd.isna(row["slon"]) else 0.0
        c[10] = float(row["alt"])  if not pd.isna(row["alt"])  else 0.0
        c[11] = float(row["salt"]) if not pd.isna(row["salt"]) else 0.0
        coords.append(c)

    return t, stime, coords
# ____________________________________________________Parser for .txyz2 files_______________________  using pandas

def parse_txyz2(path):
    columns = [ "station_name", "date", "decimal_year", "x", "y", "z", "x_sigma", "y_sigma", "z_sigma",
        "xy_corr", "yz_corr", "xz_corr", "antenna_height"]
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 13:
                data.append(parts)

    df = pd.DataFrame(data, columns=columns)
    n_cols = columns[2:]
    df[n_cols] = df[n_cols].apply(pd.to_numeric, errors='coerce')

    df["date & time"] = df["decimal_year"].apply(decimal2date)
    return df

def _txyz2_df_extract(df):
    t = df["date & time"].tolist()
    stime = t[:] 
    coords = []
    for _, row in df.iterrows():
        c = [0.0]*12
        c[0] = float(row["x"]) if not pd.isna(row["x"]) else 0.0
        c[1] = float(row["x_sigma"]) if not pd.isna(row["x_sigma"]) else 0.0
        c[2] = float(row["y"]) if not pd.isna(row["y"]) else 0.0
        c[3] = float(row["y_sigma"]) if not pd.isna(row["y_sigma"]) else 0.0
        c[4] = float(row["z"]) if not pd.isna(row["z"]) else 0.0
        c[5] = float(row["z_sigma"]) if not pd.isna(row["z_sigma"]) else 0.0
    
        # ENU sigmas for weighting 
        e_sig = row["e_sigma"] if "e_sigma" in df.columns else np.nan
        n_sig = row["n_sigma"] if "n_sigma" in df.columns else np.nan
        u_sig = row["u_sigma"] if "u_sigma" in df.columns else np.nan

        c[7]  = float(e_sig) if not pd.isna(e_sig) else c[1]  # σE
        c[9]  = float(n_sig) if not pd.isna(n_sig) else c[3]  # σN
        c[11] = float(u_sig) if not pd.isna(u_sig) else c[5]  # σU
        coords.append(c)
    stime_sort = [x for _, x in sorted(zip(t, stime))]
    coords_sort = [x for _, x in sorted(zip(t, coords))]
    t_sort = sorted(t)
    return t_sort, stime_sort, coords_sort

def parse_tenv(path):
    columns = [ "station_id", "date", "decimal_year", "mjd", "gps_week", "dow", "de", "dn", "dv","antenna_height",
        "sigma_e", "sigma_n", "sigma_v", "corr_en", "corr_ev", "corr_nv"]

    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            # skip header and short lines
            if len(parts) < len(columns):
                continue
            if not parts[1].isdigit() or len(parts[1]) != 8:
                continue
            data.append(parts[:len(columns)])

    df = pd.DataFrame(data, columns=columns)

    num_cols = ["decimal_year", "mjd", "gps_week", "dow", "de", "dn", "dv", "antenna_height", "sigma_e", "sigma_n", "sigma_v",
                "corr_en", "corr_ev", "corr_nv"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # same time base as txyz2 from decimal year
    df["date & time"] = df["decimal_year"].apply(decimal2date)

    return df

def parse_txyz2_with_tenv(path):
    df = parse_txyz2(path)

    tenv_path = Path(path).with_suffix(".tenv")
    if tenv_path.is_file():
        df_tenv = parse_tenv(str(tenv_path))

        # merge on datetime
        df = df.merge(
            df_tenv[["date & time", "sigma_e", "sigma_n", "sigma_v"]],
            on="date & time",
            how="left"
        )

        df["e_sigma"] = df["sigma_e"]
        df["n_sigma"] = df["sigma_n"]
        df["u_sigma"] = df["sigma_v"]

    return df

# ____________________________________________________Parser for .pos files_______________________  using pandas

def parse_pos(PATH):
    columns = ["date", "time", "mjd","X", "Y", "Z","Sx", "Sy", "Sz","Rxy", "Rxz", "Ryz","NLat", "Elong", "Height",
        "dN", "dE", "dU","Sn", "Se", "Su","Rne", "Rnu", "Reu","Soln"]

    data = []
    with open(PATH, "r") as f:
        for i, line in enumerate(f):
            # start reading data at line 37 
            if i < 36:
                continue
            parts = line.split()
            if len(parts) < len(columns):
                continue
            data.append(parts[:len(columns)])

    df = pd.DataFrame(data, columns=columns)

    # numeric columns (everything except date, time, Soln)
    numeric_cols = [c for c in columns if c not in ("date", "time", "Soln")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # observation datetime
    df["tv"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y%m%d %H%M%S",
        errors="coerce"
    )
    # pos has no separate solution time -> use tv
    df["stime"] = df["tv"]

    df = df.rename(columns={
        "Sx": "sX",
        "Sy": "sY",
        "Sz": "sZ",
        "NLat": "lat",
        "Elong": "lon",
        "Height": "alt"
    })

    df["slat"] = df["Sn"]
    df["slon"] = df["Se"]
    df["salt"] = df["Su"]

    df = df.dropna(subset=["tv"])
    df = df.sort_values("tv")

    return df


def _pos_df_extract(df):
    t = df["tv"].tolist()
    stime = df["stime"].tolist() 

    coords = []
    for _, row in df.iterrows():
        c = [0.0] * 12
        c[0]  = float(row["X"])    if not pd.isna(row["X"])    else 0.0
        c[1]  = float(row["sX"])   if not pd.isna(row["sX"])   else 0.0
        c[2]  = float(row["Y"])    if not pd.isna(row["Y"])    else 0.0
        c[3]  = float(row["sY"])   if not pd.isna(row["sY"])   else 0.0
        c[4]  = float(row["Z"])    if not pd.isna(row["Z"])    else 0.0
        c[5]  = float(row["sZ"])   if not pd.isna(row["sZ"])   else 0.0
        c[6]  = float(row["lat"])  if not pd.isna(row["lat"])  else 0.0
        c[7]  = float(row["slat"]) if not pd.isna(row["slat"]) else 0.0
        c[8]  = float(row["lon"])  if not pd.isna(row["lon"])  else 0.0
        c[9]  = float(row["slon"]) if not pd.isna(row["slon"]) else 0.0
        c[10] = float(row["alt"])  if not pd.isna(row["alt"])  else 0.0
        c[11] = float(row["salt"]) if not pd.isna(row["salt"]) else 0.0
        coords.append(c)

    stime_sort  = [x for _, x in sorted(zip(t, stime))]
    coords_sort = [x for _, x in sorted(zip(t, coords))]
    t_sort      = sorted(t)

    return t_sort, stime_sort, coords_sort


# ____________________________________________________File selector_______________________________________________    
def parse_auto(PATH):
    p = Path(PATH)
    ext = p.suffix.lower()

    if ext == ".cts":
        df = parse_cts(PATH)
        t, stime, coords = _cts_df_extract(df)
        src_kind = "cts"

    elif ext == ".txyz2":
        df = parse_txyz2_with_tenv(PATH)
        t, stime, coords = _txyz2_df_extract(df)
        src_kind = "txyz2"

    elif ext == ".pos":
        df = parse_pos(PATH)
        t, stime, coords = _pos_df_extract(df)
        src_kind = "pos"

    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .cts, .txyz2, or .pos")

    return (t, stime, coords), src_kind

#_______________________________________________________Coords & errors___________________________________________
#get (σeast, σnorth, σup)
def geterrors(coordinates):
    sf, sl, sh = [], [], []
    for flh_elements in coordinates:
        sf.append(flh_elements[7])
        sl.append(flh_elements[9])
        sh.append(flh_elements[11])
    return sf, sl, sh


#get x,y,x
def getxyz(coordinates):
    x, y, z = [], [], []
    for xyz_elements in coordinates:
        x.append(xyz_elements[0])
        y.append(xyz_elements[2])
        z.append(xyz_elements[4])
    return x, y, z


#get average x,y,z
def mean_xyz(x, y, z):
    xm, ym, zm = map(statistics.mean, [x, y, z])
    return xm, ym, zm

#____________________________________________________Coordinate Conversion_________________________________________
#converting Earth Centered, Earth Fixed (ECEF) x,y,z coordinates in meters to Latitude, Longitude, height using ellipsoid WGS84 
def xyz2latlon(x, y, z):

    lon = math.atan2(y, x)
    S = math.sqrt(x*x + y*y)
    lat_old = math.atan( z / ((1 - EPSILON) * S) )
    div = math.sqrt(1 - EPSILON*(math.sin(lat_old)*math.sin(lat_old)))
    Ni = (6378137/div)

    j = 0
    dx = 1 
    lat_new = 1

    while dx > 1e-12 and j < 100:
        lat_new = math.atan((z + 0.006694380*Ni*math.sin(lat_old))/S)
        dx = abs(lat_old - lat_new)
        lat_old = lat_new
        j = j + 1
    return lat_new, lon

#rotation matrix
def rotation_matrix(phi, lamda):
    a = math.sin(phi)
    b = math.sin(lamda)
    c = math.cos(phi)
    d = math.cos(lamda)

    r = [ 
        [-1*b, d, 0],
        [-1*a*d, -1*a*b, c],
        [c*d , c*b , a]
        ]
    return r

#conversion to topocentric
def topocentric_conversion(x, y, z):
    xm, ym, zm = mean_xyz(x, y, z)
    lat, lon = xyz2latlon(xm, ym, zm)
    r_matrix = rotation_matrix(lat, lon)
    e, n, u = [], [], []
    for xyz in zip(x, y, z):
        difxyz =  [xyz[0] - xm, xyz[1] - ym, xyz[2] - zm]
        i=0; e.append(difxyz[0] * r_matrix[i][0] + difxyz[1] * r_matrix[i][1] + difxyz[2] * r_matrix[i][2])
        i=1; n.append(difxyz[0] * r_matrix[i][0] + difxyz[1] * r_matrix[i][1] + difxyz[2] * r_matrix[i][2])
        i=2; u.append(difxyz[0] * r_matrix[i][0] + difxyz[1] * r_matrix[i][1] + difxyz[2] * r_matrix[i][2])
    return e, n, u

#____________________________________________________Least Squares Solution______________________________________________
""" Generating a design matrix based on the given time differences (each time instance (epoch) - the middle epoch), and 2 or more given angular frequencies (omega1,2):
  The resulting design matrix named "A" has dimensions "(len(dates), number of coefficients(6))"
  The equation being modeled is: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)..."""


# weight calculation σ/sigma
def weights_calc(coord_errors, sigma0):
    weights = [(1 / ce ) * sigma0 for ce in coord_errors]
    weights_array = np.array(weights)
    #P_matrix = np.diag(weights_array)
    #print(P_matrix)
    return weights_array #P_matrix

# design matrix calculation
def Design_Matrix(t, jump_list, angular_freq):
    A = []
    m = 2 + len(angular_freq) * 2 + len(jump_list)               #Number of parameters 
    dt = fractionaldt(t)   
    for d in zip(t, dt):
        A_row = []                                               #Creating a row containing the values of the independent variables for each date
        A_row += [d[1], 1]                                       #Adding the linear terms
        for w in angular_freq:
            A_row += [math.sin(w * d[1]), math.cos(w * d[1])]    #Adding as many harmonic terms as in the omega list
        for jump_t in jump_list:
            if d[0] >= jump_t:
                A_row += [1]
            else:
                A_row += [0]
        assert(len(A_row) == m)
        A.append(A_row)                                          #Append the constructed row to the Design Matrix (list named "A")
    A = np.array(A)
    return A

#_______________________________________________________Linear Regression____________________________________________________
""" Solving a linear regression problem using the least square method
 Converting the "y" input list to a NumPy array and reshape it into a column vector ensuring 
 that "y" is in the correct format for further computations """
 
 #model fitting
def fit(y, t, P_array , freq, jumps, SIGMA0):                    # t = dates in datetime, y = parsed solutions, freq = given angular frequencies, jumps = list of datetime objects (time instance when the jump happened)
    #P_array =  np.ones(len(y))
    #model = np.reshape(y, (len(y),1))
    model = y * np.sqrt(P_array)
    A = Design_Matrix(t, jumps, freq)
    AP = A * np.sqrt(P_array)[:, None]
    dx, sumres, _,  _ = np.linalg.lstsq(AP, model, rcond=None)
    dx = dx.flatten()
    AT = np.transpose(AP)
    cov_matrix = np.linalg.inv(AT@AP)*SIGMA0**2
    return dx, cov_matrix, sumres/(len(y)-1)

#calculating residuals
def residuals(y, y_predicted):
    residuals = [actual - pred for actual, pred in zip(y, y_predicted)]
    squared_residuals = [residual ** 2 for residual in residuals]
    sum_sq_res = sum(squared_residuals)
    mse = sum_sq_res / (len(y) - 1)                              #mean squared error
    return residuals, mse

#outliers removal
def remove_outliers(y, t, P_array, residuals):
    r = np.asarray(residuals, dtype=float)
    n = len(r)
    
    Win = 365.25            # !!!!!!!!! Window in days
    t_arr = np.array(t)     # list of datetime
    
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        ti = t_arr[i]

        # indices of epochs within ±0.5 year of ti
        idx = [
            j for j in range(n)
            if not np.isnan(r[j]) and abs((t_arr[j] - ti).days) <= Win / 2.0
        ]

        if len(idx) < 5:
            continue

        window = r[idx]
        window = window[~np.isnan(window)] #remove NaN
        med = np.median(window)
        q25, q75 = np.percentile(window, [25, 75])
        iqr = q75 - q25

        # r-med>3iqr
        if iqr > 0 and abs(r[i] - med) > 3.0 * iqr:
            outlier_mask[i] = True

    # keep only non-outliers
    yy = [val for val, flag in zip(y, outlier_mask) if not flag]
    tt = [val for val, flag in zip(t, outlier_mask) if not flag]
    pp = [val for val, flag in zip(P_array, outlier_mask) if not flag]

    return yy, tt, np.array(pp)


#____________________________________________________Compute Model____________________________________________
""" Given the least square solutions for an equation: 
y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates) + ... + j1 + ...jν
For each date, calculate the corresponding solution creating the final model """

def compute_model(dx, t, freq, jumps, white_noise=None):
    y = []   
    dt = fractionaldt(t)   
    
    for d in zip(t, dt):
        # linear part
        yi = dx[0]*d[1] + dx[1]
        # harmonic part
        offset = 2
        for i, fr in enumerate(freq):
            yi += dx[i*2+2] * math.sin(fr * d[1]) + dx[i*2+3] * math.cos(fr * d[1])
        # add jumps
        offset = 2 + len(freq)*2
        for i, date in enumerate(jumps):
            if d[0] >= date:
               yi += dx[offset + i]
        # add white noise
        if white_noise is not None:
            yi += np.random.normal(white_noise[0], white_noise[1])
        # append to returned list
        y.append(yi)
    return y

#____________________________________________________Offset Detector________________________________________________
#PELT algorithm for auto-jump recognistion

def detect_earthquake_epochs(t, e, n, u):
    # PELT per axis 
    arr_e = np.asarray(e).reshape(-1, 1)
    arr_n = np.asarray(n).reshape(-1, 1)
    arr_u = np.asarray(u).reshape(-1, 1)

    # breakpoints per axis (+model parameters)
    #model: rbf/l2/linear, higher pen--> more strict, higher min_size --> shorter segments, higher jump --> less  precise breakpoints but faster
    bkps_e = rpt.Pelt(model="rbf", min_size=10, jump=3).fit(arr_e).predict(pen=30) 
    bkps_n = rpt.Pelt(model="rbf", min_size=10, jump=3).fit(arr_n).predict(pen=30)
    bkps_u = rpt.Pelt(model="rbf", min_size=10, jump=3).fit(arr_u).predict(pen=30)

    # merge breakpoints across components
    signal = np.column_stack((e, n, u))
    N = len(signal)
    
    all_bkps = [bkps_e, bkps_n, bkps_u]
    tol = 4      # merge nearby detection to one event (decrease it for bigger precision)
    comp = 2     # 1 for per axis (better offsets detection), 2 or 3 componets agree (better auto seismic detection) !!!!!!!!!!!!!!!!

    # candidates =breakpoints detected
    candidates = sorted(set(i for bkps in all_bkps for i in bkps if i < N))
    merged_idx = []

    # loop over candidates to check if they agree
    for c in candidates:
        count = 0
        # if they are inside tol consider them the same event
        cluster = [c]
        for bkps in all_bkps:
            near = [b for b in bkps if b < N and abs(b - c) <= tol]
            if near:
                count += 1
                cluster.extend(near)

        # check if comp agree 
        if count >= comp:
            idx = int(round(np.median(cluster)))
            if 0 < idx < N:
                merged_idx.append(idx)

    # convert indices to times
    earthquake_list = []
    for idx in sorted(set(merged_idx)):
        i = min(idx, len(t) - 1)
        earthquake_list.append(t[i])

#_____________________________________________________CSV handling___________________________________________
    station = Path(PATH).stem[:4] #station name extraction

    # CSV path: same folder as the input file
    name = Path(PATH).with_suffix("")          # remove extension
    csv_path = str(name) + "_pelt_jumps.csv"   # /XXXX_pelt_jumps.csv

    rows = []

    # rows per component
    for ts in earthquake_list:
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        for comp in ["E", "N", "U"]:
            rows.append({
                "timestamp": ts_str,
                "station": station,
                "component": comp,
                "jump_displacement": 0.0,  # always 0 
                "label": "",               # user type 'earthquake' here if needed
                "comment": "",             # optional comment
            })

    df_jumps = pd.DataFrame(rows, columns=[ "timestamp", "station", "component", "jump_displacement","label", "comment"],)
    df_jumps.to_csv(csv_path, index=False)
    return earthquake_list, csv_path
    
def open_file_default(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

#____________________________________________________Main Function__________________________________________________
if __name__ == "__main__":
    #dates = date_calc(timeframe)
    (t, stime, coords), src_kind = parse_auto(PATH)
    x, y, z = getxyz(coords)
    sof, sol, soh = geterrors(coords)

    e, n, u = topocentric_conversion(x, y, z)
    na = n
    ea = e
    ua = u  
    ta, tn, te, th = t, t, t, t

    sof, sol, soh = geterrors(coords)
    PN_array = weights_calc(sof, SIGMA0)
    PE_array = weights_calc(sol, SIGMA0)
    PU_array = weights_calc(soh, SIGMA0)

#____________________________________________________Earthquake epoch recognition_______________________________
    station = Path(PATH).stem[:4]
    base = Path(PATH).with_suffix("")            
    csv_path = str(base) + "_pelt_jumps.csv"

    eq_times  = []   # list for earthquakes used for PSD
    jump_list = []   # list of all jumps used as offsets in the linear+seasonal model


    if os.path.isfile(csv_path):
        print(f"Using existing CSV, skipping PELT: {csv_path}")
        df_labels = pd.read_csv(csv_path)

        ts_all = pd.to_datetime(df_labels["timestamp"], errors="coerce").dropna()
        jump_list = sorted(set(ts_all.tolist()))

        # label 'eq' 
        mask_eq = df_labels["label"].astype(str).str.contains(r"\beq\b", case=False, na=False)
        if mask_eq.any():
            ts_eq = pd.to_datetime(df_labels.loc[mask_eq, "timestamp"], errors="coerce").dropna()
            eq_times = sorted(set(ts_eq.tolist()))
            
            print("Earthquake epochs for psd:")
            for ts in eq_times:
                print("   ", ts)
        else:
            print("Proceeding without PSD (linear+seasonal model).")
            
            
    else:
        # run PELT algorithm 
        all_jumps, csv_path = detect_earthquake_epochs(t, e, n, u)  
  
        if all_jumps:
            print("Opening CSV with jumps. Add 'eq' in the label column if earthquake detected")
            open_file_default(csv_path)
            input("After editing and saving the CSV, press Enter to continue")

            df_labels = pd.read_csv(csv_path)
            # All jumps from CSV --> used as offsets
            ts_all = pd.to_datetime(df_labels["timestamp"])
            jump_list = sorted(set(ts_all.tolist()))

            # Jumps labeled 'earthquake' --> used for PSD
            mask_eq = df_labels["label"].astype(str).str.contains(r"\beq\b", case=False, na=False)
            if mask_eq.any():
                ts_eq = pd.to_datetime(df_labels.loc[mask_eq, "timestamp"])
                eq_times = sorted(set(ts_eq.tolist()))

                print("Earthquake epochs for psd:")
                for ts in eq_times:
                    print("   ", ts)
            else:
                print("Proceeding without PSD (linear+seasonal model).")
        else:
            print("No jumps found by PELT. Proceeding without PSD and offsets (linear+seasonal only).")
    
    #saving terminal prints        
    report_path = str(Path(PATH).with_name(f"{station}_report.txt"))
    report_file = open(report_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, report_file)
    sys.stderr = Tee(sys.__stderr__, report_file)

    print(f"Writing report to: {report_path}")

#____________________________________________________linear+seasonal model ___________________________________________________________
    #function for timeseries analysis per component
   
    def analyze_axis(label, y, t, P_array, angular_freq, jump_list, eq_times, sigma0_init=0.001, mse_tol=1.0e-6, max_outlier_iter=10,
        psd_model_type="auto",  tau_bounds=(0.01, 10.0), psd_max_iter=30, psd_tol=1.0e-6, psd_min_rel_improve=0.05, verbose_psd=True,
        psd_A0_init=None, psd_tau0_init=None):

        # linear + seasonal model
        j = 0                   # iteration counter           
        mse = 1.0               # current mean squared error
        mset = 1000.0           # pre mean squared error
        SIGMA0_loc =sigma0_init # 

        # loop that stops when max iter is hit or mse stops changing by more than mse tolerance
        while abs(mse - mset) > mse_tol and j < max_outlier_iter:
            if j != 0:
                mset = mse
            #dx: (trend +intercept+sin/cos terms+jumps), vx: var of dx
            dx, vx, mse = fit(y, t, P_array, angular_freq, jump_list, SIGMA0_loc)
            y_model = compute_model(dx, t, angular_freq, jump_list)
            res, _ = residuals(y, y_model) #residuals (observed - modeled)

            mse = float(np.mean(mse))
            SIGMA0_loc = math.sqrt(mse)

            # outlier removal
            y, t, P_array = remove_outliers(y, t, P_array, res)
            j += 1

        # linear+seasonal fit after outlier removal
        dx, vx, mse = fit(y, t, P_array, angular_freq, jump_list, SIGMA0_loc)
        y_model = compute_model(dx, t, angular_freq, jump_list)
        res, _ = residuals(y, y_model)

        print(f"{label} Solutions (linear+seasonal):")

        names = ["v (speed)", "a0 (intercept)"]

        # sin / cos seasonal terms
        for k, w in enumerate(angular_freq, start=1):
            if abs(w - 2 * math.pi * 1.0) < 1e-12:
                nm = "annual"
            elif abs(w - 2 * math.pi * 2.0) < 1e-12:
                nm = "semiannual"
            else:
                nm = f"w={w:.6g}"
            names += [f"A{k} (sin {nm})", f"B{k} (cos {nm})"]

        # step offsets (jumps)
        for j, jt in enumerate(jump_list, start=1):
            try:
                d = jt.date()
            except Exception:
                d = jt
            names.append(f"J{j} (step @ {d})")

        for i, x in enumerate(dx):
            sx = math.sqrt(vx[i][i])

            if i == 0:
                print("{:<35s} {:+14.3e} m/yr +- {:14.3e} m/yr".format( names[i], x, sx))
            else:
                print("{:<35s} {:+14.3e} m    +- {:14.3e} m".format( names[i], x, sx))
        print()
        print()

        # WHEN NO EARTHQUAKES ADDED 
        theta_psd   = dx            # same parameters as linear+seasonal
        psd_terms   = []            # no PSD terms
        y_lin_psd   = y_model       # linear model
        res_lin_psd = res           # residuals of linear model

        # linear SSR calculation
        w_sqrt = np.sqrt(P_array)
        ssr_lin_psd = float(np.sum((res_lin_psd * w_sqrt) ** 2))

        # PSD CORRECTION AND FULL lin+seasonal+psd model
        psd_corr = compute_psd_series_joint(t, psd_terms)  # zeros if no psd_terms
        full_model = y_lin_psd + psd_corr                  # = y_mod if no PSD
        res_full_psd = y - full_model
        ssr_full_psd = float(np.sum((res_full_psd * w_sqrt) ** 2))

#___________________________________________________linear+seasonal+psd model__________________________________________________________________
        # WHEN EARTHQUAKES ADDED in eq_times list
        if eq_times:
            # fitting model with psd params
            theta_psd_tmp, psd_terms_tmp, y_lin_tmp, y_full_tmp, res_lin_tmp, res_full_tmp, ssr_lin_tmp, ssr_full_tmp = \
                fit_joint_linear_psd( t_list=t, y=y, P_array=P_array, jump_list=jump_list, angular_freq=angular_freq, eq_times=eq_times, model_type=psd_model_type,
                    tau_bounds=tau_bounds, max_iter=psd_max_iter, tol=psd_tol, min_rel_improve=psd_min_rel_improve, verbose=verbose_psd, A0_init=psd_A0_init,tau0_init=psd_tau0_init)

            # if PSD gets accepted 
            if psd_terms_tmp:
                theta_psd   = theta_psd_tmp
                psd_terms   = psd_terms_tmp
                y_lin_psd   = y_lin_tmp
                res_lin_psd = res_lin_tmp
                ssr_lin_psd = ssr_lin_tmp

                # recompute correction and full model from final psd_terms
                psd_corr = compute_psd_series_joint(t, psd_terms)
                full_model = y_lin_psd + psd_corr
                res_full_psd = y - full_model
                ssr_full_psd = float(np.sum((res_full_psd * w_sqrt) ** 2))

                # covariance of joint linear+seasonal+PSD parameters
                cov_lin_psd, cov_psd, var_psd = compute_psd_covariance( t_list=t, P_array=P_array, jump_list=jump_list, angular_freq=angular_freq, psd_params=psd_terms, SIGMA0=SIGMA0)

                # print dx joint solution and PSD terms
                print(f"{label} Solutions (linear+seasonal+PSD):")

                names = ["v (speed)", "a0 (intercept)"]

                for k, w in enumerate(angular_freq, start=1):
                    if abs(w - 2 * math.pi * 1.0) < 1e-12:
                        nm = "annual"
                    elif abs(w - 2 * math.pi * 2.0) < 1e-12:
                        nm = "semiannual"
                    else:
                        nm = f"w={w:.6g}"
                    names += [f"A{k} (sin {nm})", f"B{k} (cos {nm})"]

                for j, jt in enumerate(jump_list, start=1):
                    try:
                        d = jt.date()
                    except Exception:
                        d = jt
                    names.append(f"J{j} (step @ {d})")

                # linear + seasonal + steps
                for i, x in enumerate(theta_psd):
                    sx = math.sqrt(cov_lin_psd[i, i])

                    if i == 0:
                        print("{:<35s} {:+14.3e} m/yr +- {:14.3e} m/yr".format( names[i], x, sx))
                    else:
                        print("{:<35s} {:+14.3e} m    +- {:14.3e} m".format( names[i], x, sx))

                # append PSD parameters
                if psd_terms:
                    for p, term in enumerate(psd_terms):
                        A   = term["A"]
                        tau = term["tau"]
                        t0  = term["t0"]
                        m   = term["model_type"]

                        try:
                            t0s = t0.date()
                        except Exception:
                            t0s = t0

                        sA   = math.sqrt(cov_psd[2*p,     2*p    ])
                        sTau = math.sqrt(cov_psd[2*p + 1, 2*p + 1])

                        print("{:<35s} {:+14.3e} m +- {:14.3e} m".format( f"PSD A ({m}, t0={t0s})", A, sA))
                        print("{:<35s} {:14.3e} yr +- {:14.3e} yr".format( f"PSD tau ({m}, t0={t0s})",tau,sTau))

                print()


                #print each psd term results
                print(f"\n{label} PSD terms:")
                for k, term in enumerate(psd_terms):
                    sA   = math.sqrt(cov_psd[2 * k,     2 * k])
                    sTau = math.sqrt(cov_psd[2 * k + 1, 2 * k + 1])
                    print(
                        "  {model}  A={A:.4e} ± {sA:.4e},  "
                        "tau={tau:.3f} ± {sTau:.3f} yr  t0={t0}".format(
                            model=term["model_type"].upper(),
                            A=term["A"],   sA=sA,
                            tau=term["tau"], sTau=sTau,
                            t0=term["t0"],
                        )
                    )
                print(f"{label} SSR no PSD   = {ssr_lin_psd:.3e}")
                print(f"{label} SSR with PSD = {ssr_full_psd:.3e}")
                print()

        return ( y, t, P_array, dx, vx, y_model, res, theta_psd, psd_terms, y_lin_psd, full_model, res_lin_psd, res_full_psd, ssr_lin_psd, ssr_full_psd)

    #____________________________________________________ ANALYSIS PER AXIS ________________________________________

    #______ North
    (n, tn, PN_array, dxN, vxN, N_lin_robust, resN_robust, thetaN, psdN, N_lin, N_psd, resN_lin, resN_full, ssrN_lin, ssrN_full)= analyze_axis(
        label="North",              # component label 'North'
        y=n,                        # North displacement (m)
        t=tn,                       # North time list
        P_array=PN_array,           # North weights matrix
        angular_freq=angular_freq,  # Angular frequencies (annual, semi-annual)
        jump_list=jump_list,        # offsets epochs (all jumps)     
        eq_times=eq_times,          # offsets of earthquakes
        sigma0_init=0.001,          # initial a priori error
        mse_tol=1.0e-6,             # tolerance for iteration   
        max_outlier_iter=10,        # max oulier removal loops
        psd_model_type="auto",      # auto (or exp or log)
        tau_bounds=(0.01, 50.0),    # min and max years of relaxation
        psd_max_iter=50,            # max iteration
        psd_tol=1.0e-6,             # tolerance for psd iterations
        psd_min_rel_improve=0.01,   # min relative improvement of psd vs linear+seasonal to accept psd
        verbose_psd=True,           # print or not psd process
        psd_A0_init=A0_N,           # psd Amplitude initial value
        psd_tau0_init=tau0_N,       # psd Relaxation time initial value
    )

    #______ East
    (e, te, PE_array,dxE, vxE, E_lin_robust, resE_robust,thetaE, psdE, E_lin, E_psd,resE_lin, resE_full, ssrE_lin, ssrE_full) = analyze_axis(
        label="East",               # component label 'East'
        y=e,                        # East displacement (m)
        t=te,                       # East time list
        P_array=PE_array,           # East weights matrix
        angular_freq=angular_freq,  # Angular frequencies (annual, semi-annual)
        jump_list=jump_list,        # offsets epochs (all jumps)     
        eq_times=eq_times,          # offsets of earthquakes
        sigma0_init=0.001,          # initial a priori error
        mse_tol=1.0e-6,             # tolerance for iteration   
        max_outlier_iter=10,        # max oulier removal loops
        psd_model_type="auto",      # auto (or exp or log)
        tau_bounds=(0.01, 50.0),    # min and max years of relaxation
        psd_max_iter=50,            # max iteration
        psd_tol=1.0e-6,             # tolerance for psd iterations
        psd_min_rel_improve=0.01,   # min relative improvement of psd vs linear+seasonal to accept psd
        verbose_psd=True,           # print or not psd process
        psd_A0_init=A0_E,           # psd Amplitude initial value
        psd_tau0_init=tau0_E,       # psd Relaxation time initial value
    )

    #______ Up
    (u, th, PU_array,dxU, vxU, U_lin_robust, resU_robust, thetaU, psdU, U_lin, U_psd, resU_lin, resU_full, ssrU_lin, ssrU_full) = analyze_axis(
        label="Up",                 # component label 'Up'
        y=u,                        # Up displacement (m)
        t=th,                       # Up time list
        P_array=PU_array,           # Up weights matrix
        angular_freq=angular_freq,  # Angular frequencies (annual, semi-annual)
        jump_list=jump_list,        # offsets epochs (all jumps)     
        eq_times=eq_times,          # offsets of earthquakes
        sigma0_init=0.001,          # initial a priori error
        mse_tol=1.0e-6,             # tolerance for iteration   
        max_outlier_iter=10,        # max oulier removal loops
        psd_model_type="auto",      # auto (or exp or log)
        tau_bounds=(0.01, 50.0),    # min and max years of relaxation
        psd_max_iter=50,            # max iteration
        psd_tol=1.0e-6,             # tolerance for psd iterations
        psd_min_rel_improve=0.01,   # min relative improvement of psd vs linear+seasonal to accept psd
        verbose_psd=True,           # print or not psd process
        psd_A0_init=A0_U,           # psd Amplitude initial value
        psd_tau0_init=tau0_U,       # psd Relaxation time initial value
    )


    #___________________________________________________Plotting Solutions____________________________________________
    # function for plotting the chosen psd model type 
    def describe_psd_model(psd_terms):
        if not psd_terms:
            return "NO PSD"

        # model types in the order they were fitted
        types = [term["model_type"].upper() for term in psd_terms]

        #compress consecutive duplicates (handle multiple earthquakes)
        comp = []
        for t in types:
            if not comp or comp[-1] != t:
                comp.append(t)

        return "+".join(comp)

    model_N = describe_psd_model(psdN)
    model_E = describe_psd_model(psdE)
    model_U = describe_psd_model(psdU)

    # convert to mm for plotting
    n_mm      = np.array(n)      * 1000.0
    e_mm      = np.array(e)      * 1000.0
    u_mm      = np.array(u)      * 1000.0
    N_lin_mm  = np.array(N_lin)  * 1000.0
    E_lin_mm  = np.array(E_lin)  * 1000.0
    U_lin_mm  = np.array(U_lin)  * 1000.0
    N_psd_mm  = np.array(N_psd)  * 1000.0
    E_psd_mm  = np.array(E_psd)  * 1000.0
    U_psd_mm  = np.array(U_psd)  * 1000.0

    fig, (axN, axE, axU) = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    # ______North
    axN.scatter(tn, n_mm, s=3, label='Raw', color='blue')
    #sigN = pd.Series(np.asarray(sof), index=pd.to_datetime(t)).reindex(pd.to_datetime(tn)).to_numpy()
    #axN.errorbar(tn, n_mm, yerr=sigN * 1000.0, fmt='none', ecolor='k', elinewidth=0.3, capsize=2, capthick=0.3)
    axN.plot(tn, N_lin_mm, label='Linear+seasonal', color='yellow')
    axN.plot(tn, N_psd_mm, label='+PSD model', color='red')

    for jt in jump_list:
        if jt in eq_times:
            axN.axvline(jt, color='0.5', linewidth=1)
        else:
            axN.axvline(jt, color='0.8', linewidth=1)

    axN.set_ylabel("North (mm)")
    axN.legend(loc='best')
    axN.text(0.02, 0.95, f"PSD: {model_N}", transform=axN.transAxes, va="top")


    # ______East
    axE.scatter(te, e_mm, s=3, label='Raw', color='blue')
    #sigE = pd.Series(np.asarray(sol), index=pd.to_datetime(t)).reindex(pd.to_datetime(te)).to_numpy()
    #axE.errorbar(te, e_mm, yerr=sigE * 1000.0, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
    axE.plot(te, E_lin_mm, label='Linear+seasonal', color='yellow')
    axE.plot(te, E_psd_mm, label='+PSD model', color='red')

    for jt in jump_list:
        if jt in eq_times:
            axE.axvline(jt, color='0.5', linewidth=1)
        else:
            axE.axvline(jt, color='0.8', linewidth=1)

    axE.set_ylabel("East (mm)")
    axE.text(0.02, 0.95, f"PSD: {model_E}", transform=axE.transAxes, va="top")


    # ______Up
    axU.scatter(th, u_mm, s=3, label='Raw', color='blue')
    #sigU = pd.Series(np.asarray(soh), index=pd.to_datetime(t)).reindex(pd.to_datetime(th)).to_numpy()
    #axU.errorbar(th, u_mm, yerr=sigU * 1000.0, fmt='none', ecolor='k', elinewidth=0.1, capsize=2, capthick=0.1)
    axU.plot(th, U_lin_mm, label='Linear+seasonal', color='yellow')
    axU.plot(th, U_psd_mm, label='+PSD model', color='red')

    for jt in jump_list:
        if jt in eq_times:
            axU.axvline(jt, color='0.5', linewidth=1)
        else:
            axU.axvline(jt, color='0.8', linewidth=1)

    axU.set_ylabel("Up (mm)")
    axU.set_xlabel("Year")
    axU.text(0.02, 0.95, f"PSD: {model_U}", transform=axU.transAxes, va="top")

    axU.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axU.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(f"{PATH.split(os.sep)[-1]} trajectory", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])


    #___________________________________________________Residuals plots____________________________________________
    # residuals in mm
    resN_lin_mm  = np.array(resN_lin)  * 1000.0
    resN_full_mm = np.array(resN_full) * 1000.0
    resE_lin_mm  = np.array(resE_lin)  * 1000.0
    resE_full_mm = np.array(resE_full) * 1000.0
    resU_lin_mm  = np.array(resU_lin)  * 1000.0
    resU_full_mm = np.array(resU_full) * 1000.0

    fig_res, (axNr, axEr, axUr) = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    # ______North residuals
    axNr.scatter(tn, resN_lin_mm, s=3, label='Linear+seasonal residuals', color='yellow')
    axNr.scatter(tn, resN_full_mm, s=3, label='+PSD residuals', color='red')
    axNr.axhline(0.0, color='0.7', linewidth=1)

    for jt in jump_list:
        if jt in eq_times:
            axNr.axvline(jt, color='0.5', linewidth=1)
        else:
            axNr.axvline(jt, color='0.8', linewidth=1)

    axNr.set_ylabel("North residuals (mm)")
    axNr.legend(loc='best')


    # ______East residuals
    axEr.scatter(te, resE_lin_mm, s=3, label='Linear+seasonal residuals', color='yellow')
    axEr.scatter(te, resE_full_mm, s=3, label='+PSD residuals', color='red')
    axEr.axhline(0.0, color='0.7', linewidth=1)

    for jt in jump_list:
        if jt in eq_times:
            axEr.axvline(jt, color='0.5', linewidth=1)
        else:
            axEr.axvline(jt, color='0.8', linewidth=1)

    axEr.set_ylabel("East residuals (mm)")


    # ______Up residuals
    axUr.scatter(th, resU_lin_mm, s=3, label='Linear+seasonal residuals', color='yellow')
    axUr.scatter(th, resU_full_mm, s=3, label='+PSD residuals', color='red')
    axUr.axhline(0.0, color='0.7', linewidth=1)

    for jt in jump_list:
        if jt in eq_times:
            axUr.axvline(jt, color='0.5', linewidth=1)
        else:
            axUr.axvline(jt, color='0.8', linewidth=1)

    axUr.set_ylabel("Up residuals (mm)")
    axUr.set_xlabel("Year")

    axUr.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axUr.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig_res.suptitle(f"{PATH.split(os.sep)[-1]} residuals", fontsize=14)
    fig_res.tight_layout(rect=[0, 0.03, 1, 0.97])

    report_file.close()
    plt.show()

