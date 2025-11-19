# pip install matplotlib numpy pandas ruptures

import numpy as np
import sys
import statistics
import datetime
from datetime import datetime as dt, timedelta
from matplotlib import pyplot as plt
import math
from itertools import chain
from pathlib import Path
import pandas as pd
import ruptures as rpt

#   471 ALOG_E SPAN  A ---- 15:321:25807 m    2 -5.19025774935067e-03 1.27922e-04
#   472 TLOG_E SPAN  A ---- 15:321:25807 m    2  1.28214850809435e-01 1.95986e-02
#   473 ALOG_N SPAN  A ---- 15:321:25807 m    2 -5.32509199464391e-03 1.26611e-04
#   474 TLOG_N SPAN  A ---- 15:321:25807 m    2  1.34576603442766e-01 1.92184e-02

# !!!!!!!!!!!!!!!!!!!!!!!!!!Path
PATH = r'../hector21_gr/cts_repro/span.cts'          #change path for .cts or .txyz2 file


#_____________________________________________________Model Parameters_______________________________________________________
# Coefficients of the equation being modeled
# y = a * x + b + c1 * math.sin(w1 * x) + d1 * math.cos(w1 * x) + c2 * math.sin(w2 * x) + d2 * math.cos(w2 * x)... + J1 + ... Jn

model_coef = [100, 0, -30, -11, 6, 36, 50, -100]
timeframe = [datetime.datetime(2015,1,1), datetime.datetime(2020,1,1)]
jump_list = []
angular_freq = [2 * math.pi * 1, 2 * math.pi * 2]
white_noise_parameters = [0, 1]
SIGMA0 = 0.001
EPSILON = 0.006694380

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
        coords.append(c)
        c[7]  = c[1]  # σE
        c[9]  = c[3]  # σN
        c[11] = c[5]  # σU
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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        df["x_sigma"] = df["sigma_e"]
        df["y_sigma"] = df["sigma_n"]
        df["z_sigma"] = df["sigma_v"]

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
        df = parse_txyz2(PATH)
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
    mse = sum_sq_res / (len(y) - 1)                              #mean square error
    return residuals, mse

#outliers removal
def remove_outliers(y, t, P_array, residuals, sigmap , SIGMA0):
    limit3s = 3 * sigmap                                         #THRESHOLD OUTLIERS!!!!!!!
    yy, tt, pp = [], [], []
    for i, res in enumerate(residuals):
        if abs(res) < limit3s:
            tt.append(t[i])
            yy.append(y[i])
            pp.append(P_array[i])
    #P = np.array(pp)
    #P_matrix = np.diag(P)
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
# Run PELT

    RPT_MODEL = "rbf"                       # l2 / rbf
    RPT_MIN_SIZE = 5                        # min samples between jumps
    RPT_JUMP = 5                            # step
    RPT_PEN = 10                            # penalty
    earthquake_threshold = 10.0             # jumps >= distance (mm)
    earthquake_w = 10                       # window on each side to find step (median)

    RPT_PEN_E = RPT_PEN                     # penalty for East
    RPT_PEN_N = RPT_PEN                     # penalty for North
    RPT_PEN_U = RPT_PEN                     # penalty for Up
    RPT_MERGE_TOL = 5                       # tolerance (in samples) to consider 2 breakpoints the same
    RPT_MIN_AXES_AGREE = 2                  # require breakpoint to appear on at least this many axes

    def _pelt_1d(values, pen, model=RPT_MODEL, min_size=RPT_MIN_SIZE, jump=RPT_JUMP):
        arr = np.asarray(values).reshape(-1, 1)
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(arr)
        return algo.predict(pen=pen)  # list of segment end indices

    def _merge_bkps(list_of_bkps, n, tol=RPT_MERGE_TOL, min_axes=RPT_MIN_AXES_AGREE):
        candidates = sorted(set(i for bkps in list_of_bkps for i in bkps if i < n))
        merged = []
        for c in candidates:
            if any(abs(c - m) <= tol for m in merged):
                continue
            # count how many axes have a breakpoint near c
            count = 0
            cluster = [c]
            for bkps in list_of_bkps:
                near = [b for b in bkps if b < n and abs(b - c) <= tol]
                if near:
                    count += 1
                    cluster.extend(near)
            if count >= min_axes:
                m_idx = int(round(np.median(cluster)))
                if 0 < m_idx < n:   # keep strictly inside (0, n)
                    merged.append(m_idx)
        return sorted(set(merged))

    # Stack to compute step sizes
    signal = np.column_stack((e, n, u))                 # shape: (N, 3)
    Nsig = len(signal)

    # PELT per axis
    bkps_e = _pelt_1d(e, RPT_PEN_E)
    bkps_n = _pelt_1d(n, RPT_PEN_N)
    bkps_u = _pelt_1d(u, RPT_PEN_U)

    # Merge
    bkps_idx = _merge_bkps([bkps_e, bkps_n, bkps_u], Nsig, tol=RPT_MERGE_TOL, min_axes=RPT_MIN_AXES_AGREE)

    # step size at each merged breakpoint (using all 3 axes for magnitude)
    def step_mm(sig, idx, w=earthquake_w):
        nloc = len(sig)                                    # length of signal
        i0 = max(0, idx - w)                               # start of before window
        i1 = min(idx, nloc)                                # end of before segment
        i2 = min(idx + w, nloc)                            # end of after segment
        if i1 <= i0 or i2 <= i1:
            return 0.0, np.array([0.0, 0.0, 0.0])
        pre = np.nanmedian(sig[i0:i1], axis=0)             # median before
        post = np.nanmedian(sig[i1:i2], axis=0)            # median after
        delta = post - pre                                 # per axis (m)
        mag_mm = float(np.linalg.norm(delta) * 1000.0)     # magnitude in mm
        return mag_mm, delta

    earthquake_list = []
    kept_info = []  # (epoch, mag_mm, deltaE_mm, deltaN_mm, deltaU_mm)

    for idx in bkps_idx:
        mag_mm, d_m = step_mm(signal, idx, earthquake_w)
        if mag_mm >= earthquake_threshold:
            i = min(idx, len(t) - 1)
            ep = t[i]
            earthquake_list.append(ep)
            kept_info.append((
                ep,
                mag_mm,
                d_m[0] * 1000.0,   # E in mm
                d_m[1] * 1000.0,   # N in mm
                d_m[2] * 1000.0    # U in mm
            ))

    earthquake_list = sorted(set(earthquake_list))

    # Connect jump_list with earthquake_list
    jump_list = earthquake_list[:]
    earthquake_list = jump_list

    # Print results
    if earthquake_list:
        print("\nDetected earthquake jumps (UTC) [merged per-axis PELT]:")
        for ep in sorted(earthquake_list):
            date_str = ep.strftime("%Y-%m-%d")
            print(f"  - {ep}  (date: {date_str})")
    else:
        print("\nNo earthquake jumps detected (merged per-axis PELT).")

        #____________________________________________________Timeseries___________________________________________________
    #dx, vx, mse = fit(n, tn, PN_array, angular_freq, jump_list, SIGMA0)
    #E_tel = compute_model(dx, tn, angular_freq, jump_list)
    #uun, _ = residuals(n, E_tel)
    #n, tn, PN_array = remove_outliers(n, tn, PN_array, uun, math.sqrt(mse), SIGMA0)

    #dx, vx, mse = fit(e, te, PE_array, angular_freq, jump_list, SIGMA0)
    #E_tel = compute_model(dx, te, angular_freq, jump_list)
    #uue, _ = residuals(e, E_tel)
    #e, te, PE_array = remove_outliers(e, te, PE_array, uue, math.sqrt(mse), SIGMA0)

    #dx, vx, mse = fit(u, th, PU_array, angular_freq, jump_list, SIGMA0)
    #U_tel = compute_model(dx, th, angular_freq, jump_list)
    #uuu, _ = residuals(u, U_tel)
    #u, th, PU_array = remove_outliers(u, th, PU_array, uuu, math.sqrt(mse), SIGMA0)


    # Timeseries analysis of X - North axis:
    j = 0
    mse=1
    mset=1000
    while abs(mse - mset) > 1.0e-6 and j < 10:
        if j!=0:
            mset = mse
        dx, vx, mse = fit(n, tn, PN_array, angular_freq, jump_list, SIGMA0)
        y_tel = compute_model(dx, tn, angular_freq, jump_list)
        uun, _ = residuals(n, y_tel)
        tres = tn
        n, tn, PN_array = remove_outliers(n, tn, PN_array, uun, math.sqrt(mse), SIGMA0)
        #print(PN_array.shape)
        SIGMA0=math.sqrt(mse)
        j+=1
    dx, vx, mse = fit(n, tn, PN_array, angular_freq, jump_list, SIGMA0)
    N_tel = compute_model(dx, tn, angular_freq, jump_list)
    uun, _ = residuals(n, N_tel)
    print('North Solutions:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.3f}'.format(x*1000, math.sqrt(vx[i][i])*1000))


    # Timeseries analysis of Y - East axis:
    k = 0
    SIGMA0 = 0.001
    mse=1
    mset=1000
    while abs(mse - mset) > 1.0e-6 and k < 10:
        if k!=0:
            mset = mse
        dx, vx, mse = fit(e, te, PE_array, angular_freq, jump_list, SIGMA0)
        y_tel = compute_model(dx, te, angular_freq, jump_list)
        uue, _ = residuals(e, y_tel)
        tres = te
        e, te, PE_array = remove_outliers(e, te, PE_array, uue, math.sqrt(mse), SIGMA0)
        SIGMA0=math.sqrt(mse)
        k+=1
    dx, vx, mse = fit(e, te, PE_array, angular_freq, jump_list, SIGMA0)
    E_tel = compute_model(dx, te, angular_freq, jump_list)
    uue, _ = residuals(e, E_tel)
    print('East Solutions:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.3f}'.format(x*1000, math.sqrt(vx[i][i])*1000))


    # Timeseries analysis of Z - Up axis:
    counter = 0
    SIGMA0 = 0.001
    mse=1
    mset=1000
    while abs(mse - mset) > 1.0e-3 and counter < 10:
        if counter!=0:
            mset = mse
        dx, vx, mse = fit(u, th, PU_array, angular_freq, jump_list, SIGMA0)
        y_tel = compute_model(dx, th, angular_freq, jump_list)
        uuu, _ = residuals(u, y_tel)
        tres = th
        u, th, PU_array = remove_outliers(u, th, PU_array, uuu, math.sqrt(mse), SIGMA0)
        SIGMA0=math.sqrt(mse)
        counter+=1
    dx, vx, mse = fit(u, th, PU_array, angular_freq, jump_list, SIGMA0)
    U_tel = compute_model(dx, th, angular_freq, jump_list)
    uuu, _ = residuals(u, U_tel)
    print('Up Solutions:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.3f}'.format(x*1000, math.sqrt(vx[i][i])*1000))

#___________________________________________________Plotting Solutions____________________________________________
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.scatter(te, e, s=1.5, label='Solution - East')
    ax0.plot(te, E_tel, label='Final Model - East', color='red')
    ax1.scatter(tn, n, s=1.5, label='Solution - North')
    ax1.plot(tn, N_tel, label='Final Model - North', color='red')
    ax2.scatter(th, u, s=1.5, label='Solution - Up')
    ax2.plot(th, U_tel, label='Final Model - Up', color='red')

    # Outliers comparison
    #ax0.scatter(t, ea, s=1.5, label='Outliers - East ', color='red')
    #ax0.scatter(te, e, s=1.5, label='East', color='blue')
    #ax1.scatter(t, na, s=1.5, label='Outliers - North', color='red')
    #ax1.scatter(tn, n, s=1.5, label='North', color='blue')
    #ax2.scatter(t, ua, s=1.5, label='Outliers - Up', color='red')
    #ax2.scatter(th, u, s=1.5, label='Up', color='blue')

    font1 = {'family':'serif','color':'blue','size':30}
    font2 = {'family':'serif','color':'darkred','size':20}

    ax0.set_title(f'Time Series Final Model Plot - {PATH[:4]}', fontdict = font1)
    ax2.set_xlabel("Dates", fontdict = font2)

    ax0.legend()
    ax1.legend()
    ax2.legend()

    ax0.set_ylabel("East (m)", fontdict = font2)
    ax1.set_ylabel("North (m)", fontdict = font2)
    ax2.set_ylabel("Up (m)", fontdict = font2)

    plt.show()

    ## PLOTTING THE RESIDUALS
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.scatter(te, uue, s=1.5, label='East (m)')
    ax1.scatter(tn, uun, s=1.5, label='North (m)')
    ax2.scatter(th, uuu, s=1.5, label='Up (m)')

    ax0.set_title(f'Residuals Plot - {PATH[:4]}', fontdict = font1)
    ax2.set_xlabel("Dates", fontdict = font2)

    ax0.set_ylabel("East (m)", fontdict = font2)
    ax1.set_ylabel("North (m)", fontdict = font2)
    ax2.set_ylabel("Up (m)", fontdict = font2)

    plt.show()
