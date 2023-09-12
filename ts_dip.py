import datetime
from matplotlib import pyplot as plt
import numpy as np
import math
from itertools import chain
import sys
import statistics

model_coef = [100, 0, -30, -11, 6, 36, 50, -100] #Coefficients of the equation being modeled y = a * x + b + c1 * math.sin(w1 * x) + d1 * math.cos(w1 * x) + c2 * math.sin(w2 * x) + d2 * math.cos(w2 * x)...
timeframe = [datetime.datetime(2015,1,1), datetime.datetime(2020,1,1)]
jump_list = [] #datetime.datetime(2017,1,1), datetime.datetime(2018,12,12)
w = 2 * math.pi * 2 
w2 = 2 * math.pi * 1 
angular_freq = [w2, w]
white_noise_parameters = [0, 1]
SIGMA0 = 0.001
EPSILON = 0.006694380 
PATH = 'tuc2.cts'

## PARSING CTS FILES
def parse(PATH):
    t, stime, coords = [], [], []
    with open(PATH, 'r') as fin:
        for line in fin.readlines():
            l = line.split()
            if len(l) >= 15:
                tv = (datetime.datetime.strptime(l[0]+' '+l[1],"%Y-%m-%d %H:%M:%S"))
                coordsv = ([ float(j) for j in l[2:14] ])
                try:
                    at = datetime.datetime.strptime(l[14]+' '+l[15],"%Y-%m-%d %H:%M:%S.%f")
                except:
                    at = datetime.datetime.strptime(l[14]+' '+l[15],"%Y-%m-%d %H:%M:%S")
                stimev = at
            try:
                tind = t.index(tv)
                if stimev > stime[tind]:
                    coords[tind] = coordsv
                    stime[tind] = stimev
            except:
                t.append(tv)
                coords.append(coordsv)
                stime.append(stime)
    stime = [x for _,x in sorted(zip(t,stime))]
    coords = [x for _,x in sorted(zip(t,coords))]
    t = sorted(t)
    return t, stime, coords

def geterrors(coordinates):
    sf, sl, sh = [], [], []
    for flh_elements in coordinates:
        sf.append(flh_elements[7])
        sl.append(flh_elements[9])
        sh.append(flh_elements[11])
    return sf, sl, sh

def getxyz(coordinates):
    x, y, z = [], [], []
    for xyz_elements in coordinates:
        x.append(xyz_elements[0])
        y.append(xyz_elements[2])
        z.append(xyz_elements[4])
    return x, y, z

def mean_xyz(x, y, z):
    xm, ym, zm = map(statistics.mean, [x, y, z])
    return xm, ym, zm

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

#calculating rotation matrix
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


# Given two datetime instances (Starting date and finishing date) create a list
# containing every date (as datetime object) between those instances 
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

def weights_calc(coord_errors, sigma0):
    weights = [(1 / ce ) * sigma0 for ce in coord_errors]
    weights_array = np.array(weights)
    #P_matrix = np.diag(weights_array)
    #print(P_matrix)
    return weights_array #P_matrix

## Given the least square solutions for an equation: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates) + ... + j1 + ...jν
## For each date, calculate the corresponding solution creating the final model 
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

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------LEAST SQUARE SOLUTIONS-----------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
## Generating a design matrix based on the given time differences (each time instance (epoch) - the middle epoch), and 2 or more given angular frequencies (omega1,2):
## The resulting design matrix named "A" has dimensions "(len(dates), number of coefficients(6))"
## The equation being modeled is: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)...
def Design_Matrix(t, jump_list, angular_freq):
    A = []
    m = 2 + len(angular_freq) * 2 + len(jump_list)                                #Number of parameters 
    dt = fractionaldt(t)   
    for d in zip(t, dt):
        A_row = []                                                              #Creating a row containing the values of the independent variables for each date
        A_row += [d[1], 1]                                                      #Adding the linear terms
        for w in angular_freq:
            A_row += [math.sin(w * d[1]), math.cos(w * d[1])]                   #Adding as many harmonic terms as in the omega list
        for jump_t in jump_list:
            if d[0] >= jump_t:
                A_row += [1]
            else:
                A_row += [0]
        assert(len(A_row) == m)
        A.append(A_row)                                                         #Append the constructed row to the Design Matrix (list named "A")
    A = np.array(A)
    return A

## Solving a linear regression problem using the least square method
## Converting the "y" input list to a NumPy array and reshape it into a column vector ensuring that "y" is in the correct format for further computations
def fit(y, t, P_array , freq, jumps, SIGMA0): ## t = dates in datetime, y = parsed solutions, freq = given angular frequencies, jumps = list of datetime objects (time instance when the jump happened)
    #P_array =  np.ones(len(y))
    #model = np.reshape(y, (len(y),1))
    model = y * np.sqrt(P_array)
    A = Design_Matrix(t, jumps, freq)
    print(A.shape)
    AP = A * np.sqrt(P_array)[:, None]
    dx, sumres, _,  _ = np.linalg.lstsq(AP, model, rcond=None)
    dx = dx.flatten()
    AT = np.transpose(AP)
    cov_matrix = np.linalg.inv(AT@AP)*SIGMA0**2
    return dx, cov_matrix, sumres/(len(y)-1)

def residuals(y, y_predicted):
    residuals = [actual - pred for actual, pred in zip(y, y_predicted)]
    squared_residuals = [residual ** 2 for residual in residuals]
    sum_sq_res = sum(squared_residuals)
    mse = sum_sq_res / (len(y) - 1) ##mean square error
    return residuals, mse

def remove_outliers(y, t, P_array, residuals, sigmap , SIGMA0):
    limit3s = 3 * sigmap 
    yy, tt, pp = [], [], []
    for i, res in enumerate(residuals):
        if abs(res) < limit3s:
            tt.append(t[i])
            yy.append(y[i])
            pp.append(P_array[i])
    #P = np.array(pp)
    #P_matrix = np.diag(P)
    return yy, tt, np.array(pp)

if __name__ == "__main__":
    dates = date_calc(timeframe)

    t, stime, coords = parse(PATH)
    x, y, z = getxyz(coords)
    sof, sol, soh = geterrors(coords)
    e, n, u = topocentric_conversion(x, y, z)
    na = n
    ea = e    
    ta, tn, te, th = t, t, t, t

    PN_array = weights_calc(sof, SIGMA0)
    PE_array = weights_calc(sol, SIGMA0)
    PU_array = weights_calc(soh, SIGMA0)

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
    y_tel = compute_model(dx, tn, angular_freq, jump_list)                
    uun, _ = residuals(n, y_tel)                                          
    print('Ο Αριθμός των επαναλήψεων είναι: ', j)
    print('Τα αποτελέσματα της συνόρθωσης είναι:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.6f}'.format(x*1000, math.sqrt(vx[i][i])))

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
    y_tel = compute_model(dx, te, angular_freq, jump_list)                     
    uue, _ = residuals(e, y_tel)
    print('Ο Αριθμός των επαναλήψεων είναι: ', k)
    print('Τα αποτελέσματα της συνόρθωσης είναι:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.6f}'.format(x*1000, math.sqrt(vx[i][i])))

    # Timeseries analysis of Z - Up axis:
    counter = 0
    SIGMA0 = 0.001
    mse=1
    mset=1000
    while abs(mse - mset) > 1.0e-6 and counter < 10:
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
    y_tel = compute_model(dx, th, angular_freq, jump_list)
    uuu, _ = residuals(u, y_tel)
    print('Ο Αριθμός των επαναλήψεων είναι: ', counter)
    print('Τα αποτελέσματα της συνόρθωσης είναι:')
    for i,x in enumerate(dx):
        print('{:+10.3f} +- {:10.6f}'.format(x*1000, math.sqrt(vx[i][i])))

    ## PLOTTING THE SOLUTION
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.scatter(te, e, s=1.5, label='East')
    ax1.scatter(tn, n, s=1.5, label='North')
    ax2.scatter(th, u, s=1.5, label='Up')

    font1 = {'family':'serif','color':'blue','size':30}
    font2 = {'family':'serif','color':'darkred','size':20}

    # ax0.set_ylim(-0.1, 0.7)
    # ax1.set_ylim(-0.5, 0.2)
    # ax2.set_ylim(-0.02, 0.02)

    ax0.set_title('Timeseries Plot', fontdict = font1)
    ax2.set_xlabel("Dates", fontdict = font2)

    ax0.set_ylabel("East (m)", fontdict = font2)
    ax1.set_ylabel("North (m)", fontdict = font2)
    ax2.set_ylabel("Up (m)", fontdict = font2)

    plt.show()
    
    ## PLOTTING THE SOLUTION                                 
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.scatter(te, uue, s=1.5, label='East')
    ax1.scatter(tn, uun, s=1.5, label='North')
    ax2.scatter(th, uuu, s=1.5, label='Up')
 
    font1 = {'family':'serif','color':'blue','size':30}
    font2 = {'family':'serif','color':'darkred','size':20}
 
    # ax0.set_ylim(-0.1, 0.7)
    # ax1.set_ylim(-0.5, 0.2)
    # ax2.set_ylim(-0.02, 0.02)
 
    ax0.set_title('Timeseries Plot', fontdict = font1)
    ax2.set_xlabel("Dates", fontdict = font2)
 
    ax0.set_ylabel("East (m)", fontdict = font2)
    ax1.set_ylabel("North (m)", fontdict = font2)
    ax2.set_ylabel("Up (m)", fontdict = font2)
 
    plt.show()