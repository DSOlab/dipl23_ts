import datetime
from matplotlib import pyplot as plt
import numpy as np
import math

model_coef = [40, 0, -3, -11, -3, -11] #Coefficients of the equation being modeled y = a * x + b + c1 * math.sin(w1 * x) + d1 * math.cos(w1 * x) + c2 * math.sin(w2 * x) + d2 * math.cos(w2 * x)...
timeframe = [datetime.datetime(2015,1,1), datetime.datetime(2019,1,1)]
jump_list = [[datetime.datetime(2016,1,1), 60], [datetime.datetime(2017,1,1), 0]]
w = 2 * math.pi * 2 
w2 = 2 * math.pi * 1 
angular_freq = [w, w2]
white_noise_parameters = [0, 1]

def middle_epoch(timeframe):
    t1, t2 = timeframe
    t0 = t1 + (t2 - t1)/2
    return t0

def middle_epoch_fractional(epochs):
    t0_fractional=(epochs[-1] + epochs[0])/2
    return t0_fractional

# Given a time-frame, the mean value and the std value
# Calculate a white noise signal 
def white_noise(timeframe, white_noise_parameters):
    t1, t2 = timeframe
    mean, std = white_noise_parameters
    white_noise = []
    i = 0
    while t1 <= t2 and i < 1e-5:
        random_number = np.random.normal(mean, std)
        white_noise.append(random_number)
        t1 += datetime.timedelta(days = 1)
    return(white_noise)

# Given two datetime instances (Starting date and finishing date) create a list
# containing every date (as datetime object) between those instances 
def date_calc(timeframe):
    t1, t2 = timeframe
    dates = []
    i = 0
    while t1 <= t2 and i < 1e-5:
        dates.append(t1)
        t1 += datetime.timedelta(days = 1)
    return(dates)

## Converting datetime values to epochs 
def datetime2epochs(dates):
    epochs = []
    for date in dates:
        epochs.append(date.year + date.timetuple().tm_yday / 365.25)
    return epochs

## Calculating the middle epoch and return a list of time differences between each epoch and the middle epoch 
def epochs2dt(epochs):
    dt=[]
    t0 = middle_epoch_fractional(epochs)
    for epoch in epochs:
        dt.append(epoch - t0)      #Calculating and appending the differences
    return dt

# Given the coefficients of a linear model (a and b) and the arguement x (= dt),
# calculate the value of the model at a requested point using: y = a * x + b
def linear(dt, model_coef):
    linear_model = []
    for x in dt:
        lin_y =  x * model_coef[0] + model_coef[1]
        linear_model.append(lin_y)
    return(linear_model)

# Given the coefficients C and D (in-phase, out-of-phase) of a harmonic signal,
# and its angular frequency (omega), calculate the value of the harmonic signal
# at a given epoch (t) using: y = A * sin(omega * t) + B * cos(omega * t)
# Note: the following function calculates 2 signals, for 2 different frequencies
def harmonic(dt, angular_freq, model_coef):
    harmonic_model = []
    for x in dt:
        harmonic = model_coef[2] * math.sin(angular_freq[0] * x) + model_coef[3] * math.cos(angular_freq[0] * x)
        harmonic2 = model_coef[4] * math.sin(angular_freq[1] * x) + model_coef[5] * math.cos(angular_freq[1] * x)
        harmonic_tot = harmonic + harmonic2
        harmonic_model.append(harmonic_tot)
    return(harmonic_model)

def jumps(dates, jump_list):
    offset = []
    for d in dates:
        OS_value = 0
        for jump_t, jump_val in jump_list:
            if d >= jump_t:
                OS_value += jump_val
        offset.append(OS_value)
    return offset

def model(linear_model, harmonic_model, noise, offset):
    model = []
    for i in range (len(linear_model)):
        model.append(linear_model[i] + harmonic_model[i] + offset[i] + noise[i])
    return model
## Generating a design matrix based on the given time differences (each time instance (epoch) - the middle epoch), and 2 or more given angular frequencies (omega1,2):
## The resulting design matrix named "A" has dimensions "(len(dates), number of coefficients(6))"
## The equation being modeled is: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)...
def Design_Matrix(dates, dt, jump_list, angular_freq):
    A = []
    m = 2 + len(angular_freq) * 2 + len(jump_list)                                #Number of parameters 
    for d in zip(dates, dt):
        A_row = []                                                              #Creating a row containing the values of the independent variables for each date
        A_row += [d[1], 1]                                                      #Adding the linear terms
        for w in angular_freq:
            A_row += [math.sin(w * d[1]), math.cos(w * d[1])]           #Adding as many harmonic terms as in the omega list
        for jump in jump_list:
            if d[0] >= jump[0]:
                A_row += [1]
            else:
                A_row += [0]
        assert(len(A_row) == m)
        A.append(A_row)                                                         #Append the constructed row to the Design Matrix (list named "A")
    A = np.array(A)
    return A

## Solving a linear regression problem using the least square method
## Converting the "y" input list to a NumPy array and reshape it into a column vector ensuring that "y" is in the correct format for further computations
def Least_Squares(A, model):
    model = np.array(model)
    model = np.reshape(model, (len(model),1))

    dx = np.linalg.lstsq(A, model, rcond=None)[0]
    return dx

## Given the least square solutions for an equation: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates) + ... + j1 + ...jν
## For each date, calculate the corresponding solution creating the final model 
def LS_solutions(dx, dates, dt, angular_freq, jump_list):
    dx = list(dx)
    solutions = []   
    
    for d in zip(dates, dt):
        lin_sol = dx[0]*d[1] + dx[1]
        harmonic = 0.0
        offset = 2
        for j in range(len(angular_freq)):
            harmonic += dx[offset+j] * math.sin(angular_freq[j] * d[1]) + dx[offset+1+j] * math.cos(angular_freq[j] * d[1])
        jumps = 0.0
        offset = offset + len(angular_freq)*2
        for j in range(len(jump_list)):
            if d[0] >= jump_list[j][0]:
                jumps += dx[offset + j]
        solution = lin_sol + harmonic + jumps
        solutions.append(solution)
    # print(solutions)

    return solutions

def lin_sol(dx, dt):
    dx = list(dx)
    solutions = []
    for d in dt:
        l_sol = dx[0] * d + dx[1]
        solutions.append(l_sol)
    return solutions

def harm_sol(dx, angular_freq, dt):
    dx = list(dx)
    n = len(angular_freq)
    harm = []
    for d in dt:
        goaler = 0
        gorew = 0
        for j in range(n):
            goaler += dx[j] * math.sin(angular_freq[j] * d)
            gorew += dx[j+1] * math.sin(angular_freq[j] * d)
        harm2 = goaler + gorew
        harm.append(harm2)
    
    return harm


t0 = middle_epoch(timeframe)
dates = date_calc(timeframe)
dt = epochs2dt(datetime2epochs(dates))
linear_model = linear(dt, model_coef)
harmonic_model = harmonic(dt, angular_freq, model_coef)
offset = jumps(dates, jump_list)
noise = white_noise(timeframe, white_noise_parameters)
dx = Least_Squares(Design_Matrix(dates, dt, jump_list, angular_freq), model(linear(dt, model_coef), harmonic(dt, angular_freq, model_coef), jumps(dates, jump_list), white_noise(timeframe, white_noise_parameters)))
solution = LS_solutions(dx, dates, dt, angular_freq, jump_list)
y = model(linear(dt, model_coef), harmonic(dt, angular_freq, model_coef), jumps(dates, jump_list), white_noise(timeframe, white_noise_parameters))

lin_test = lin_sol(dx, dt)
harm_test = harm_sol(dx, angular_freq, dt)


plt.plot(dates, harm_test, label = 'Model')
plt.plot(dates, harmonic_model, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('y - values (mm)', fontdict = font1)

plt.show()
