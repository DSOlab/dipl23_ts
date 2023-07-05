import datetime
from matplotlib import pyplot as plt
import numpy
import math

# # Enter a Starting and a Finishing date from keyboard
# start_str = input("Enter a Starting Date using (Year-Month-Day) - format: ")
# end_str = input("Enter a Finishing Date using (Year-Month-Day) - format: ")

# START_DATE = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
# END_DATE = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

# Offset timeframe + offset number
jump_list = []
while True:
    Offset_str = input("Enter the offset date using (Year-Month-Day) format, enter 'q' to quit: ")

    if Offset_str == 'q':
        break
    
    OFFSET_DATE = datetime.datetime.strptime(Offset_str, "%Y-%m-%d").date()

    Offset_h_str = input("Enter the offset height in mm: ")
    OFFSET_H = float(Offset_h_str)

    jump_list.append((OFFSET_DATE, OFFSET_H))

START_DATE = datetime.date(2015,1,1)
END_DATE = datetime.date(2019,1,1)

# START_OFFSET = datetime.date(2016,1,1)
# START_H = 60
# END_OFFSET = datetime.date(2017,1,1)
# END_H = 0

A = 40 # mm/year
B = 0 
C = -3
D = -11
mid_epoch = (END_DATE.year + START_DATE.year)/2
mean = 0
std = 1 

# Given a time-frame, the mean value and the std value
# Calculate a white noise signal 
def white_noise_calc(START_DATE, END_DATE, mean, std):
    white_noise = []
    i = 0
    while START_DATE <= END_DATE and i < 1e-5:
        rand_numb = numpy.random.normal(mean, std)
        white_noise.append(rand_numb)
        START_DATE += datetime.timedelta(days = 1)      #Replace Start - End date with the length of the dates list
    
    return(white_noise)

# Given two datetime instances (Starting date and finishing date) creat a list
# containing every date (as datetime object) between those instances 
def date_calc(START_DATE, END_DATE):
    dates = []
    i = 0
    while START_DATE <= END_DATE and i < 1e-5:
        dates.append(START_DATE)
        START_DATE += datetime.timedelta(days = 1)

    return(dates)

# Given the coefficients of a linear model (a and b) and the arguement x,
# calculate the value of the model at a requested point using: y = a * x + b
def linear_model(START_DATE, END_DATE, mid_epoch, A, B):
    linear_model = []
    j = 0

    ## Creating year-day numbers and calculating the linear model
    while START_DATE <= END_DATE and j < 1e-5:
        epochs = START_DATE.year + START_DATE.timetuple().tm_yday / 365.25
        linear_y = (epochs - mid_epoch) * A + B

        linear_model.append(linear_y)

        START_DATE += datetime.timedelta(days = 1)

    return(linear_model)

# Given the coefficients C and D (in-phase, out-of-phase) of a harmonic signal,
# and its angular frequency (omega), calculate the value of the harmonic signal
# at a given epoch (t) using: y = A * sin(omega * t) + B * cos(omega * t)
# Note: the following function calculates 2 signals, for 2 different frequencies
def harmonic_model(START_DATE, END_DATE, mid_epoch, C, D):
    sin_model = []
    j = 0
    while START_DATE <= END_DATE and j < 1e-5:
        dt = (START_DATE.year + START_DATE.timetuple().tm_yday / 365.25) - mid_epoch
        omega = 2 * math.pi * 2 
        omega2 = 2 * math.pi * 1 
        harmonic = C*math.sin(omega*dt) + D*math.cos(omega*dt)
        harmonic2 = C*math.sin(omega2*dt) + D*math.cos(omega2*dt)
        harmonic_tot = harmonic + harmonic2
        sin_model.append(harmonic_tot)

        START_DATE += datetime.timedelta(days = 1)
    return(sin_model)

# Given the dates and the offset number, create a list that contains the offset values
# def jumps(dates):
#     OS = []
#     for date in dates:
#         if date <= datetime.date(2016,1,1):
#             OS.append(0)
#         elif date <= datetime.date(2017,1,1):
#             OS.append(50)
#         else:
#             OS.append(-120)
#     return OS

def jumps_import(dates, jump_list):
    OS = []
    for d in dates:
        if d <= jump_list[0][0]:
            OS.append(0)
        elif d <= jump_list[1][0]:
            OS.append(jump_list[0][1])
        else:
            OS.append(jump_list[1][1])
    return OS


linear_model = linear_model(START_DATE, END_DATE, mid_epoch, A, B)
harmonic_model = harmonic_model(START_DATE, END_DATE, mid_epoch, C, D)
white_noise = white_noise_calc(START_DATE, END_DATE, mean, std)
dates = date_calc(START_DATE, END_DATE)
# offset = jumps(dates)
offset = jumps_import(dates, jump_list)

## Ploting the model: Linear model + harmonic model (frequency = 1 year || frequency = 0.5 years) + offset

model = []

for i in range (len(linear_model)):
    model.append(linear_model[i] + harmonic_model[i] + white_noise[i] + offset[i])

## Creating a text file containing dates + y values

with open("Linear_Model.txt", "w") as file:
    print('{:20s} {:20s}'.format('#dates', 'Model'), file=file)
    for i in range(len(dates)):
        print('{:} {:+.20e}'.format(datetime.datetime.strftime(dates[i],"%Y-%m-%dT%H:%M:%S"), model[i]),file=file)

plt.plot(dates, model, label = 'Model')
# plt.plot(dates, w_noise, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('y - values (mm)', fontdict = font1)

plt.show()