import datetime
from matplotlib import pyplot as plt
import numpy
import math

start_str = input("Enter a Starting Date using (Year-Month-Day) - format: ")
end_str = input("Enter a Finishing Date using (Year-Month-Day) - format: ")

START_DATE = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
END_DATE = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

A = 40 # mm/year
B = 0 
C = -3
D = -11
mid_epoch = (END_DATE.year + START_DATE.year)/2
mean = 0
std = 1 

def white_noise_calc(START_DATE, END_DATE, mean, std):
    white_noise = []
    i = 0
    while START_DATE <= END_DATE and i < 1e-5:
        rand_numb = numpy.random.normal(mean, std)
        white_noise.append(rand_numb)
        START_DATE += datetime.timedelta(days = 1)
    
    return(white_noise)

def date_calc(START_DATE, END_DATE):
    dates = []
    i = 0
    while START_DATE <= END_DATE and i < 1e-5:
        dates.append(START_DATE)
        START_DATE += datetime.timedelta(days = 1)

    return(dates)

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

lin_model = linear_model(START_DATE, END_DATE, mid_epoch, A, B)
sin_model = harmonic_model(START_DATE, END_DATE, mid_epoch, C, D)
white_noise = white_noise_calc(START_DATE, END_DATE, mean, std)
dates = date_calc(START_DATE, END_DATE)


## Creating a text file, and writing dates + white noise in it

with open("Linear_Model.txt", "w") as file:
    print('{:20s} {:20s}'.format('#dates', 'linear_y'), file=file)
    for i in range(len(dates)):
        print('{:} {:+.20e}'.format(datetime.datetime.strftime(dates[i],"%Y-%m-%dT%H:%M:%S"), white_noise[i]),file=file)

## Ploting the linear and the harmonic model at the same time

model = []

for i in range (len(lin_model)):
    model.append(lin_model[i] + sin_model[i] + white_noise[i])
    

plt.plot(dates, model, label = 'Linear Model')
# plt.plot(dates, w_noise, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Linear Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('Model (mm)', fontdict = font1)

plt.show()