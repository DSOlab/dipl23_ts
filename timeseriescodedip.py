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
C = 40
D = 40
T0 = (END_DATE.year + START_DATE.year)/2
print(T0)
mean = 0
std = 1 
num_samples = 1


def linear_model(START_DATE, END_DATE, T0, A, B):
    dates, linear_model, white_noise  = [], [], []
    j = 0

    ## Creating year-day numbers and calculating the linear model
    while START_DATE <= END_DATE and j < 10000:
        rand_numb = numpy.random.normal(mean, std)

        epochs = START_DATE.year + START_DATE.timetuple().tm_yday / 365.25
        linear_y = (epochs - T0) * A + B

        white_noise.append(rand_numb)
        dates.append(START_DATE)
        linear_model.append(linear_y)

        START_DATE += datetime.timedelta(days = 1)

    return(dates, white_noise, linear_model)


def sinusoidal_model(START_DATE, END_DATE, T0, C, D):
    sin_model = []
    j = 0
    while START_DATE <= END_DATE and j < 10000:
        epochs = (START_DATE.year + START_DATE.timetuple().tm_yday / 365.25) - T0
        w = 2 * math.pi * 0.5
        sm = C*math.sin(w*epochs) + D*math.cos(w*epochs)

        sin_model.append(sm)

        START_DATE += datetime.timedelta(days = 1)
    return(sin_model)

dates, white_noise, lin_model = linear_model(START_DATE, END_DATE, T0, A, B)
sin_model = sinusoidal_model(START_DATE, END_DATE, T0, C, D)


## Creating a text file, and writing dates + white noise in it

with open("Linear_Model.txt", "w") as file:    ##Needs to be fixed 
    print('{:20s} {:20s}'.format('#dates', 'linear_y'), file=file)
    for i in range(len(dates)):
        print('{:} {:+.20e}'.format(datetime.datetime.strftime(dates[i],"%Y-%m-%dT%H:%M:%S"), white_noise[i]),file=file)

## Ploting the linear and the sinusoidal model at the same time

model, w_noise = [], []

for i in range (len(lin_model)):
    model.append(lin_model[i] + sin_model[i])
    w_noise.append(model[i] + white_noise[i])

plt.plot(dates, model, label = 'Linear Model')
plt.plot(dates, w_noise, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Linear Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('Model (mm)', fontdict = font1)

plt.show()