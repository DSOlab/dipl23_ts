import datetime
from matplotlib import pyplot as plt
import numpy

start_str = input("Enter a Starting Date using (Year-Month-Day) - format: ")
end_str = input("Enter a Finishing Date using (Year-Month-Day) - format: ")

START_DATE = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
END_DATE = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

A = 40 # mm/year
B = 0 
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
        rand_numb = numpy.random.normal(mean, std, size=num_samples)

        epochs = START_DATE.year + START_DATE.timetuple().tm_yday / 365.25
        linear_y = (epochs - T0) * A + B
        w_noise = rand_numb + linear_y

        white_noise.append(w_noise)
        dates.append(epochs)
        linear_model.append(linear_y)

        START_DATE += datetime.timedelta(days = 1)

    return(dates, white_noise, linear_model)

dates, white_noise, model = linear_model(START_DATE, END_DATE, T0, A, B)

## Creating a text file, and writing dates + white noise in it

with open("Linear_Model.txt", "w") as file:
    file.write("dates \t \t \t    linear_y \n")                 ##Needs to be fixed 

    for i in range(len(dates)):
        file.write(str(dates[i]) + "\t" + str(white_noise[i]) + "\n")

## Plotting linear model 

plt.plot(dates, model, label = 'Linear Model')
plt.plot(dates, white_noise, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Linear Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('Model (mm)', fontdict = font1)

plt.show()