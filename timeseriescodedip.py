import datetime
from matplotlib import pyplot as plt
import numpy
from numpy import random


start_str = input("Enter a Starting Date using (Year-Month-Day) - format: ")
end_str = input("Enter a Finishing Date using (Year-Month-Day) - format: ")

START_DATE = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
END_DATE = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

A = 40 # mm/year
B = 0 
mean = 0
std = 1 
num_samples = 1


def linear_model(START_DATE, END_DATE, A, B):
    dates, linear_model, random_numb  = [], [], []
    j = 0

    ## Creating year-day numbers and calculating the linear model
    while START_DATE <= END_DATE and j < 3000:
        #rand_numb = random.randint(-5, 5) # mm
        ## Add white noise in additional to random
        rand_numb = numpy.random.normal(mean, std, size=num_samples)
        
        date_numb = START_DATE.year + START_DATE.timetuple().tm_yday / 365.25
        linear_model_calc = (date_numb - 2015) * A + B
        rand_numb = rand_numb + linear_model_calc

        random_numb.append(rand_numb)
        dates.append(date_numb)
        linear_model.append(linear_model_calc)
        START_DATE += datetime.timedelta(days = 1)  ## Move the line at the end of the loop
         
    print(dates)
    print(random_numb)
    print(linear_model)
    return(dates, random_numb, linear_model)

dates, random_numb, model = linear_model(START_DATE, END_DATE, A, B)

## Plotting linear model 

plt.plot(dates, model, label = 'Linear Model')
plt.plot(dates, random_numb, label = 'White noise')

font = {'family':'serif','color':'blue','size':30}
font1 = {'family':'serif','color':'darkred','size':20}

plt.title('Linear Model', fontdict = font)
plt.xlabel('Dates', fontdict = font1)
plt.ylabel('Model (mm)', fontdict = font1)

plt.show()