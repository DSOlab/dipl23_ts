import datetime
from matplotlib import pyplot as plt
import numpy as np
import math

PATH = 'Linear_Model.txt'
OMEGA1 = 2 * math.pi * 2
OMEGA2 = 2 * math.pi * 1

## Parse a text file
## Process the data to convert the dates to the number of year using the mean (t0) as the center
## Store in two separate lists
def parse_txt(path):
    dates, model = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            if line[0]!='#': 
                l=line.split()
                dates.append(datetime.datetime.strptime(l[0],"%Y-%m-%dT%H:%M:%S"))
                model.append(float(l[1]))
            else:
                print('header')
    return dates, model

## Converting datetime values to epochs 
def dates2epochs(dates):
    epochs = []
    for date in dates:
        epochs.append(date.year + date.timetuple().tm_yday / 365.25)
    return epochs

## Calculating the middle epoch and return a list of time differences between each epoch and the middle epoch 
def epochs2dtime(epochs):
    dtime=[]
    middle_epoch=(epochs[-1]+epochs[0])/2     #Calculating middle epoch
    print('t0 = ', middle_epoch)
    for epoch in epochs:
        dtime.append(epoch - middle_epoch)      #Calculating and appending the differences
    return dtime

## Implementing the basic formula for Linear Regression using the least squares method to fit a straight line to a set of data points
## Using two lists of data points and returns the slope and y-intercept of the best-fit line for the data
def Linear_Regression(dates, y):
    xx, yy, sx, sy, xy = 0, 0, 0, 0, 0
    for dnum, num in zip(dates, y):
        sx += dnum
        sy += num
        xx += dnum*dnum
        yy += num*num
        xy += num*dnum
    n = float(len(y))

    a = (n*xy - sx*sy)/(n*xx - sx*sx)
    b = ((sy*xx) - (sx*xy))/(n*xx - sx*sx)

    return a, b

## Generating a design matrix based on the given time differences (each time instance (epoch) - the middle epoch), and 2 given angular differences (omega1,2):
## The resulting design matrix named "A" has dimensions "(len(dates), number of coefficients(6))"
## The equation being modeled is: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)
def Design_Matrix(dates, OMEGA1, OMEGA2):
    A = []
    for i in range (len(dates)):
        dm_row = [                          #Creating a row containing the values of the independent variables for each date
            dates[i],
            1,
            math.sin(OMEGA1 * dates[i]),
            math.cos(OMEGA1 * dates[i]),
            math.sin(OMEGA2 * dates[i]),
            math.cos(OMEGA2 * dates[i])
        ]
        A.append(dm_row)                    #Append the constructed row to the Design Matrix (list named "A")
    A = np.array(A)
    print(A)

    return A

## Solving a linear regression problem using the least square method
## Converting the "y" input list to a NumPy array and reshape it into a column vector ensuring that "y" is in the correct format for further computations
def Least_Squares(A, y):
    y = np.array(y)
    y = np.reshape(y, (len(y),1))

    dx = np.linalg.lstsq(A, y, rcond=None)[0]
    return dx

## Given the least square solutions for the equation: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)
## For each date, calculate the corresponding solution creating the final model 
def LS_solutions(dx, dates, OMEGA1, OMEGA2):

    dx = list(dx)
    solutions = []

    for i in range (len(dates)):
        solution = dx[0]*dates[i] + dx[1] + dx[2]*math.sin(OMEGA1*dates[i]) + dx[3]*math.cos(OMEGA1*dates[i]) + dx[4]*math.sin(OMEGA2*dates[i]) + dx[5]*math.cos(OMEGA2*dates[i])
        solutions.append(solution)
    # print(solutions)

    return solutions
## Calculating the linear model
def Linear_sol(a, b, dates):
    lin_sol = []
    for i in range (len(dates)):
        lin_sol.append(a*dates[i] + b)
    return lin_sol

if __name__ == "__main__":
    dates, y = parse_txt(PATH)
    A = Design_Matrix(epochs2dtime(dates2epochs(dates)), OMEGA1, OMEGA2) #Datetime -> Epochs -> dtime
    Final_Model = LS_solutions(Least_Squares(A, y), epochs2dtime(dates2epochs(dates)), OMEGA1, OMEGA2) 
    a, b = Linear_Regression(epochs2dtime(dates2epochs(dates)), y)
    Lin_Solution = Linear_sol(a, b, epochs2dtime(dates2epochs(dates)))

## Plotting the original data imported from the text file (created by timeseriescodedip.py) + the created model

    plt.plot(dates, y, 'o', label = 'Original Data', markersize=1)
    plt.plot(dates, Final_Model, 'r', label = "Final Model")
    plt.plot(dates, Lin_Solution, 'y', label = "Linear Model")

    font = {'family':'serif','color':'blue','size':30}
    font1 = {'family':'serif','color':'darkred','size':20}

    plt.title('Final Model', fontdict = font)
    plt.xlabel('Dates', fontdict = font1)
    plt.ylabel('y - values (mm)', fontdict = font1)
    plt.legend()

    plt.show()
