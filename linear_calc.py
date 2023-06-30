import datetime
import numpy as np
import math

PATH = 'Linear_Model.txt'

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

def Harmonic_Regression(dates):
    # A = np.ones(shape=(len(dates),4))
    # dx = np.ones(shape=(1,4))
    # y = c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)
    omega1 = 2 * math.pi * 2
    omega2 = 2 * math.pi * 1

    A = []
    dx = []

    for i in range (len(dates)):
        A.append(math.sin(omega1*dates[i]))
        A.append(math.cos(omega1*dates[i]))
        A.append(math.sin(omega2*dates[i]))
        A.append(math.cos(omega2*dates[i]))

    A = np.array(A)
    A = np.reshape(A,(len(dates),4))
    print(A.shape)
    return A

dates, y = parse_txt(PATH)
#dates = epochs2dtime(dates2epochs(dates)) #Datetime -> Epochs -> dtime
A = Harmonic_Regression(epochs2dtime(dates2epochs(dates)))
#a, b = Linear_Regression(dates, y)

## for least squares:
## https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
