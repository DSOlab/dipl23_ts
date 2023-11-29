import datetime
from matplotlib import pyplot as plt
import numpy as np
import math

PATH = 'Linear_Model.txt'
OMEGA1 = 2 * math.pi * 2
OMEGA2 = 2 * math.pi * 1
omega_list = [OMEGA1, OMEGA2]
#omega_list=[]
START_OFFSET = datetime.datetime(2016,1,1)
START_H = 50
END_OFFSET = datetime.datetime(2018,2,20)
END_H = -100
jump_list = [[START_OFFSET, START_H], [END_OFFSET, END_H]]
#jump_list=[]
##--------------------------------INPUTS--------------------------------------------##
# PATH = input('Enter the name of the text file: ')
# Offset timeframe + offset number
# jump_list = []
# while True:
#     Offset_str = input("Enter the offset date using (Year-Month-Day) format, enter 'q' to quit: ")

#     if Offset_str == 'q':
#         break
    
#     OFFSET_DATE = datetime.datetime.strptime(Offset_str, "%Y-%m-%d").date()

#     Offset_h_str = input("Enter the offset height in mm: ")
#     OFFSET_H = float(Offset_h_str)

#     jump_list.append([OFFSET_DATE, OFFSET_H])
# # OMEGA list
# omega_list = []
# while True:
#     w = input("Enter the the angular frequency of the harmonic signal, enter 'q' to quit: ")

#     if w == 'q':
#         break

#     omega = float(w)
#     omega_list.append(omega)

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
    # print('t0 = ', middle_epoch)
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

## Generating a design matrix based on the given time differences (each time instance (epoch) - the middle epoch), and 2 or more given angular frequencies (omega1,2):
## The resulting design matrix named "A" has dimensions "(len(dates), number of coefficients(6))"
## The equation being modeled is: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates)...
def Design_Matrix(dates, jump_list, omega_list):
    A = []
    m = 2 + len(omega_list) * 2 + len(jump_list)                                #Number of parameters 
    frdates = epochs2dtime(dates2epochs(dates))
    for d in zip(dates, frdates):
        A_row = []                                                              #Creating a row containing the values of the independent variables for each date
        A_row += [d[1], 1]                                                  #Adding the linear terms
        for omega in omega_list:
            A_row += [math.sin(omega * d[1]), math.cos(omega * d[1])]   #Adding as many harmonic terms as in the omega list
        for jump in jump_list:
            if d[0] >= jump[0]:
                A_row += [1]
            else:
                A_row += [0]
        #A_row += [1 for _ in jump_list]
        assert(len(A_row) == m)
        A.append(A_row)                                                         #Append the constructed row to the Design Matrix (list named "A")
    A = np.array(A)
    return A

## Solving a linear regression problem using the least square method
## Converting the "y" input list to a NumPy array and reshape it into a column vector ensuring that "y" is in the correct format for further computations
def Least_Squares(A, y):
    y = np.array(y)
    y = np.reshape(y, (len(y),1))

    dx = np.linalg.lstsq(A, y, rcond=None)[0]
    return dx

## Given the least square solutions for an equation: y = a*dates + b + c1*math.sin(w1*dates) + d1*math.cos(w1*dates) + c2*math.sin(w2*dates) + d2*math.cos(w2*dates) + ... + j1 + ...jν
## For each date, calculate the corresponding solution creating the final model 
def LS_solutions(dx, dates, omega_list, jump_list):
    frdates = epochs2dtime(dates2epochs(dates))
    dx = list(dx)
    solutions = []   
    
    for d in zip(dates, frdates):
        lin_sol = dx[0]*d[1] + dx[1]
        harmonic = 0.0
        offset = 2
        for j in range(len(omega_list)):
            harmonic += dx[offset+j] * math.sin(omega_list[j] * d[1]) + dx[offset+1+j] * math.cos(omega_list[j] * d[1])
        jumps = 0.0
        offset = offset + len(omega_list)*2
        for j in range(len(jump_list)):
            if d[0] >= jump_list[j][0]:
                jumps += dx[offset + j]
        solution = lin_sol + harmonic + jumps
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
    #A = Design_Matrix(epochs2dtime(dates2epochs(dates)), jump_list, omega_list) #Datetime -> Epochs -> dtime
    A = Design_Matrix(dates, jump_list, omega_list) #Datetime -> Epochs -> dtime
    dx = Least_Squares(A, y)
    Final_Model = LS_solutions(Least_Squares(A, y), dates, omega_list, jump_list) 
#    a, b = Linear_Regression(epochs2dtime(dates2epochs(dates)), y)
#    Lin_Solution = Linear_sol(a, b, epochs2dtime(dates2epochs(dates)))
    print(dx)
##  Plotting the original data imported from the text file (created by timeseriescodedip.py) + the created model

    plt.plot(dates, y, 'o', label = 'Original Data', markersize=1)
    plt.plot(dates, Final_Model, 'r', label = "Final Model")
#   plt.plot(dates, Lin_Solution, 'y', label = "Linear Model")

    font = {'family':'serif','color':'blue','size':30}
    font1 = {'family':'serif','color':'darkred','size':20}

    plt.title('Final Model', fontdict = font)
    plt.xlabel('Dates', fontdict = font1)
    plt.ylabel('y - values (mm)', fontdict = font1)
    plt.legend()

    plt.show()
