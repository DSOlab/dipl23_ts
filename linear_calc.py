import datetime
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

## Calculating the middle epoch and returns a list of time differences between each epoch and the middle epoch 
def epochs2dtime(epochs):
    dtime=[]
    t0=(epochs[-1]+epochs[0])/2     #Calculating middle epoch
    print('t0 = ', t0)
    for epoch in epochs:
        dtime.append(epoch-t0)      #Calculating and appending the differences
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

dates, y = parse_txt(PATH)
dates = epochs2dtime(dates2epochs(dates)) #Datetime -> Epochs -> dtime
a, b = Linear_Regression(dates, y)
print(dates)
print(b)