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
    #with open(path, 'r') as f:
    #    next(f)
    #    contents = f.read()
    #for line in contents.splitlines():
    #    elements = line.split('\t') 
    #    d = float(elements[0])
    #    y = float(elements[1].strip('[]'))
    #    dates.append(d)
    #    model.append(y)
    #t0 = sum(dates) / len(dates)
    #dates = [d - t0 for d in dates]
    return dates, model

def dates2epochs(dates):
    epochs = []
    for date in dates:
        epochs.append(date.year + date.timetuple().tm_yday / 365.25)
    return epochs

def epochs2dtime(epochs):
    dtime=[]
    t0=(epochs[-1]+epochs[0])/2
    print(t0)
    for epoch in epochs:
        dtime.append(epoch-t0)
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

dates, y= parse_txt(PATH)
#epochs = dates1epochs(dates)
dates=epochs2dtime(dates2epochs(dates))

a, b = Linear_Regression(dates, y)
print(a)
print(b)