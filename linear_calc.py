PATH = 'Linear_Model.txt'

## Parse a text file
## Process the data to convert the dates to the number of year using the mean (t0) as the center
## Store in two separate lists
def parse_txt(path):
    dates, model = [], []
    with open(path, 'r') as f:
        next(f)
        contents = f.read()
    for line in contents.splitlines():
        elements = line.split('\t') 
        d = float(elements[0])
        y = float(elements[1].strip('[]'))
        dates.append(d)
        model.append(y)
    t0 = sum(dates) / len(dates)
    dates = [d - t0 for d in dates]
    return dates, model
dates, y= parse_txt(PATH)

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

a, b = Linear_Regression(dates, y)