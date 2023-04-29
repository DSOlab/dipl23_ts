import datetime

PATH = 'Linear_Model.txt'

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
    return dates, model

dates, y= parse_txt(PATH)

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
print(a)
print(b)