import numpy
from .core import getNumPyArray

def describe(filename):
    data = getNumPyArray(filename)
    headers = data[0]
    count = []
    mean = []
    std = []
    min = []
    percentil25 = []
    percentil50 = []
    percentil75 = []
    max = []

    data = data[1:, :]

    for i in range(len(headers)):
        try:
            values = numpy.array(data[:, i], dtype=float)
            values = values[~numpy.isnan(values)]
            if len(values) == 0:
                raise Exception()
            count.append(count_(values))
            mean.append(mean_(values))
            std.append(stdDev_(values))
            min.append(min_(values))
            percentil25.append(percentil_(25, values))
            percentil50.append(percentil_(50, values))
            percentil75.append(percentil_(75, values))
            max.append(max_(values))
        except:
            pass

    numeric_headers = []
    for i in range(len(headers)):
        try:
            values = numpy.array(data[:, i], dtype=float)
            values = values[~numpy.isnan(values)]
            if i > 0:
                numeric_headers.append(headers[i])
        except:
            pass


    print_('', numeric_headers)
    print_('Count', count)
    print_('Mean', mean)
    print_('Std', std)
    print_('Min', min)
    print_('25%', percentil25)
    print_('50%', percentil50)
    print_('75%', percentil75)
    print_('Max', max)

def print_(label, list):
    print(f'{label:10.10}', end=' | ')
    for value in list:
        try:
            print(f'{value:>10.4f}', end=' | ')
        except:
            print(f'{str(value):>10.10}', end=' | ')
    print()

def count_(values):
    try:
        values = values.astype('float')
        values = values[~numpy.isnan(values)]
        return len(values)
    except:
        return len(values)

def mean_(values):
    try:
        values = values.astype('float')
        values = values[~numpy.isnan(values)]
        total = 0
        for i in values:
            total += i
        total /= len(values)
        return total
    except:
        return 0
    
def min_(values):
    min = values[0]
    for v in values:
        if v < min:
            min = v
    return min

def max_(values):
    max = values[0]
    for v in values:
        if v > max:
            max = v
    return max

def stdDev_(values):
    mean = mean_(values)

    variance = 0
    for v in values:
        diff = v - mean
        variance += diff * diff
    return(variance / len(values)) ** 0.5

def percentil_(percentil, value):
    value.sort()
    index = (len(value) - 1) * (percentil / 100)
    ceiling = numpy.ceil(index)
    floor = numpy.floor(index)

    if ceiling == floor:
        return value[index]
    
    i0 = value[int(ceiling)] * (index - floor)
    i1 = value[int(floor)] * (ceiling - index)

    return i0 + i1

def var_(values):
    values = values.astype('float')
    values = values[~numpy.isnan(values)]
    mean = mean_(values)
    total = 0
    for num in values:
        total = total + (num - mean) ** 2
    return total / len(values)

