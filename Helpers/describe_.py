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

    # bonus
    ranges = []
    iqr = []
    skewness = []
    kurtosis = []
    coef_variation = []



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

            # bonus
            ranges.append(range_(values))
            iqr.append(iqr_(values))
            skewness.append(skewness_(values))
            kurtosis.append(kurtosis_(values))
            coef_variation.append(coef_variation_(values))
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
    print_('range', ranges)
    print_('iqr', iqr)
    print_('skewness', skewness)
    print_('kurtosis', kurtosis)
    print_('coef_variation', coef_variation)

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
        diff = num - mean
        total += diff ** 2
    return total / len(values)

# bonus
def range_(values):
    return max_(values) - min_(values)

def iqr_(values): #interquartile Range: Q3 - Q1
    q1 = percentil_(25, values)
    q3 = percentil_(75, values)
    return q3 - q1
#"IQR shows the range where the middle half of people scored.
# It ignores the top 25% and bottom 25%, so extreme scores don't mess up the picture."

def skewness_(values): # Skewness: measure of asymmetry of distribution
    values = values.astype('float')
    values = values[~numpy.isnan(values)]
    mean = mean_(values)
    std = stdDev_(values)
    n = len(values)

    if std == 0:
        return 0

    total = 0
    for v in values:
        diff = v - mean
        total += diff ** 3
    
    return (total/n) / (std ** 3)

# Example: [2, 3, 4, 7, 8, 9, 9, 9, 10, 10, 10, 10]
           #↑_____↑ (Few low values pulling mean left)

# Example: [1, 2, 3, 4, 5, 6, 7, 8, 9]
         #(Balanced on both sides)

# Example: [1, 1, 1, 1, 2, 2, 3, 4, 7, 10]
                              #↑_____↑ (Few high values pulling mean right)

def kurtosis_(values):
    values = values.astype('float')
    values = values[~numpy.isnan(values)]
    mean = mean_(values)
    std = stdDev_(values)
    n = len(values)

    if std == 0:
        return 0
    
    total = 0

    for v in values:
        diff = v - mean
        total += diff ** 4
    
    return (total/n) / (std ** 4) - 3

# UNIFORM (Kurtosis < 0)
 #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  → Kurtosis ≈ -1.2
  #**********

# NORMAL-ISH (Kurtosis ≈ 0)
#[3, 4, 5, 5, 5, 5, 6, 6, 7]      → Kurtosis ≈ 0
#*****

# PEAKED (Kurtosis > 0)
#[5, 5, 5, 5, 5, 5, 5, 1, 10]     → Kurtosis > 0

def coef_variation_(values):
    """Coefficient of Variation: (std/mean) * 100"""
    values = values.astype('float')
    values = values[~numpy.isnan(values)]
    mean = mean_(values)
    std = stdDev_(values)
    
    if mean == 0 or std == 0:
        return 0

    return (std / mean) * 100

#CV < 15%   →  Low variability    (very consistent)
     #****
    #******

#CV 15-30%  →  Moderate           (normal variation)
   #********

#CV > 30%   →  High variability   (inconsistent/risky)
#*     *     *

#CV > 100%  →  Extreme!            (std > mean)
#*               *
