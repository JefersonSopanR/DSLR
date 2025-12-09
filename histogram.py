from DSLR.core import getNumPyArray
import numpy
from DSLR.describe_ import stdDev_, var_

def find_best_col(data):
    houses = numpy.unique(data[:, 1])
    housesIndex = {}
    min_variance = float('inf')
    best_col = None

    start = 0
    for housename in sorted(houses):
        count = numpy.sum(data[:, 1] == housename)
        housesIndex[housename] = (start, start + count)
        start += count
    for col in range(6, data.shape[1]):
        try:
            col_data = numpy.array(data[:, col], dtype=float)

            stds = []
            for house in houses:
                start, end = housesIndex[house]
                house_data = col_data[start:end]
                house_data = house_data[~numpy.isnan(house_data)]
                if len(house_data) > 0:
                    stds.append(stdDev_(house_data))
            var = var_(numpy.array(stds, dtype=float))
            if var < min_variance:
                min_variance = var
                best_col = col
        except:
            continue


    return best_col


if __name__ == '__main__':
    data = getNumPyArray('dataset_train.csv')
    data = data[1:, :]
    indices = data[:, 1].argsort()
    data = data[indices]

    best_col = find_best_col(data)
    print(data[:, best_col])
