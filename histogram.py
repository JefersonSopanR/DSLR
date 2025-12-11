from Helpers.core import getNumPyArray
import numpy
from Helpers.describe_ import stdDev_, var_
import matplotlib.pyplot as plt

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
    headers = data[0]
    data = data[1:, :]
    indices = data[:, 1].argsort()
    data = data[indices]

    best_col = find_best_col(data)
    print(best_col)
    houses = numpy.unique(data[:, 1])
    color_houses = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    
    # Plot histogram for each house
    # this creates a blank canvas figsize(x, y)
    plt.figure(figsize=(10, 6))

    for house in houses:
        house_data = data[data[:, 1] == house][:, best_col]
        house_data = numpy.array(house_data, dtype=float)
        house_data = house_data[~numpy.isnan(house_data)]
        # this Draws the histogram bars:
        plt.hist(house_data, alpha=0.5, label=house, bins=20)
        # alpha=0.5 - Make bars 50% transparent (so overlapping bars are visible)
        # label=house - Name for legend -> example ('Gryffindor')
        # bins=20 - Divide score range into 20 groups (blocks)
    
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Score Distribution - {headers[best_col]}')
    plt.legend()
    plt.show()
