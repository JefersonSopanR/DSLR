import csv
import numpy

def describe(filename):
    data = getNumPyArray(filename)
    headers = data[0]

    data = data[1:, :]


    for i in range(1):
        print(headers[i], end=' | ')
        try:
            values = numpy.array(data[:, i], dtype=float)
            values = values[~numpy.isnan(values)]
            if not values.any():
                raise Exception()
            print(f'{count_(values):>12.4f}', end=" | ")
        except:
            print("something went wrong!")




def getNumPyArray(filename):
    datalist = list()

    try:
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for _ in reader:
                row = list()
                for value in _:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = numpy.nan
                    row.append(value)
                datalist.append(row)
    except:
        print(f"Error opening tye file: {filename}")
    return numpy.array(datalist, dtype=object)

def count_(values):
    try:
        values = values.astype('float')
        values = values[~numpy.isnan(values)]
        return len(values) + 10
    except:
        print("something went wrong! in count_")
        return len(values)

