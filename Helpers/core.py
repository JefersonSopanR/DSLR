import numpy
import csv

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
