from Helpers.describe_ import stdDev_, mean_
from Helpers.core import getNumPyArray
import numpy
import matplotlib.pyplot as plt

def correlation_(values_x, values_y):
    mean_x = mean_(values_x)
    mean_y = mean_(values_y)
    std_x = stdDev_(values_x)
    std_y = stdDev_(values_y)
    n = len(values_x) # This is the same as -> len(values_y)

    numerator = sum(((values_x[i] - mean_x) * (values_y[i] - mean_y)) for i in range(n))
    denominator = n * std_x * std_y

    if denominator == 0:
        return 0
    return numerator / denominator

def find_best_correlations(data):
    best_pairs = (None, None)
    max_correlation = -1

    for i in range(6, data.shape[1]):
        try:
            X = numpy.array(data[:, i], dtype=float)
            for j in range(i + 1, data.shape[1]):
                y = numpy.array(data[:, j], dtype=float)

                valid_values = ~(numpy.isnan(X) | numpy.isnan(y))
                X_valid = X[valid_values]
                y_valid = y[valid_values]
                if len(X_valid) > 0:
                    correlation = correlation_(X_valid, y_valid)
                    
                    if abs(correlation) > max_correlation:
                        max_correlation = abs(correlation)
                        best_pairs = (i, j)
        except:
            continue
    return best_pairs, max_correlation

if __name__ == "__main__":
    data = getNumPyArray('dataset_train.csv')
    headers = data[0]
    data = data[1:, :]
    data = data[data[:, 1].argsort()]

    (col1, col2), max_corr = find_best_correlations(data)

    X = numpy.array(data[:, col1], dtype=float)
    y = numpy.array(data[:, col2], dtype=float)

    valid_values = ~(numpy.isnan(X) | numpy.isnan(y))

    X_clean = X[valid_values]
    y_clean = y[valid_values]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_clean, y_clean, alpha=0.5, s=30)
    
    plt.xlabel(headers[col1])
    plt.ylabel(headers[col2])
    plt.title(f'Scatter Plot: {headers[col1]} vs {headers[col2]}\nCorrelation: {max_corr:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.show()
