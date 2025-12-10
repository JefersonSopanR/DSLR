from DSLR.describe_ import stdDev_, mean_, var_
from DSLR.core import getNumPyArray
import numpy

def correlation_(X, y):
    """Calculate Pearson correlation coefficient between X and y."""
    
    # Remove NaN values
    mask = ~(numpy.isnan(X) | numpy.isnan(y))

    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        return 0
    
    mean_x = mean_(X)
    mean_y = mean_(y)
    std_x = stdDev_(X)
    std_y = stdDev_(y)
    
    # Calculate correlation
    n = len(X)
    numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = n * std_x * std_y
    
    return numerator / denominator if denominator != 0 else 0

def find_most_similar_features(data):
    """Find the two most correlated features."""
    max_correlation = -1
    best_pair = (None, None)
    
    # Test all pairs of numeric columns (6-18)
    for i in range(6, data.shape[1]):
        for j in range(i + 1, data.shape[1]):  # Only test each pair once
            try:
                X = numpy.array(data[:, i], dtype=float)
                y = numpy.array(data[:, j], dtype=float)
          
                # Remove rows where either value is NaN
                mask = ~(numpy.isnan(X) | numpy.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) > 0:
                    # Calculate correlation coefficient
                    correlation = correlation_(X_clean, y_clean)
                    
                    # Track highest correlation
                    if abs(correlation) > max_correlation:
                        print(f'correlation->> {abs(correlation):12.4f}', end=' | ')
                        max_correlation = abs(correlation)
                        best_pair = (i, j)
                        print(f'best pair ->> {best_pair}')
                        
            except:
                continue
    
    return best_pair

if __name__ == "__main__":
    data = getNumPyArray('dataset_train.csv')
    data = data[1:, :]
    n_data = [80.0,     85.0,     90.0]
    new_data = numpy.array(n_data, dtype=float)
    std = stdDev_(new_data)
    mean = mean_(new_data)
    var = var_(new_data)
    col1, col2 = find_most_similar_features(data)
    print(f'col1 --> {col1}')
    print(f'col2 --> {col2}')
