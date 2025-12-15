import matplotlib.pyplot as plt
import numpy
from Helpers.core import getNumPyArray

if __name__ == "__main__":
    data = getNumPyArray('dataset_train.csv')
    headers = data[0]
    data = data[1:, :]
    
    houses = data[:, 1]
    house_colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 
                    'Ravenclaw': 'blue', 'Slytherin': 'green'}
    
    # Select numeric columns (6 onwards for courses)
    feature_cols = range(6, data.shape[1])
    num_features = len(feature_cols)
    
    # Create grid of subplots
    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))
    
    for i, col_i in enumerate(feature_cols):
        for j, col_j in enumerate(feature_cols):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: histogram
                for house in numpy.unique(houses):
                    house_data = numpy.array(data[houses == house][:, col_i], dtype=float)
                    house_data = house_data[~numpy.isnan(house_data)]
                    ax.hist(house_data, alpha=0.5, label=house, 
                           color=house_colors.get(house, 'gray'), bins=20)
            else:  # Off-diagonal: scatter plot
                for house in numpy.unique(houses):
                    house_mask = houses == house
                    x = numpy.array(data[house_mask][:, col_j], dtype=float)
                    y = numpy.array(data[house_mask][:, col_i], dtype=float)
                    
                    valid = ~(numpy.isnan(x) | numpy.isnan(y))
                    ax.scatter(x[valid], y[valid], alpha=0.3, s=1,
                              color=house_colors.get(house, 'gray'))
            
            # Labels only on edges
            if i == num_features - 1:  # Last row
                ax.set_xlabel(headers[col_j], fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
                
            if j == 0:  # First column
                ax.set_ylabel(headers[col_i], fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    
    # Add legend
    handles = [plt.Line2D(
        [0],              # x-coordinates (dummy, not shown)
        [0],              # y-coordinates (dummy, not shown)
        marker='o',       # Shape: circle
        color='w',        # Line color: white (invisible)
        markerfacecolor=color,  # Fill color of the circle
        markersize=8      # Size of the circle
    ) for color in ['red', 'yellow', 'blue', 'green']]
    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    plt.tight_layout()
    plt.show()