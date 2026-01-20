import pandas as pd
import numpy as np
import argparse
from Helpers.describe_ import mean_, stdDev_
from sklearn.metrics import accuracy_score


def train_test_split(x, y, test_size=0.3, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
    
    n_elements = len(x)

    indices = np.random.permutation(n_elements)

    split_point = int(n_elements * (1 - test_size))

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return x_train, x_test, y_train, y_test

def encode_labels(y_train, y_test):
    unique_houses = np.unique(y_train)

    class_house = {house:i for i, house in enumerate(unique_houses)}
    y_train_encoded = np.array([class_house[house] for house in y_train])
    y_test_encoded = np.array([class_house[house] for house in y_test])
    return y_train_encoded, y_test_encoded, class_house

def standardize_features(x_train, x_test):
    means = []
    stds = []

    for col in range(x_train.shape[1]):
        feature_values = x_train[:, col]
        means.append(mean_(feature_values))
        stds.append(stdDev_(feature_values))
    
    means = np.array(means)
    stds = np.array(stds)

    # Zero standard deviation means no variation: When a feature has std = 0, all values are identical.
    stds[stds == 0] = 1

    x_train_scaled = np.zeros_like(x_train)
    for col in range(x_train.shape[1]):
        x_train_scaled[:, col] = ((x_train[:, col] - means[col]) / stds[col])

    x_test_scaled = np.zeros_like(x_test)
    for col in range(x_test.shape[1]):
        x_test_scaled[:, col] = ((x_test[:, col] - means[col]) / stds[col])

    return x_train_scaled, x_test_scaled, means, stds
    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(x, y, weights):
    
    m = len(y)

    h = sigmoid(np.dot(x, weights))

    epsilon = 1e-15 # A very small number (0.000000000000001)
    h = np.clip(h, epsilon,  1 - epsilon) # -> np.clip("value", "Minimum value", "Maximum value")

    cost = (-1/m) * np.sum((y * np.log(h) + (1 - y ) * np.log(1 - h))) # the "-" in (-1/m) is because we have to convert it into a positive a number
    return cost

def gradient_descent(x, y, learning_rate, iterations, weights):

    m = x.shape[0]
    cost_history = []
    for i in range(iterations):
        z = np.dot(x, weights)
        h = sigmoid(z)

        gradient = (1/m) * np.dot(x.T, (h - y))
        weights = weights - learning_rate * gradient

        if i % 100 == 0:
            cost = compute_cost(x, y, weights)
            cost_history.append(cost)
    return weights, cost_history

def gradient_descent_stochastic(x, y,learning_rate,  epochs, weights):
    m = x.shape[0]
    total_cost = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)

        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = x_shuffled[i:i +1]
            yi = y_shuffled[i:i + 1]

            z = np.dot(xi, weights)
            h = sigmoid(z)

            gradient = np.dot(xi.T, (h - yi))

            weights = weights - learning_rate * gradient

        if epoch % 10 == 0:
            cost = compute_cost(x, y, weights)
            total_cost.append(cost)
    return weights, total_cost

def gradient_descent_mini_batch(x, y, learning_rate, iterations, weights, batch_size=32):
    m = x.shape[0]
    total_cost = []

    for epoch in range(iterations):
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            xi = x_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]

            batch_m = len(yi)

            z = np.dot(xi, weights)
            h = sigmoid(z)

            gradient = (1/batch_m) * np.dot(xi.T, (h - yi))
            weights = weights - learning_rate * gradient
        
        if epoch % 10 == 0:
            cost = compute_cost(x, y, weights)
            total_cost.append(cost)
    return weights, total_cost
    
def train_one_vs_all(x_train_scaled, y_train_encoded, n_classes, learning_rate=0.1, iterations=1000, method='batch'):
    m = x_train_scaled.shape[0]

    print(method)

    x_with_bias = np.column_stack([np.ones(m), x_train_scaled])

    all_weights = []

    for house in range(n_classes):
        weights = np.zeros(x_with_bias.shape[1])

        y_train_encoded_binary = (y_train_encoded == house).astype(int)

        if method == 'batch':
            weights, cost_history = gradient_descent(x_with_bias, y_train_encoded_binary, learning_rate, iterations, weights)
        elif method == 'stochastic':
            weights, cost_history = gradient_descent_stochastic(x_with_bias, y_train_encoded_binary, learning_rate, iterations, weights)
        elif method == 'mini-batch':
            weights, cost_history = gradient_descent_mini_batch(x_with_bias, y_train_encoded_binary, learning_rate, iterations, weights, batch_size=32)
        all_weights.append(weights)
        print(f"Final cost: {cost_history[-1]:.6f}")

    print(all_weights)
    return np.array(all_weights)


def predict(x_test, weights):
    z = np.dot(x_test, weights.T)
    probabilities = sigmoid(z)
    print(probabilities.shape)
    predictions = np.argmax(probabilities, axis=1)
    print(predictions.shape)
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='logreg_train')
    parser.add_argument('dataset', type=str, help='Path to the csv file')
    parser.add_argument('--method', type=str, default='batch', choices=['batch', 'stochastic', 'mini-batch'] ,help="Method of training")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df_clean = df[['Hogwarts House', 'Astronomy', 'Herbology', 'Ancient Runes']].dropna()
    x = df_clean[['Astronomy', 'Herbology', 'Ancient Runes']].values
    houses = df_clean['Hogwarts House'].values

    x_train, x_test, y_train, y_test = train_test_split(x, houses, test_size=0.3, random_state=42)

    y_train_encoded,y_test_encoded, class_names = encode_labels(y_train, y_test)

    x_train_scaled, x_test_scaled, means, stds = standardize_features(x_train, x_test)

    n_classes = len(np.unique(y_train_encoded))

    weights = train_one_vs_all(x_train_scaled, y_train_encoded, n_classes, learning_rate=0.1, iterations=1000, method=args.method)

    x_test_scaled = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

    prediction = predict(x_test_scaled, weights)
    
    print(f'Accuracy: {accuracy_score(y_test_encoded, prediction):.2f}')

    np.savez('weights.npz', weights=weights, means=means, stds=stds, class_names=class_names)
    print(f"Weight saved! Shape: {weights.shape}")
