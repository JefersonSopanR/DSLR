import argparse
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def predict(x_test, weights):
    z = np.dot(x_test, weights.T)
    probabilities = sigmoid(z)
    print(probabilities.shape)
    predictions = np.argmax(probabilities, axis=1)
    print(predictions.shape)
    return predictions
    


if __name__ == '__main__':

    argparser = argparse.ArgumentParser("logreg_predict")
    argparser.add_argument("dataset", type=str, help="dataset_test.csv")
    argparser.add_argument("weights", type=str, help="weights.npz")

    args = argparser.parse_args()

    model = np.load(args.weights, allow_pickle=True)
    weights = model['weights']
    stds = model['stds']
    means = model['means']
    houses = model['class_names']

    df_test = pd.read_csv(args.dataset)

    features = ['Astronomy', 'Herbology', 'Ancient Runes']

    x_test = df_test[features].copy()

    for i, col in enumerate(features):
        x_test[col] = x_test[col].fillna(means[i])
    x_test = x_test.values

    # scaling data
    x_test_scaled = (x_test - means) / stds

    # add bias (column of 1s)
    x_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

    prediction = predict(x_with_bias, weights)

    house_dic = {0: 'Gryffindor', 1: 'Hufflepuff', 2:'Ravenclaw', 3: 'Slytherin'}

    predicted_houses = [house_dic[value] for value in prediction]


    output = pd.DataFrame({
        'Index': range(len(predicted_houses)),
        'Hogwarts House': predicted_houses
    })

    output.to_csv('houses.csv', index=False)

