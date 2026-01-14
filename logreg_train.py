import pandas as pd
import numpy as np
import argparse


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
    return y_train_encoded, y_test_encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='logreg_train')
    parser.add_argument('dataset', type=str, help='Path to the csv file')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df_clean = df[['Hogwarts House', 'Astronomy', 'Herbology', 'Ancient Runes']].dropna()
    x = df_clean[['Astronomy', 'Herbology', 'Ancient Runes']].values
    houses = df_clean['Hogwarts House'].values

    x_train, x_test, y_train, y_test = train_test_split(x, houses, test_size=0.3, random_state=42)

    y_train_encoded,y_test_encoded = encode_labels(y_train, y_test)


    

    
    