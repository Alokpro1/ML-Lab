import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self) -> None:
        self._is_fitted = False

    def _pre_processor(self, X):
        _n_samples, _n_features = X.shape

        _X = np.zeros(shape=(_n_samples, _n_features+1))
        for i in range(_n_samples):
            _X[i][0] = 1
            for j in range(_n_features):
                _X[i][j+1] = X[i][j]

        return _X

    def _compute_weight_matrix(self):
        _part1 = np.dot(self._X.T, self._X)
        _part2 = np.dot(self._X.T, self._y)

        return np.dot(np.linalg.inv(_part1), _part2)

    def fit(self, X, y):
        self._n_samples, self._n_features = X.shape

        self._X = self._pre_processor(X)

        self._y = y.copy()

        self._weight_matrix = self._compute_weight_matrix()

        self._is_fitted = True

    def predict(self, X):
        assert (self._is_fitted ==
                True), 'ERROR: Please fit data before predictions'

        _X = self._pre_processor(X)

        return np.dot(_X, self._weight_matrix)


def train_test_split(data, target, train_ratio):
    _n_samples = data.shape[0]

    _train_size = int(train_ratio*_n_samples)

    X_train = data[:_train_size]
    X_test = data[_train_size:]
    y_train = target[:_train_size]
    y_test = target[_train_size:]

    return X_train, X_test, y_train, y_test


def preprocess_data(X, y):
    _n_samples, _n_features = X.shape

    _X_cleaned, _y_cleaned = [], []

    for ii in range(_n_samples):
        dirty_flag = False
        for jj in range(_n_features):
            if X[ii][jj] == '-':
                dirty_flag = True
        if y[ii] == 'T':
            dirty_flag = True

        if not dirty_flag:
            _X_cleaned.append(X[ii])
            _y_cleaned.append(y[ii])

    return np.array(_X_cleaned).astype('float64'), np.array(_y_cleaned).astype('float64')


def main():
    dataset = pd.read_csv("Assignment4\\weather.csv")

    data = dataset[['TempHighF', 'TempAvgF', 'TempLowF']].to_numpy()
    target = dataset['PrecipitationSumInches'].to_numpy()

    data, target = preprocess_data(data, target)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, train_ratio=0.8)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_predicted = regressor.predict(X_test)

    print("\nSum of square error for temp: {}".format(
        mean_squared_error(y_test, y_predicted)))


if __name__ == "__main__":
    main()
