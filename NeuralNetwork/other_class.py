import numpy as np
from initializer import *
from optimizer import *

class GetMiniBatch():
    """
    ミニバッチを取得するイテレータ
    """
    def __init__(self, X, y, batch_size = 20, seed=0):
        """
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練用データ
        y : 次の形のndarray, shape (n_samples, 1)
            正解値
        batch_size : int
            バッチサイズ
        seed : int
            NumPyの乱数のシード
        """

        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)

    def __len__(self):
        return self._stop

    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self._X[p0:p1], self._y[p0:p1]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]
