import numpy as np

class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
        パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
        学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
        検証用データに対する損失の記録
    """

    def __init__(self, num_iter, lr=0.01, no_bias=False, verbose=False):
        """
        Parameters
        ----------
        num_iter : int
            イテレーション数
        lr : float, default 0.01
            学習率
        no_bias : bool, default False
            バイアス項を入れない場合はTrue
        verbose : bool , default False
            学習過程を出力する場合はTrue
        """

        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.coef_ = None
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        if not self.no_bias: # no_biasがFalseの場合はバイアス項を追加
            X = np.c_[np.ones(X.shape[0]), X]

            if (X_val is not None)&(y_val is not None): # valデータがある場合はX_valにもバイアス項を追加
                X_val = np.c_[np.ones(X_val.shape[0]), X_val]

        np.random.seed(42)
        self.coef_ = np.random.rand(X.shape[1]) # self.coef_の初期値をランダムに設定

        for i in range(self.iter): # iterの回数を繰り返す
            y_pred = self._linear_hypothesis(X) # 仮定関数で予測値を計算
            error = y_pred - y # 正解値と予測値の差を算出
            self.loss[i] = self.calc_mse(y_pred, y) # 平均二乗誤差を算出して格納

            if (X_val is not None)&(y_val is not None):
                y_val_pred = self._linear_hypothesis(X_val) # 検証用データも仮定関数で予測値を計算
                val_error = y_val_pred - y_val # 正解値と予測値の差を算出
                self.val_loss[i] = self.calc_mse(y_val_pred, y_val) # 平均二乗誤差を算出して格納
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("iter:{} train_mse:{:.8f}, val_mse:{:.8f}".format(i, self.loss[i], self.val_loss[i]))
            else:
                #verboseをTrueにした際は学習過程を出力
                print("iter:{} train_mse:{:.8f}".format(i, self.loss[i]))

            self._gradient_descent(X, error) # 最急降下法でパラメータを更新


    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        ----------
        y_pred : 次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        if not self.no_bias: # no_biasがFalseの場合はバイアス項を追加
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = self._linear_hypothesis(X) # 線形回帰を使い推定
        return y_pred

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習データ

        Returns
        -------
        y_pred : 次の形のndarray, shape (n_samples, 1)
            線形の仮定関数による推定結果
        """

        y_pred = X@self.coef_
        return y_pred

    def _gradient_descent(self, X, error):
        """
        最急降下法による更新式

        Parameters
        ----------
        X：次の形のndarray, shape (n_samples, n_features)
            学習データ
        error：次の形のndarray, shape (n_samples, 1)
            正解値と予測値の差

        """
        self.coef_ = self.coef_ - (self.lr / X.shape[0] * (X.T@error)) # パラメータを更新

    def calc_mse(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
            推定した値
        y : 次の形のndarray, shape (n_samples,)
            正解値

        Returns
        ----------
        mse : numpy.float
            平均二乗誤差
        """

        mse = 1/len(y)/2 * sum((y_pred - y)**2)
        return mse
