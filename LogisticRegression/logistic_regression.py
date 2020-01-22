import numpy as np
import math

class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装

    Attributes
    ----------
    iter : int
        イテレーション数
    lr : float
        学習率
    no_bias : bool
        バイアス項を入れない場合はTrue
    verbose : bool
        学習過程を出力する場合はTrue
    lambda_ratio : float
        正則化パラメータ
    coef_ : 次の形のndarray, shape (n_features,)
        パラメータ
    values : 次の形のndarray, shape (n_class, )
        正解ラベルのユニーク数の配列
    loss : 次の形のndarray, shape (self.iter,)
        学習用データに対する損失の記録
    val_loss : 次の形のndarray, shape (self.iter,)
        検証用データに対する損失の記録

    """

    def __init__(self, num_iter=100, lr=0.01, no_bias=False, verbose=False, lambda_ratio=1):
        """
        Parameters
        ----------
        num_iter : int, default 100
            イテレーション数
        lr : float, default 0.01
            学習率
        no_bias : bool, default False
            バイアス項を入れない場合はTrue
        verbose : bool, default False
            学習過程を出力する場合はTrue
        lambda_ratio : float, default 1
            正則化パラメータ
        """

        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.lambda_ratio = lambda_ratio
        self.coef_ = None
        self.values = None
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        ロジスティック回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features), default None
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, ), default None
            検証用データの正解値
        """
        self.values = np.unique(y) # ラベルのユニーク値をインスタンス変数に保存
        assert len(self.values) == 2, "yが2値ではありません" # ラベルが2値出ない場合はアラート
        y = np.where(y==self.values[0], 0, 1) # 学習データのラベルを0 or 1の値に変換
        y_val = np.where(y_val==self.values[0], 0, 1) # 検証データのラベルを0 or 1の値に変換

        if not self.no_bias: # no_biasがFalseの場合はバイアス項を追加
            X = np.c_[np.ones(X.shape[0]), X]

            if (X_val is not None)&(y_val is not None): # valデータがある場合はX_valにもバイアス項を追加
                X_val = np.c_[np.ones(X_val.shape[0]), X_val]

        self.coef_ = np.random.RandomState(0).rand(X.shape[1]) # 重みの初期値をランダムに設定

        for i in range(self.iter):
            y_pred = self._logistic_hypothesis(X) # 仮定関数で予測値を算出
            error = y_pred - y # 正解値と予測値の差を算出
            self.loss[i] = self._get_loss(y_pred, y) # 目的関数でlossを算出

            if (X_val is not None)&(y_val is not None):
                y_val_pred = self._logistic_hypothesis(X_val) # 検証用データも仮定関数で予測値を算出
                val_error = y_val_pred - y_val # 正解値と予測値の差を算出
                self.val_loss[i] = self._get_loss(y_val_pred, y_val) # 目的関数でlossを算出
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("iter:{} train_loss:{}, val_loss:{}".format(i, self.loss[i], self.val_loss[i]))
            else:
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("iter:{} train_loss:{}".format(i, self.loss[i]))

            self._gradient_descent(X, error) # 最急降下法で重みを更新


    def predict(self, X):
        """
        ロジスティック回帰を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
        y_pred : 次の形のndarray, shape (n_samples, )
            ロジスティック回帰による推定結果(2値)
        """

        y_pred = self.predict_proba(X) # ロジスティック回帰で確率を推定

        for i in range(len(y_pred)):
            assert y_pred[i] >= 0 and y_pred[i] <= 1, "範囲外です"
            y_pred = np.where(y_pred<0.5, 0, 1) # 推定した確率が0.5未満なら0,以上なら1に変換

        y_pred = np.where(y_pred.astype(int)==0, self.values[0], self.values[1]) # 予測値0 or 1 を入力時の2値に再変換
        return y_pred

    def predict_proba(self, X):
        """
        ロジスティック回帰を使い確率を推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, )
            ロジスティック回帰による推定結果(確率)
        """

        if not self.no_bias: # no_biasがFalseの場合はバイアス項を追加
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = self._logistic_hypothesis(X)
        return y_pred

    def _logistic_hypothesis(self, X):
        """
        ロジスティック回帰の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
        h : 次の形のndarray, shape (n_samples, )
          シグモイドの仮定関数による推定結果

        """

        h = 1/(1 + math.e**(-1*(X@self.coef_)))
        return h

    def _gradient_descent(self, X, error):
        """
        最急降下法による重みの更新式

        Parameters
        ----------
        X：次の形のndarray, shape (n_samples, n_features)
          学習データ
        error：次の形のndarray, shape (n_samples, )
          正解値と予測値の差
        """

        self.coef_ = self.coef_ - self.lr*(1/X.shape[0]*(X.T@error) + self.lambda_ratio/\
                                           X.shape[0]*np.r_[np.array(0), self.coef_[1:]])

    def _get_loss(self, y_pred, y):
        """
        目的関数による計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        loss : numpy.float
          目的関数によるコストの算出
        """
        loss = 1/len(y)*sum((y*-1*np.log(y_pred) - (1-y)*np.log(1-y_pred))) + self.lambda_ratio/(2*len(y))*sum(self.coef_**2)
        return loss
