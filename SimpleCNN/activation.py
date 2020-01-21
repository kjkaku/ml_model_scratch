import numpy as np

class Tanh():
    """
    ハイパボリックタンジェントによる活性化関数

    Attributes
    ----------
    A : 次の形のndarray, shape (batch_size, n_nodes1)
        入力データの配列
    """

    def forward(self, A):
        """
        フォワードプロパゲーションでの活性化関数

        Parameters
        ----------
        A : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            入力データ

        Returns
        ----------
        Z : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            活性化関数で計算した配列
        """
        self.A = A # 入力時の配列を保存
        Z = np.tanh(self.A) # ハイパボリックタンジェントによって計算
        return Z

    def backward(self, dZ):
        """
        バックワードプロパゲーションでの活性化関数

        Parameters
        ----------
        dZ : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            Zに関する損失の勾配

        Returns
        ----------
        dA : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            Aに関する損失の勾配
        """

        dA = dZ * (1 - np.tanh(self.A)**2) # 損失の勾配を計算
        return dA

class ReLU():
    """
    ReLUによる活性化関数

    Attributes
    ----------
    A : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
        入力データの配列
    """

    def forward(self, A):
        """
        フォワードプロパゲーションでの活性化関数

        Parameters
        ----------
        A : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            入力データの配列

        Returns
        ----------
        Z : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            活性化関数で計算した配列
        """

        self.A = A # 入力時の配列を保存
        Z = np.where(self.A>0, self.A, 0) # ReLU関数を通す
        return Z

    def backward(self, dZ):
        """
        バックワードプロパゲーションでの活性化関数

        Parameters
        ----------
        dZ : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            Zに関する損失の勾配

        Returns
        ----------
        dA : 次の形のndarray, shape (batch_size, in_ch, h, w) or (batch_size, n_nodes1)
            Aに関する損失の勾配
        """

        dA = dZ * np.where(self.A>0, 1, 0) # 損失の勾配を計算
        return dA

class Softmax():
    """
    Softmaxによる活性化関数

    Atributes
    ----------
    loss : float
        交差エントロピー誤差
    """

    def forward(self, A):
        """
        フォワードプロパゲーションでの活性化関数

        Parameters
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes1)
            全結合層を通って出力された配列

        Returns
        ----------
        Z : 次の形のndarray, shape (batch_size, n_output)
            活性化関数で計算した各ラベルの確率の配列
        """

        max_a = np.max(A) # 入力データの最大値を計算
        Z = np.exp(A-max_a) / np.sum(np.exp(A-max_a), axis=1).reshape(-1, 1) # 各ラベルの確率を計算
        return Z

    def backward(self, Z, Y):
        """
        バックワードプロパゲーションでの活性化関数

        Parameters
        ----------
        Z : 次の形のndarray, shape (batch_size, n_output)
            Softmax関数で計算した各ラベルの確率の配列
        Y : 次の形のndarray, shape (batch_size, n_output)
            正解ラベルの配列

        Returns
        ----------
        dA : 次の形のndarray, shape (batch_size), n_output)
             Aに関する損失の勾配
        """

        dA = (Z - Y)/Y.shape[0] # Aに関する損失の勾配を計算
        return dA
