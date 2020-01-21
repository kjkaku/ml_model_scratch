from initializer import *
from optimizer import *
from activation import *
from other_class import *
import numpy as np

class FC():
    """
    ノード数n_nodes1からn_nodes2への全結合層

    Attributes
    ----------
    W : 次の形のndarray, shape (n_nodes1, n_nodes2)
        重みの配列
    B : 次の形のndarray, shape (n_nodes2, )
        バイアスの配列
    optimizer : instance
        最適化方法のインスタンス
    WH : 次の形のndarray, shape (n_nodes1, n_nodes2)
        最適化手法がAdaGradの場合の重みの勾配の2乗和
    BH : 次の形のndarray, shape (n_nodes2, )
        最適化手法がAdaGradの場合のバイアスの勾配の2乗和
    X : 次の形のndarray, shape (batch_size, n_nodes1)
        入力データ
    dW : 次の形のndarray, shape (n_nodes1, n_nodes2)
        Wに関する損失の勾配
    dB : 次の形のndarray, shape (1, n_nodes2)
        Bに関する損失の勾配
    """

    def __init__(self, n_nodes1, n_nodes2, lr, initializer, optimizer):
        """
        Parameters
        ----------
        n_nodes1 : int
            前の層のノード数
        n_nodes2 : int
            後の層のノード数
        lr : float
            学習率
        initializer : str
            初期化方法名
        optimizer : str
            最適化手法名
        """

        # 初期化インスタンスンスの辞書
        initializer_dic = {"gausu":SimpleInitializer,
                           "xavier":XavierInitializer,
                           "he":HeInitializer}
        # 最適化インスタンスの辞書
        optimizer_dic = {"sgd":SGD,
                         "adagrad": AdaGrad}
        # initializerのメソッドを使い、self.Wとself.Bを初期化する
        initializer = initializer_dic[initializer]() # initializerをインスタンス化
        self.W = initializer.W(n_nodes1, n_nodes2) # 重みを初期化
        self.B = initializer.B(n_nodes2) # バイアスを初期化

        self.optimizer = optimizer_dic[optimizer](lr=lr) # optimizerをインスタンス化
        if optimizer == "adagrad":
          # optimizerがAdaGradの場合はself.WHとself.BHを0で初期化
          self.WH = 0
          self.BH = 0

        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        """
        全結合層のフォワードプロパゲーション

        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, n_nodes1)
            入力データ

        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力データ
        """

        self.X = X
        A = self.X@self.W + self.B # fowardを計算し後ろに流す
        return A

    def backward(self, dA):
        """
        全結合層のバックワードプロパゲーション

        Parameters
        ----------
        dA : 次の形のndarray, shape (1, n_nodes2)
            後ろから流れてきた勾配

        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """

        self.dW = self.X.T @ dA # Wに関する損失の勾配
        self.dB = np.sum(dA, axis=0) # Bに関する損失の勾配
        dZ = dA @ self.W.T # Zに関する損失の勾配
        self = self.optimizer.update(self) # 重み、バイアスの更新
        return dZ
