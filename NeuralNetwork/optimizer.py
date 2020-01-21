import numpy as np

class SGD():
    """
    確率的勾配降下法

    Attributes
    ----------
    lr : float
        学習率
    """
    def __init__(self, lr):
        """
        Parameters
        ----------
        lr : 学習率
        """

        self.lr = lr

    def update(self, layer):
        """
        ある層の重みやバイアスの更新

        Parameters
        ----------
        layer : 更新前の層のインスタンス

        Returns
        ----------
        layer : 更新後の層のインスタンス
        """

        layer.W = layer.W - self.lr*layer.dW # インスタンスの重みを更新
        layer.B = layer.B - self.lr*layer.dB # インスタンスのバイアスを更新
        return layer

class AdaGrad():
    """
    AdaGradによる最適化方法

    Attributes
    ----------
    lr : 学習率
    """

    def __init__(self, lr):
        """
        Parameters
        ----------
        lr : float
            学習率
        """

        self.lr = lr

    def update(self, layer):
        """
        ある層の重みやバイアスの更新

        Parameters
        ----------
        layer : 更新前の層のインスタンス

        Returns
        ----------
        layer : 更新後の層のインスタンス
        """

        layer.WH = layer.WH + layer.dW*layer.dW # 前のイテレーションまでの重みの勾配の二乗和の更新
        layer.BH = layer.BH + layer.dB*layer.dB # 前のイテレーションまでのバイアスの勾配の二乗和の更新

        layer.W = layer.W - self.lr/(np.sqrt(layer.WH)+1e-7)*layer.dW # 重みの更新
        layer.B = layer.B - self.lr/(np.sqrt(layer.BH)+1e-7)*layer.dB # バイアスの更新
        return layer
