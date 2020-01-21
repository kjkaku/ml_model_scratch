import numpy as np

class SimpleInitializer():
    """
    ガウス分布による重みとバイアスのシンプルな初期化

    Attributes
    ----------
    SIGMA : 定数0.01
        ガウス分布の標準偏差0.01
    """

    def __init__(self):
        self.SIGMA = 0.01

    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化

        Parameters
        ----------
        n_nodes1 : int
            前の層のノード数
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        W : 次の形のndarray, shape (n_nodes1, n_nodes2)
            初期化した重みの配列
        """

        W = self.SIGMA * np.random.randn(n_nodes1, n_nodes2) # ガウス分布標準偏差にランダムな値を掛けて初期化
        return W

    def B(self, n_nodes2):
        """
        バイアスの初期化

        Parameters
        ----------
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        B : 次の形のndarray, shape (n_nodes2, )
        　　初期化したバイアスの配列
        """

        B = np.random.randn(n_nodes2) # ランダムに初期化
        return B

class XavierInitializer():
    """
    Xavierの初期値による重みとバイアスの初期化

    Attributes
    ----------
    sigma : float
        Xavierの初期値における標準偏差
    """
    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化

        Parameters
        ----------
        n_nodes1 : int
            前の層のノード数
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        W : 次の形のndarray, shape (n_nodes1, n_nodes2)
            初期化した重みの配列
        """

        self.sigma = 1 / np.sqrt(n_nodes1) # # Xavierの初期値の標準偏差を計算
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2) # Xavierの初期値の標準偏差にランダムな値を掛けて初期化
        return W

    def B(self, n_nodes2):
        """
        バイアスの初期化

        Parameters
        ----------
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        B : 次の形のndarray, shape (n_nodes2, )
        　　初期化したバイアスの配列
        """

        B = self.sigma * np.random.randn(n_nodes2) # Xavierの初期値の標準偏差にランダムな値を掛けて初期化
        return B

class HeInitializer():
    """
  　Heの初期値による重みとバイアスの初期化

    Attributes
    ----------
    sigma : float
        Heの初期値における標準偏差
    """

    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化

        Parameters
        ----------
        n_nodes1 : int
            前の層のノード数
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        W : 次の形のndarray, shape (n_nodes1, n_nodes2)
            初期化した重みの配列
        """

        self.sigma = np.sqrt(2/n_nodes1) # Heの初期値の標準偏差を計算
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2) # Heの初期値の標準偏差にランダムな値を掛けて初期化
        return W

    def B(self, n_nodes2):
        """
        バイアスの初期化

        Parameters
        ----------
        n_nodes2 : int
            後の層のノード数

        Returns
        ----------
        B : 次の形のndarray, shape (n_nodes2, )
        　　初期化したバイアスの配列
        """

        B = self.sigma * np.random.randn(n_nodes2) # Heの初期値の標準偏差にランダムな値を掛けて初期化
        return B
