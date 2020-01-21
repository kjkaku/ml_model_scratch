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

    def W(self, in_ch, out_ch, Fh=None, Fw=None):
        """
        重みの初期化

        Parameters
        ----------
        in_ch : int
            前の層のチャンネル数
        out_ch : int
            後の層のチャンネル数
        Fh : int, default None
            フィルタの行方向のサイズ、全結合層の場合は平滑化されているためNone
        Fw : int, default None
            フィルタの列方向のサイズ、全結合層の場合は平滑化されているためNone

        Returns
        ----------
        W : 次の形のndarray, shape (out_ch, in_ch, Fh, Fw) or (in_ch, out_ch)
            初期化した重みの配列
        """
        if (Fh is None) & (Fw is None):
            W = self.SIGMA * np.random.randn(in_ch, out_ch) # ガウス分布標準偏差にランダムな値を掛けて初期化(全結合層)
        else:
            W = self.SIGMA * np.random.randn(out_ch, in_ch, Fh, Fw) # ガウス分布標準偏差にランダムな値を掛けて初期化(畳込層)
        return W

    def B(self, out_ch):
        """
        バイアスの初期化

        Parameters
        ----------
        out_ch : int
            後の層のチャンネル数

        Returns
        ----------
        B : 次の形のndarray, shape (out_ch, )
        　　初期化したバイアスの配列
        """

        B = np.random.randn(out_ch) # ランダムに初期化
        return B

class XavierInitializer():
    """
    Xavierの初期値による重みとバイアスの初期化

    Attributes
    ----------
    sigma : float
        Xavierの初期値における標準偏差
    """
    def W(self, in_ch, out_ch, Fh=None, Fw=None):
        """
        重みの初期化

        Parameters
        ----------
        in_ch : int
            前の層のチャンネル数
        out_ch : int
            後の層のチャンネル数
        Fh : int, default None
            フィルタの行方向のサイズ、全結合層の場合は平滑化されているためNone
        Fw : int, default None
            フィルタの列方向のサイズ、全結合層の場合は平滑化されているためNone

        Returns
        ----------
        W : 次の形のndarray, shape (out_ch, in_ch, Fh, Fw) or (in_ch, out_ch)
            初期化した重みの配列
        """

        self.sigma = 1 / np.sqrt(in_ch) # Xavierの初期値の標準偏差を計算
        if (Fh is None) & (Fw is None):
            W = self.sigma * np.random.randn(in_ch, out_ch) # Xavierの初期値の標準偏差にランダムな値を掛けて初期化(全結合層)
        else:
            W = self.sigma * np.random.randn(out_ch, in_ch, Fh, Fw) # Xavierの初期値の布標準偏差にランダムな値を掛けて初期化(畳込層)
        return W

    def B(self, out_ch):
        """
        バイアスの初期化

        Parameters
        ----------
        out_ch : int
            後の層のチャンネル数

        Returns
        ----------
        B : 次の形のndarray, shape (out_ch, )
        　　初期化したバイアスの配列
        """

        B = self.sigma * np.random.randn(out_ch) # Xavierの初期値の標準偏差にランダムな値を掛けて初期化
        return B

class HeInitializer():
    """
  　Heの初期値による重みとバイアスの初期化

    Attributes
    ----------
    sigma : float
        Heの初期値における標準偏差
    """

    def W(self, in_ch, out_ch, Fh=None, Fw=None):
        """
        重みの初期化

        Parameters
        ----------
        in_ch : int
            前の層のチャンネル数
        out_ch : int
            後の層のチャンネル数
        Fh : int, default None
            フィルタの行方向のサイズ、全結合層の場合は平滑化されているためNone
        Fw : int, default None
            フィルタの列方向のサイズ、全結合層の場合は平滑化されているためNone

        Returns
        ----------
        W : 次の形のndarray, shape (out_ch, in_ch, Fh, Fw) or (in_ch, out_ch)
            初期化した重みの配列
        """

        self.sigma = np.sqrt(2/in_ch) # Heの初期値の標準偏差を計算
        if (Fh is None) & (Fw is None):
            W = self.sigma * np.random.randn(in_ch, out_ch) # Heの初期値の標準偏差にランダムな値を掛けて初期化(全結合層)
        else:
            W = self.sigma * np.random.randn(out_ch, in_ch, Fh, Fw) # Heの初期値の布標準偏差にランダムな値を掛けて初期化(畳込層)
        return W

    def B(self, out_ch):
        """
        バイアスの初期化

        Parameters
        ----------
        out_ch : int
            後の層のチャンネル数

        Returns
        ----------
        B : 次の形のndarray, shape (out_ch, )
        　　初期化したバイアスの配列
        """

        B = self.sigma * np.random.randn(out_ch) # Heの初期値の標準偏差にランダムな値を掛けて初期化
        return B
