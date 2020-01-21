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
        instance = MakeInstance()
        initializer = instance.initialize(initializer) # initializerをインスタンス化
        self.W = initializer.W(n_nodes1, n_nodes2) # 重みを初期化
        self.B = initializer.B(n_nodes2) # バイアスを初期化

        self.optimizer = instance.optimize(optimizer, lr) # optimizerをインスタンス化
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

class SimpleConv2d():
    """
    畳込層

    Attributes
    ----------
    batch_size : int
        ミニバッチサイズ
    optimizer : str
        最適化方法名
    W : 次の形のndarray, shape(n_filters, n_ch, Fh, Fw)
        重みの配列
    B : 次の形のndarray, shape(n_filters, )
        バイアスの配列
    WH : 次の形のndarray, shape (n_filters, n_ch, Fh, Fw)
        最適化手法がAdaGradの場合の重みの勾配の2乗和
    BH : 次の形のndarray, shape (n_filters, )
        最適化手法がAdaGradの場合のバイアスの勾配の2乗和
    dW : 次の形のndarray, shape ((n_filters, n_ch, Fh, Fw)
        Wに関する損失の勾配
    dB : 次の形のndarray, shape (n_filters, )
        Bに関する損失の勾配
    X : 次の形のndarray, shape(batch_size, n_ch, h, w)
        入力データ
    indexes : 次の形のndarray, shape(h_iter*w_iter, Fh*Fw)
        スライドウィンドウを動かして切り取ったインデックスの配列
    indexes_X : 次の形のndarray, shape(batch_size, n_ch, h_iter*w_iter, Fh*Fw)
        indexesに沿って切り取ったXの配列
    """

    def __init__(self, initializer, optimizer, lr, in_ch, out_ch,
                 filter_size, padding_size, stride_size):
        """
        Parameters
        ----------
        initializer : str, default
            重みとバイアスの初期化方法名
        optimizer : str
            最適化方法名
        in_ch : int
            入力チャネル数
        out_ch : int
            出力チャネル数(フィルタ枚数)
        filter_size : 次の形のtupple(Fh, Fw)
            フィルターサイズ
        padding_size : 次の形のtupple(Ph, Pw)
            ゼロパディングのサイズ
        stride_size : 次の形のtupple(Sh, Sw)
            ストライドサイズ
        """
        instance = MakeInstance()
        initializer = instance.initialize(initializer) # initializerをインスタンス化
        self.optimizer = instance.optimize(optimizer, lr) # optimizerをインスタンス化

        if optimizer == "adagrad":
            # optimizerがAdaGradの場合はself.WHとself.BHを0で初期化
            self.WH = 0
            self.BH = 0

        self.W = initializer.W(in_ch, out_ch, filter_size[0], filter_size[1]) # 重みを初期化
        self.B = initializer.B(out_ch) # バイアスを初期化

        self.dB = None
        self.dW = None
        self.indexes = None
        self.indexes_X = None
        self.X = None

        self.get_out_size = GetOutSize(filter_size=filter_size, padding_size=padding_size, stride_size=stride_size)
        self.make_slide_window = MakeSlideWindow(filter_size=filter_size, stride_size=stride_size)
        self.padding = Padding(padding_size=padding_size)

    def forward(self, X):
        """
        Parameters
        ----------
        X : 次の形のndarray, shape(batch_size, in_ch, h, w)
            入力データ

        Returns
        ----------
        A : 次の形のndarray, shape(batch_size, out_ch, out_h, out_w)
            出力データ
        """

        out_h, out_w = self.get_out_size.calc_size(Nh_in=X.shape[2], Nw_in=X.shape[3]) # 出力時の行列サイズを算出

        self.X = self.padding.forward(X) # 入力データをパディング

        batch_size, in_ch, Xh, Xw = self.X.shape # パディング後のshapeを保存
        out_ch = self.W.shape[0] # 出力チャネルを保存

        self.indexes = self.make_slide_window.make_array(h_iter=out_h, w_iter=out_w, Xw=Xw) # スライドウィンドウを動かしたときのインデックスの配列を作成

        self.indexes_X = np.take(self.X.reshape(batch_size, in_ch, Xh*Xw), self.indexes, axis=2) # 作成したインデックスの配列に沿ってXを切り取る配列を作成
        self.indexes_X = self.indexes_X.transpose(0, 2, 1, 3) # 軸を入れ替える
        self.indexes_X = self.indexes_X.reshape(self.indexes_X.shape[0]*self.indexes_X.shape[1], -1)

        A = self.indexes_X @ self.W.reshape(out_ch, -1).T # 出力データを計算
        A = A.reshape(batch_size, out_ch, out_h, out_w) # データを整形
        A = A + self.B[np.newaxis, :, np.newaxis, np.newaxis] # バイアスを加える
        return A

    def backward(self, dA):
        """
        Parameters
        ----------
        dA : 次の形のndarray, shape(batch_size, out_ch, out_h, out_w)
            入力データ

        Returns
        ----------
        dX : 次の形のndarray, shape(batch_size, in_ch, h, w)
            入力データ
        """

        batch_size, in_ch, Xh, Xw = self.X.shape
        out_ch, _, Fh, Fw = self.W.shape
        out_h, out_w = dA.shape[2], dA.shape[3]

        self.dB = dA.sum(axis=(0, 2, 3)) # Bに対する損失の勾配を計算
        self.dW = self.indexes_X.T @ dA.transpose(0,2,3,1).reshape(-1, out_ch) # Wに対する損失の勾配を計算
        self.dW = self.dW.T.reshape(out_ch, in_ch, Fh, Fw) # 整形

        W_flat = self.W.reshape(1, out_ch, in_ch, -1) # 重みの行,列を平滑化
        dA_flat = dA.reshape(batch_size, out_ch, 1, -1, 1) # dAにin_chの次元と最後に計算用の次元を増やし行, 列を平滑化
        dX = np.zeros((batch_size, in_ch, Xh*Xw)) # dXを代入するための配列を用意

        for i in range(out_h*out_w):
            # 逆伝播した勾配を代入していく
            W_array = np.sum(dA_flat[:, :, :, i, :]*W_flat, axis=1)
            dX[:, :, self.indexes[i]] +=  W_array

        dX = dX.reshape(self.X.shape) # 入力時のデータに整形
        dX = self.padding.backward(dX) # パディングを削除したデータに戻す

        self = self.optimizer.update(self)  # 重みとバイアスを更新
        return dX

class MaxPool2D():
    """
    Max Pooling層のクラス

    Attributes
    ----------
    filter_size : tupple
        フィルタの行列サイズのタプル
    X_shape : tupple
        入力データのshape
    max_array : 次の形のndarray, shape(batch_size, ch, h_iter*w_iter, 1)
        スライドウィンドウ内の最大インデックスの位置を格納する配列
    """

    def __init__(self, filter_size=(2, 2), padding_size=(0, 0), stride_size=(2, 2)):
        """
        Parameters
        ----------
        filter_size : tupple, default (2, 2)
            フィルタの行列サイズのタプル
        padding_size : tupple, default (0, 0)
            パディングの行列サイズのタプル
        stride_size : tupple, default (2, 2)
            ストライドの行列サイズのタプル
        """

        self.filter_size = filter_size
        self.X_shape = None
        self.max_array = None

        self.get_out_size = GetOutSize(filter_size=filter_size, padding_size=padding_size, stride_size=stride_size)
        self.make_slide_window = MakeSlideWindow(filter_size=filter_size, stride_size=stride_size)

    def forward(self, X):
        """
        フォワードプロパゲーション

        Parameters
        ----------
        X : 次の形のndarray, shape(batch_size, ch, h, w)
            入力データの配列

        Returns
        ----------
        A : 次の形のndarray, shape(batch_size, ch, out_h, out_w)
            出力データの配列
        """

        self.X_shape = X.shape # 入力データのshapeを保存
        batch_size, ch, Xh, Xw = X.shape

        out_h, out_w = self.get_out_size.calc_size(Nh_in=Xh, Nw_in=Xw) # 出力時の行列サイズを算出
        indexes = self.make_slide_window.make_array(h_iter=out_h, w_iter=out_w, Xw=Xw) # スライドウィンドウを動かしたときのインデックスの配列を作成

        indexes_X = np.take(X.reshape(batch_size, ch, Xh*Xw), indexes, axis=2) # 入力データをスライスした配列を作成
        A = np.max(indexes_X, axis=3) # ウィンドウ内の最大値を算出
        self.max_array = np.expand_dims(np.argmax(indexes_X, axis=3), axis=3) # 最大値があったインデックスに軸を増やして保存
        A = A.reshape(batch_size, ch, out_h, out_w) # 整形

        return A


    def backward(self, dA):
        """
        バックプロパゲーション

        Parameters
        ----------
        dA : 次の形のndarray, shape(batch_size, ch, out_h, out_w)
            後ろから流れてきたデータの配列

        Returns
        ----------
        dX : 次の形のndarray, shape(batch_size, ch, h, w)
            前に流すデータの配列
        """

        dA = dA.reshape(-1) # データを平滑化
        max_array = self.max_array.reshape(-1) # 最大値があったインデックスの配列も平滑化
        arange_array = np.arange(len(max_array)) # スライシングするためにmax_colsの要素数を並べた配列を作成
        dX = np.zeros(self.X_shape).reshape(self.filter_size[0]*self.filter_size[1], -1).T # 出力データを代入するための配列を作成
        dX[arange_array, max_array] = dA # スライシングして代入する
        dX = dX.reshape(self.X_shape) # 整形する
        return dX

class Flatten():
    """
    平滑化するためのクラス

    Attributes
    ----------
    X_shape : tupple
        入力時のXの配列のshape
    """

    def forward(self, X):
        """
        フォワードプロパゲーション

        Parameters
        ----------
        X : 次の形のndarray, shape(batch_size, in_ch, h, w)
            入力データ

        Returns
        ----------
        flatten_X : 次の形のndarray, shape(batch_size, in_ch*h*w)
            平滑化したデータ
        """
        self.X_shape = X.shape # 入力データのshapeを保存
        flatten_X = X.reshape(X.shape[0], -1) # データを平滑化
        return flatten_X

    def backward(self, flatten_dX):
        """
        バックプロパゲーション

        Parameters
        ----------
        flatten_dX : 次の形のndarray, shape(batch_size, in_ch*h*w)
            後ろから流れてきたデータ

        Returns
        ----------
        dX : 次の形のndarray, shape(batch_size, in_ch, h, w)
            元の形に戻したデータ
        """
        dX = flatten_dX.reshape(self.X_shape) # 入力時のデータに整形
        return dX
