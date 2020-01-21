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

class MakeInstance():
    """
    initializerとoptimizerのインスタンスを生成するクラス
    """
    def initialize(self, initializer):
        """
        Parameters
        ----------
        initializer : str
            重みとバイアスの初期化方法

        Retuens
        ----------
        initializer_instance : instance
            初期化するインスタンス
        """

        # 初期化インスタンスンスの辞書
        initializer_dic = {"gausu":SimpleInitializer,
                           "xavier":XavierInitializer,
                           "he":HeInitializer}

        initializer_instance = initializer_dic[initializer]() # initializerをインスタンス化
        return initializer_instance

    def optimize(self, optimizer, lr):
        """
        Parameters
        ----------
        optimizer : str
            最適化方法名
        lr : float
            学習率

        Retuens
        ----------
        optimizer_instance : instance
            重みとバイアスを最適化するインスタンス
        """
        # 最適化インスタンスの辞書
        optimizer_dic = {"sgd":SGD,
                         "adagrad": AdaGrad}

        optimizer_instance = optimizer_dic[optimizer](lr=lr) # optimizerをインスタンス化
        return optimizer_instance

class Padding():
    """
    ゼロパディングするクラス

    Attributes
    ----------
    Ph : int
        パディングの行サイズ
    Pw : int
        パディングの列サイズ
    """

    def __init__(self, padding_size):
        """
        Parameters
        ----------
        padding_size : tupple
            パディングの行列サイズ
        """

        self.Ph, self.Pw = padding_size

    def forward(self, X):
        """
        フォワード

        Parameters
        ----------
        X : 次の形のndarray, shape(batch_size, ch, h, w)
            入力データ

        Returns
        ----------
        padding_X : 次の形のndarray, shape(batch_size, ch, h+(Ph*2), w+(Pw*2))
            ゼロパディングしたあとのデータ
        """

        padding_X = np.pad(X, [(0, 0), (0, 0), (self.Ph, self.Ph), (self.Pw, self.Pw)], 'constant') # ゼロパディング
        return padding_X

    def backward(self, dX):
        """
        バックワード

        Parameters
        ----------
        dX : 次の形のndarray, shape(batch_size, ch, h, w)
            入力データ

        Returns
        ----------
        nonpadding_dX : 次の形のndarray, shape(batch_size, ch, h-(Ph*2), w-(Pw*2))
            ゼロパディングする前の形に戻したデータ
        """

        nonpadding_dX = dX[:, :, self.Ph:-1*self.Ph, self.Pw:-1*self.Pw] # 入力時にパディングした場合はその分をスライスして削除
        return nonpadding_dX

class GetOutSize():
    """
    畳込み層とプーリング層で出力データの行列サイズを計算するクラス

    Attributes
    ----------
    Fh : int
        フィルタの行サイズ
    Fw : int
        フィルタの列サイズ
    Ph : int
        パディングの行サイズ
    Pw : int
        パディングの列サイズ
    Sh : int
        ストライドの行サイズ
    Sw : int
        ストライドの列サイズ

    """
    def __init__(self, filter_size, padding_size, stride_size):
        """
        Parameters
        ----------
        filter_size : tupple
            フィルタの行列サイズのタプル
        padding_size : tupple
            パディングの行列サイズのタプル
        stride_size : tupple
            ストライドの行列サイズのタプル
        """

        self.Fh, self.Fw = filter_size # フィルタの行列サイズを保存
        self.Ph, self.Pw = padding_size # パディングの行列サイズを保存
        self.Sh, self.Sw = stride_size # ストライドの行列サイズを保存

    def calc_size(self, Nh_in, Nw_in):
        """
        畳込層の出力サイズの計算

        Parameters
        ----------
        Nh_in : int
            入力時の行方向の画像サイズ
        Nw_in : int
            入力時の列方向の画像サイズ

        Returns
        ----------
        Nh_out : int
            出力時の行方向の画像サイズ
        Nw_out : int
            出力時の列方向の画像サイズ
        """
        Nh_out = int((Nh_in + 2*self.Ph - self.Fh)/self.Sh + 1) # 行方向の出力サイズを計算
        Nw_out = int((Nw_in + 2*self.Pw - self.Fw)/self.Sw + 1) # 列方向の出力サイズを計算
        return Nh_out, Nw_out

class MakeSlideWindow():
    """
    畳込層とプーリング層で使用するスライドウィンドウを動かしたときの
    データをスライスするインデックスを作るクラス

    Attributes
    ----------
    Fh : int
        フィルタの行サイズ
    Fw : int
        フィルタの列サイズ
    Sh : int
        ストライドの行サイズ
    Sw : int
        ストライドの列サイズ
    """

    def __init__(self, filter_size, stride_size):
        """
        Parameters
        ----------
        filter_size : tupple
            フィルタの行, 列のサイズ
        stride_size : tupple
            ストライドの行, 列のサイズ
        """

        self.Fh, self.Fw = filter_size # フィルタの行列サイズを保存
        self.Sh, self.Sw = stride_size # ストライドの行列サイズを保存

    def make_array(self, h_iter, w_iter, Xw):
        """
        Parameters
        ----------

        h_iter : int
            行方向にウィンドウを動かす回数
        w_iter : int
            列方向にウィンドウを動かす回数
        Xw : int
            入力データの列数

        Returns
        ----------
        indexes : 次の形のndarray, shape(h_iter*w_iter, Fh*Fw)
            スライドウィンドウを動かして切り取ったインデックスの配列

        Note
        ----------
        入力データX
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

        を平滑化したインデックス番号を取得する

        filter_size(3, 3), stride_size(1, 1)の場合
        return は
        [[0, 1, 2, 4, 5, 6, 8, 9, 10],
         [1, 2, 3, 5, 6, 7, 9, 10, 11],
         [4, 5, 6, 8, 9, 10, 12, 13, 14],
         [5, 6, 7, 9, 10, 11, 13, 14, 15]]
        となる
        """

        # 初回のスライドウィンドウで切り取るインデックスを取得
        indexes_w = (np.arange(self.Fw)+((np.arange(self.Fh)*Xw).reshape(-1, 1))).reshape(1, -1)
        indexes_h = np.arange(h_iter * w_iter) # スライドする回数の枠を作成
        indexes_h = (indexes_h*self.Sw + (indexes_h//w_iter*Xw*(self.Sh-1))\
                     + indexes_h//w_iter*(self.Fw-self.Sw)).reshape(-1, 1) # 繰り返す際のウィンドウの左上の数値をそれぞれ計算
        indexes = indexes_w + indexes_h # 切り取るインデックスの配列を取得
        return indexes
