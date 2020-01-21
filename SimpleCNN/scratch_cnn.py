from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from layer import *
from activation import *
from other_class import *

class ScratchCNN():
    """
    畳込みニューラルネットワーク分類器

    Attributes
    ----------
    epoch : int
        訓練データを繰り返し学習する回数
    lr : float
        学習率
    vervose : bool
        Trueの場合は学習過程を出力する
    initializer : str
        重みとバイアスの初期化方法
    optimizer : str
        最適化法
    batch_size : int
        ミニバッチサイズ
    n_layer : int
        層の数
    layers : tupple
        中間層の構成
    activation : instance
        活性化関数のインスタンス
    activation_list : list
        活性化関数のインスタンスを格納するリスト
    FC_list : list
        全結合層のインスタンスを格納するリスト
    train_loss_list : list
        学習データの交差エントロピー誤差を格納するリスト
    val_loss_list : list
        検証データの交差エントロピー誤差を格納するリスト
    train_acc_list : list
        epochごとの学習データのaccuracyを格納するリスト
    val_acc_list : list
        epochごとの検証データのaccuracyを格納するリスト
    """
    def __init__(self, epoch=1, lr=0.001, verbose=False, initializer="he", optimizer="adagrad", activation="relu",
                 batch_size=100, n_filters=30, hidden_nodes=100, filter_size=(3, 3), padding_size=(1, 1),
                 stride_size=(1, 1), pooling_filter_size=(2, 2)):
        """
        Parameters
        ----------
        epoch : int, default 1
            訓練データを繰り返し学習する回数
        lr : float, default 0.001
            学習率
        vervose : bool, default False
            Trueの場合は学習過程を出力する
        initializer : str, default "he"
            重みとバイアスの初期化方法
        optimizer : str, default "adagrad"
            最適化法
        activation : str, default "relu"
            活性化関数
        batch_size : int, default 100
            ミニバッチサイズ
        n_filters : int, default 30
            畳込層のフィルタ数
        hidden_nodes : int, default 100
            全結合層の出力ノード数
        filter_size : 次の形のtupple(Fh, Fw)
            畳込み層のフィルターサイズ
        padding_size : 次の形のtupple(Ph, Pw)
            畳込み層のゼロパディングのサイズ
        stride_size : 次の形のtupple(Sh, Sw)
            畳込み層のストライドサイズ
        """

        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose
        self.initializer = initializer
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.n_filters = n_filters
        self.hidden_nodes = hidden_nodes
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.stride_size = stride_size
        self.pooling_filter_size = pooling_filter_size

        # 活性化関数の辞書
        activation_dic = {"tanh":Tanh,
                          "relu":ReLU}
        self.activation = activation_dic[activation] # 活性化関数に使用するクラスを保存
        self.layer_list = [] # 各層と活性化関数のインスタンスを格納するリストを作成
        self.train_loss_list = [] # 学習データのepochごとの交差エントロピー誤差を格納するリストを作成
        self.train_acc_list = [] # 学習データのepochごとのaccuracyを格納するリストを作成
        self.val_acc_list = [] # 検証データのepochごとのaccuracyを格納するリストを作成

    def fit(self, X, y, X_val=None, y_val=None):
        """
        CNNで画像を学習する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, ch, h, w)
            訓練用の画像データ
        y : 次の形のndarray, shape (n_samples, )
            訓練用データの正解値
        X_val : 次の形のndarray, shape (n_samples, ch, h, w)
            検証用の画像データ
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """

        n_output = np.unique(y).size # 正解ラベルの種類数を保存
        _, in_ch, Xh, Xw = X.shape

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False) # OneHotするためのインスタンス
        y_train_one_hot = enc.fit_transform(y[:, np.newaxis]) # 学習データのラベルをOneHot化

        get_mini_batch = GetMiniBatch(X, y_train_one_hot, batch_size=self.batch_size) # 学習データをミニバッチ化

        self._append_instance_list(in_ch=in_ch, out_ch=self.n_filters, hidden_nodes=self.hidden_nodes,
                                   n_output=n_output, Xh=Xh, Xw=Xw) # 全結合層のインスタンスと活性化関数のインスタンスをリストに格納

        if self.verbose:
            print("train data learning process\n")

        for epoch in range(self.epoch): # epochの回数学習する
            for mini_X_train, mini_y_train in get_mini_batch: # ミニバッチごとにデータを学習する
                Z = self._forward(mini_X_train) # フォワードプロパゲーション
                self._backward(Z, mini_y_train) # バックプロパゲーション
                loss = self._calc_loss(Z, mini_y_train) # ミニバッチごとのlossを算出
                self.train_loss_list.append(loss) # lossを記録
                if self.verbose:
                    # verboseがTrueの場合はlossを出力する
                    print("train_loss:{}".format(loss))

            train_acc = self._calc_acc(X, y) # epochごとの学習データのaccuracyを計算する
            self.train_acc_list.append(train_acc) # accuracyをリストに格納

            if (X_val is not None)&(y_val is not None):
                # 検証データがある場合はそちらもaccを記録する
                val_acc = self._calc_acc(X_val, y_val) # epochごとの検証データのaccuracyを計算する
                self.val_acc_list.append(val_acc) # accuracyをリストに格納

                if self.verbose:
                    # verboseがTrueの場合はlossとaccを出力する
                    print("epoch:{}  train_loss:{:.8f}, train_acc:{:.8f}, val_acc:{:.8f}"\
                          .format(epoch, loss, train_acc, val_acc))
            else:
                if self.verbose:
                    # verboseがTrueの場合はlossとaccを出力する
                    print("epoch:{}  train_loss:{:.8f}, train_acc:{:.8f}"\
                          .format(epoch, loss, train_acc))

    def _append_instance_list(self, in_ch, out_ch, hidden_nodes, n_output, Xh, Xw):
        """
        各層と活性化関数のインスタンスをリストに格納する

        Parameters
        ----------
        in_ch : int
            畳込み層の入力チャネル数
        out_ch : int
            畳込み層の出力チャネル数
        hidden_nodes : int
            全結合層の出力ノード数
        n_output : int
            出力のラベルのユニーク値の数
        Xh : int
            入力データの行サイズ
        Xw : int
            入力データの列サイズ
        """

        # 各層のインスタンスをリストに格納していく
        # Conv Relu Maxpooling Flatten FC Relu FC Softmax
        self.layer_list.append(SimpleConv2d(initializer=self.initializer, optimizer=self.optimizer, lr=self.lr, in_ch=in_ch, out_ch=out_ch,
                                            filter_size=self.filter_size, padding_size=self.padding_size, stride_size=self.stride_size))
        out_h, out_w = self._calc_out_size(self.filter_size, self.padding_size, self.stride_size, Xh, Xw) # Conv後の出力サイズを算出

        self.layer_list.append(self.activation())
        self.layer_list.append(MaxPool2D(filter_size=self.pooling_filter_size, padding_size=(0, 0), stride_size=self.pooling_filter_size))
        out_h, out_w = self._calc_out_size(filter_size=self.pooling_filter_size, padding_size=(0, 0),
                                            stride_size=self.pooling_filter_size, Xh=out_h, Xw=out_w) # pooling層後の出力サイズを算出
        self.layer_list.append(Flatten())

        self.layer_list.append(FC(n_nodes1=out_ch*out_h*out_w, n_nodes2=hidden_nodes,
                                  lr=self.lr, initializer=self.initializer, optimizer=self.optimizer))
        self.layer_list.append(self.activation())
        self.layer_list.append(FC(n_nodes1=hidden_nodes, n_nodes2=n_output,
                                  lr=self.lr, initializer=self.initializer, optimizer=self.optimizer))
        self.layer_list.append(Softmax())

    def _forward(self, X):
        """
        フォワードプロパゲーション

        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, in_ch, h, w)
            サンプル

        Returns
        -------
        A : 次の形のndarray, shape (batch_size, n_output)
            Softmax関数を通した後の配列
        """

        A_list = [] # 各層の全結合層の出力の配列を格納するリストを用意
        for i in range(len(self.layer_list)):
            if i==0:
                # 最初の層の入力には学習データの配列を入力し格納
                A_list.append(self.layer_list[i].forward(X))
            else:
                # 2層目以降の入力は前の層から流れてきた配列を入力し格納
                A_list.append(self.layer_list[i].forward(A_list[-1]))

        A = A_list[-1]
        return A

    def _backward(self, Z, Y):
        """
        バックプロパゲーション

        Parameters
        ----------
        Z : 次の形のndarray, shape (batch_size, n_output)
            活性化関数を通した後の配列
        Y : 次の形のndarray, shape (batch_size, n_output)
            入力データに対する正解ラベル
        """

        dA_list = [] # Aに関する損失の勾配の配列を格納するリストを用意
        for i in range(1, len(self.layer_list)+1):
            # バックプロパゲーション
            if i==1:
                # 最終層からの逆伝播の入力にはラベルも渡す
                dA_list.append(self.layer_list[-1].backward(Z, Y))
            else:
                # それ以降の逆伝播は後ろから流れてきた損失の勾配を渡す
                dA_list.append(self.layer_list[-i].backward(dA_list[-1]))

    def _calc_out_size(self, filter_size, padding_size, stride_size, Xh, Xw):
        """
        畳込み層とプーリング層の後の画像サイズを算出

        Parameters
        ----------
        filter_size : 次の形のtupple(Fh, Fw)
            畳込み層のフィルターサイズ
        padding_size : 次の形のtupple(Ph, Pw)
            畳込み層のゼロパディングのサイズ
        stride_size : 次の形のtupple(Sh, Sw)
            畳込み層のストライドサイズ
        Xh : int
            入力時の行サイズ
        Xw : int
            入力時の列サイズ

        Returns
        ----------
        Nh_out : int
            層の後の行サイズ
        Nw_out : int
            層の後の列サイズ
        """

        get_out_size = GetOutSize(filter_size=filter_size, padding_size=padding_size, stride_size=stride_size)
        Nh_out, Nw_out = get_out_size.calc_size(Nh_in=Xh, Nw_in=Xw)
        return Nh_out, Nw_out

    def _calc_loss(self, Z, Y):
        """
        交差エントロピー誤差を計算

        Parameters
        ----------
        Z : 次の形のndarray, shape (batch_size, n_output)
            活性化関数を通した後の配列
        Y : 次の形のndarray, shape (batch_size, n_output)
            入力データに対する正解ラベル

        Retuens
        ----------
        loss : float
            交差エントロピー誤差
        """

        loss = -1 * np.mean(np.sum(Y * np.log(Z+1e-7), axis=1)) # 交差エントロピー誤差を計算
        return loss

    def _calc_acc(self, X, y):
        """
        accuracyを計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, in_ch, h, w)
            入力データ
        y : 次の形のndarray, shape (n_samples, )
            入力データに対する正解ラベル

        Returns
        ----------
        acc : float
            入力データに対するaccuracy
        """

        y_pred = self.predict(X) # 予測
        acc = metrics.accuracy_score(y, y_pred) # accuracyを計算
        return acc

    def predict(self, X, batch_size=100):
        """
        ニューラルネットワーク分類器を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, in_ch, h, w)
            画像データ
        batch_size : int, default 100
            予測する際のバッチサイズ

        Returns
        -------
        y_pred : 次の形のndarray, shape (n_samples, )
            推定結果
        """
        num = 0
        y_pred = np.empty(X.shape[0])
        while num < X.shape[0]:
            # Xのデータを全て予測するまで繰り返す
            mini_X = X[num:num+batch_size] # Xをバッチサイズにスライス
            mini_y_pred = self._forward(mini_X) # Softmax関数を通した後の配列を取り出す
            mini_y_pred = np.argmax(mini_y_pred, axis=1) # 確率からラベルを推定
            y_pred[num:num+batch_size] = mini_y_pred
            num += batch_size

        return y_pred
