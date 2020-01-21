from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from layer import *
from activation import *
from other_class import *

class ScratchDeepNeuralNetworkClassifier():
    """
    シンプルなニューラルネットワーク分類器

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
    layers : tuple
        中間層の構成
    activation : instance
        活性化関数のインスタンス
    activation_list : list
        活性化関数のインスタンスを格納するリスト
    FC_list : list
        全結合層のインスタンスを格納するリスト
    train_loss_list : list
        ミニバッチごとの学習データの交差エントロピー誤差を格納するリスト
    train_acc_list : list
        epochごとの学習データのaccuracyを格納するリスト
    val_acc_list : list
        epochごとの検証データのaccuracyを格納するリスト
    """
    def __init__(self, epoch=10, lr=0.01, verbose=False, initializer="he", optimizer="sgd", activation="relu",
                 batch_size=20, n_layer=3, layers=(400, 200)):
        """
        Parameters
        ----------
        epoch : int, default 10
            訓練データを繰り返し学習する回数
        lr : float, default 0.01
            学習率
        vervose : bool, default False
            Trueの場合は学習過程を出力する
        initializer : str, default "he"
            重みとバイアスの初期化方法
        optimizer : str, default "adagrad"
            最適化法
        activation : str, default "relu"
            活性化関数
        batch_size : int, default 20
            ミニバッチサイズ
        n_layer : int, default 3
            層の数
        layers : tuple, default (400, 200)
            中間層の構成

        Note
        ----------
        layersの要素数はn_layerの値より1小さい値にするか、
        n_layerの値と同じ要素数の場合最後の要素はn_outputと同じ値でなければならない
        """

        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose
        self.initializer = initializer
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.n_layer = n_layer
        self.layers = layers

        # 活性化関数の辞書
        activation_dic = {"tanh":Tanh,
                          "relu":ReLU}
        self.activation = activation_dic[activation]
        self.activation_list = [] # 活性化関数のインスタンスを格納するリストを作成
        self.FC_list = [] # 全結合層のインスタンスを格納するリストを作成
        self.train_loss_list = [] # 学習データのepochごとの交差エントロピー誤差を格納するリストを作成
        self.train_acc_list = [] # 学習データのepochごとのaccuracyを格納するリストを作成
        self.val_acc_list = [] # 検証データのepochごとのaccuracyを格納するリストを作成

    def fit(self, X, y, X_val=None, y_val=None):
        """
        ニューラルネットワーク分類器を学習する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            訓練用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """

        n_features = X.shape[1] # 学習データの特徴量数
        n_output = np.unique(y).size # 正解ラベルの種類数を保存
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False) # OneHotするためのインスタンス
        y_train_one_hot = enc.fit_transform(y[:, np.newaxis]) # 学習データのラベルをOneHot化

        get_mini_batch = GetMiniBatch(X, y_train_one_hot, batch_size=self.batch_size) # 学習データをミニバッチ化

        self._append_instance_list(n_features, n_output) # 全結合層のインスタンスと活性化関数のインスタンスをリストに格納

        if self.verbose:
            print("train data learning process\n")

        for epoch in range(self.epoch): # epochの回数学習する
            for mini_X_train, mini_y_train in get_mini_batch: # ミニバッチごとにデータを学習する
                Z = self._forward(mini_X_train) # フォワードプロパゲーション
                self._backward(Z, mini_y_train) # バックプロパゲーションバックプロパゲーション
                loss = self._calc_loss(Z, mini_y_train) # ミニバッチごとのlossを算出
                self.train_loss_list.append(loss) # lossを記録
                if self.verbose:
                    # verboseがTrueの場合はlossを出力する
                    print("train_loss:{}".format(loss))

            train_acc = self._calc_acc(X, y) # epochごとの学習データのaccuracyを計算する
            self.train_acc_list.append(train_acc) # accuracyをリストに格納

            if (X_val is not None)&(y_val is not None):
                # 検証データがある場合はaccを計算する
                val_acc = self._calc_acc(X_val, y_val) # epochごとの検証データのaccuracyを計算する
                self.val_acc_list.append(val_acc) # accuracyをリストに格納

                if self.verbose:
                    # verboseがTrueの場合はepochごとのlossとaccを出力する
                    print("epoch:{}  train_loss:{:.8f}, train_acc:{:.8f}, val_acc:{:.8f}"\
                          .format(epoch, loss, train_acc, val_acc))
            else:
                if self.verbose:
                    # verboseがTrueの場合はepochごとのlossとaccを出力する
                    print("epoch:{}  train_loss:{:.8f}, train_acc:{:.8f}"\
                          .format(epoch, loss, train_acc))

    def _append_instance_list(self, n_features, n_output):
        """
        全結合層と活性化関数のインスタンスをリストに格納する

        Parameters
        ----------
        n_features : int
            入力データの特徴量数
        n_output : int
            出力のラベルのユニーク値の数
        """

        layer_list = [n_features] # 各層のnodeの数を格納するリストに入力の特徴量数を入れて作成
        if type(self.layers)==int:
            layer_list.append(self.layers) # 中間層のノード数が1つの場合はそれだけをリストに追加
        else:
            layer_list += list(self.layers) # 中間層が複数あるそのノード数をリストに追加
        layer_list.append(n_output) # 出力層の数(クラス数)をリストに追加

        for i in range(self.n_layer):
            # 各層の全結合層のインスタンスをリストに格納
            self.FC_list.append(FC(layer_list[i], layer_list[i+1], self.lr, self.initializer, self.optimizer))

        for _ in range(self.n_layer-1):
            # 各層の活性化関数のインスタンスをリストに格納
            self.activation_list.append(self.activation())
        self.activation_list.append(Softmax()) # 最後のSoftmax関数のインスタンスをリストに格納

    def _forward(self, X):
        """
        フォワードプロパゲーション

        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, n_features)
            サンプル

        Returns
        -------
        Z : 次の形のndarray, shape (batch_size, n_output)
            Softmax関数を通した後の配列
        """
        A_list = [] # 各層の全結合層の出力の配列を格納するリストを用意
        Z_list = [] # 各層の活性化関数を通した後の配列を格納するリストを用意
        for i in range(self.n_layer):
            if i==0:
              # 最初の層の入力には学習データの配列を入力し格納
              A_list.append(self.FC_list[i].forward(X))
            else:
              # 中間層の入力は前の層から流れてきた配列を入力し格納
              A_list.append(self.FC_list[i].forward(Z_list[-1]))
            Z_list.append(self.activation_list[i].forward(A_list[-1])) # 活性化関数を通した配列をリストに格納
        Z = Z_list[-1]
        return Z

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
        dZ_list = [] # Zに関する損失の勾配の配列を格納するリストを用意
        for i in range(1, self.n_layer+1):
            # バックプロパゲーション
            if i==1:
              # 最終層からの逆伝播の入力にはラベルを渡す
              dA_list.append(self.activation_list[-1].backward(Z, Y))
            else:
              # 中間層の逆伝播は後ろから流れてきた損失の勾配を渡す
              dA_list.append(self.activation_list[-i].backward(dZ_list[-1]))
            dZ_list.append(self.FC_list[-i].backward(dA_list[-1])) # Zに関する損失の勾配をリストに格納

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
        X : 次の形のndarray, shape (n_samples, n_features)
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
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル
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
