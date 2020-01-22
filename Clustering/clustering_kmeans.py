import numpy as np
import random

class ScratchKMeans():
    """
    K-meansのスクラッチ実装

    Attributes
    ----------
    n_clusters : int
        クラスタ数。
    n_init : int
        中心点の初期値を何回変えて計算するか。
    max_iter : int
        1回の計算で最大何イテレーションするか。
    tol : float
        イテレーションを終了する基準となる中心点と重心の許容誤差。
    verbose : bool
        学習過程を出力する場合はTrue。
    best_sse : float
        更新された重心位置での最小のSSEの値
    best_mu : 次の形のndarray, shape (n_clusters, n_features)
        SSEが最小になる場合の各クラスタの重心位置
    """

    def __init__(self, n_clusters=2, n_init=5, max_iter=100, tol=1e-5, verbose=False):
        """
        Parameters
        ----------
        n_clusters : int, default 2
            クラスタ数。
        n_init : int, default 5
            中心点の初期値を何回変えて計算するか。
        max_iter : int, default 100
            1回の計算で最大何イテレーションするか。
        tol : float, default 1e-5
            イテレーションを終了する基準となる中心点と重心の許容誤差。
        verbose : bool, default False
            学習過程を出力する場合はTrue。
        """

        # ハイパーパラメータを属性として記録
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.best_sse = None
        self.best_mu = None

    def fit(self, X):
        """
        K-meansによるクラスタリングを計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        """

        sse_array = np.empty(self.n_init) # n_initごとにsseの値を格納するarrayを用意
        mu_array = np.empty((self.n_init, self.n_clusters, X.shape[1])) # n_initごとにmuの値を格納するarrayを用意

        for n_init in range(self.n_init):
            random.seed(n_init)
            mu = X[random.sample(range(X.shape[0]), self.n_clusters)] # 初期値のmuをXのデータの中からランダムで選ぶ

            for iter in range(self.max_iter):
                distances = self._get_distance(X, mu) # 各クラスタの重心からの距離の配列を取得
                old_mu = mu.copy() # 古いmuの配列を保持
                mu = self._update_gravity(X, mu, distances) # muを更新する

                if np.mean(self._calc_distance(old_mu, mu)) <= self.tol:
                    # muの移動量が許容量以下の場合は反復終了
                    break

                sse = self._calc_sse(distances) # sseを算出
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    if iter==0:
                        print("\nn_init:{}".format(n_init))
                    print("iter:{}, SSE:{}".format(iter, sse))
            sse_array[n_init] = sse # n_initごとのsseをarrayに格納
            mu_array[n_init, :, :] = mu # n_initごとの更新されたmuをarrayに格納

        best_n_int = np.argmin(sse_array) # sseが最小になるn_initを選択
        self.best_sse = np.min(sse_array) # 最小のsseを保存
        if self.verbose:
            #verboseをTrueにした際は最小のsseを出力
            print("\nbest_SSE:{}\n".format(self.best_sse))
        self.best_mu = mu_array[best_n_int] # sseが最小になるmuの配列を保存

    def predict(self, X):
        """
        入力されたデータがどのクラスタに属するかを計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量

        Retuens
        ----------
        cluster : 次の形のndarray, shape (n_samples, )
            割り当てたクラスタの配列
        """

        distances = self._get_distance(X, self.best_mu) # 入力データと学習したmuの距離を計算
        cluster = self._make_cluster(distances) # 入力データにクラスタを割り当てる
        return cluster

    def _calc_sse(self, distances):
        """
        SSEを計算

        Retuens
        ----------
        sse : float
             計算された誤差平方和
        """

        distance = np.min(distances, axis=1) # 入力データと各クラスタのうち距離が最短になる値を選択
        sse = sum(distance**2) # 誤差平方根和を算出
        return sse

    def _get_distance(self, X, mu):
        """
        入力されたデータとmuの距離を計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            入力データの特徴量
        mu : 次の形のndarray, shape (n_clusters, n_features)
            各クラスタにおける重心の位置

        Retuens
        ----------
        diatances : 次の形のndarray, shape (n_samples, n_clusters)
            入力されたデータと重心muとの距離の配列
        """

        distances = np.empty((X.shape[0], self.n_clusters)) # 距離を格納する配列を用意
        for i in range(self.n_clusters):
            distances[:, i] = self._calc_distance(X, mu[i]) # 入力データに対する各クラスタごとの重心との距離を算出
        return distances

    def _calc_distance(self, q, p):
        """
        2点間のユークリッド距離を算出

        Parameters
        ----------
        q : 次の形のndarray, shape(n_clusters, n_features)
            地点qの配列
        p : 次の形のndarray, shape(n_clusters, n_features)
            地点pの配列

        Retuens
        ----------
        diatance : 次の形のndarray, shape (n_clusters, )
            2点間のユークリッド距離の配列
        """

        distance = np.linalg.norm(q-p, ord=2, axis=1) # 2点間のユークリッド距離を算出
        return distance

    def _make_cluster(self, distances):
        """
        クラスタを割り当て

        Parameters
        ----------
        distances : ユークリッド距離の配列

        Retuens
        ----------
        cluster : 次の形のndarray, shape (n_samples, )
            入力データに対し割り当てたクラスタの配列
        """

        cluster = np.argmin(distances, axis=1)  # 入力データに対しクラスタを割り当て
        return cluster

    def _update_gravity(self, X, mu, distances):
        """
        重心位置muの更新

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            入力データの特徴量
        mu : 次の形のndarray, shape (n_clusters, n_features)
            各クラスタにおける重心の位置

        Retuens
        ----------
        new_mu : 次の形のndarray, shape (n_clusters, n_features)
            更新された重心の位置
        """

        cluster = self._make_cluster(distances) # 入力データに対しクラスタを割り当て
        new_mu = np.empty_like(mu) # 更新後の重心の格納する配列を用意
        for i in range(self.n_clusters):
            new_mu[i] = np.mean(X[cluster==i], axis=0) # 各クラスタごとの重心muを更新
        return new_mu 
