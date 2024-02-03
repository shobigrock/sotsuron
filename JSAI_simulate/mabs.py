import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

"""
各種バンディットアルゴリズムのクラスを置くファイル
"""
class TreeBootstrap:
    # initialise values and raise input errors
    def __init__(self, n_actions, n_dims, tree=DecisionTreeClassifier(max_depth=2)):
        # 決定木はsklearnのDecisionTreeClassifier使用(CART分類木)
        # てっきり回帰木かと思ったが，これの.predict_proba()を利用して推定値を出力しているみたい

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.tree = tree
        self.D = [[[] for i in range(n_actions) ] for j in range(1) ]  # 行動が10種類あるので，10枠． -> 行動の種類によって柔軟に変えられるようにした方がいいかも
        self.r = [[0 for i in range(n_actions) ] for j in range(1) ]  # 最終的には[[0. ] [1. ] [1. ] [1. ] [0. ] [... [1. ]]のような形になってそう
        self.prob = np.zeros(self.n_actions)  # create zero array to save predicted probability from treeclassifier
        self.stopper = 0

        self.features = []
        self.thresholds = []
        self.values = []
        self.preds = []
        # feature = tree.feature
        # threshold = tree.threshold
        # value = tree.value  # 各クラスのデータ数


    # return the best arm
    def play(self, cnt, context):

        def vstack_for_bootstrap(older, newer):
            if len(older) == 0:
                return newer
            else:
                return np.vstack((older, newer))

        # 全ての行動について以下forループ
        for kaisuu, arm in enumerate(range(self.n_arms)):
            shaped_context = context         # 各行動に対応する文脈情報を取り出す

            # とりあえず各行動1回は実行してデータ(文脈, 報酬)を回収する
            if len(self.D[0][arm]) == 0:
                # set decision tree to predict 1 regardless of the input
                self.prob[arm] = 1.0  # predict 1
            else:
                # インデックス0を取ってくるだけで許されるのか？
                # 普通に許されなさそう．
                sample_context = self.D[0][arm]
                sample_reward = self.r[0][arm]

                # Bootstrapping
                # 「成功」と「失敗」が1つずつ存在しなければいけないので，最初の2回は先頭から2つを取る．残りは全体から選ぶ．
                b_context = np.vstack((sample_context[0], sample_context[1]))
                b_reward = np.vstack((sample_reward[0], sample_reward[1]))

                for i in range(len(sample_context)):
                    if i >= 2:
                        sampling_number = random.randint(0, len(sample_context)-1)
                        b_context = vstack_for_bootstrap(b_context, sample_context[sampling_number])
                        b_reward = vstack_for_bootstrap(b_reward, sample_reward[sampling_number])
                # Bootstrapping終了

                # tree = self.tree.fit(sample_context, sample_reward)
                tree = self.tree.fit(b_context, b_reward)          # train the tree classifier -> sample_からb_に変更
                temp_p = tree.predict_proba(shaped_context)      # predict the probability of the current context

                self.prob[arm] = temp_p[0][1]

        arm = break_tie(self.prob)  # [0.1, 0.01, 0.8, ...]行動ごと推定報酬値から

        return arm

    # update
    def update(context, action, reward):
        ss = 0
