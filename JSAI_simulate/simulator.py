import numpy as np
import pandas as pd

# クラス継承の概念が難しいが，理解している時間は無いので別でコードを作成
"""
simulator:データの行数だけシミュレーションを行い，報酬リストを返す関数
    input:
        df: d次元dataframe
            context:
                if isArtifical then context = df
                else context = df.iloc[:, :-1]
            reward:
                if isArtifical then ここで作る
                else context = df.iloc[:, -1]
        mab: 各種バンディットアルゴリズム
            mab.play(context):
                output: action
            mab.update(reward):
                output: なし．パラメータ更新
        isArtificial: 人工データか否か
            True:人工データ⇒決定木から報酬作成
            False:UCIデータ⇒最後の列を報酬として見なす
    output:
        報酬リスト(長さ: dfの行数)
"""

"""
与えられたデータ量だけシミュレーションを行い，報酬のリストを返す．
"""
def simulator(df, mab, isArtificial):
    if isArtificial:
        contexts = df
    else:  # 右端1列が正答
        contexts = df.iloc[:, :-1]
        ans = df.iloc[:, -1]
    reward_list = []

    # シミュレーション
    for cnt, i in enumerate(contexts.iterrows()):
        context = i[1]
        action = mab.play(cnt, context)
        if isArtificial:
            reward = calc_reward(context, action)
        else:
            reward = 1 if ans == action else 0
        mab.update(context, action, reward)
        reward_list.append(reward)

    return reward_list

# 使用する人工データに応じて中身が変化するのが望ましい?
def calc_reward(context, action):
    threshold = [[5,5,5],
                [3,3,3],
                [5,5,5]]  # threshold[閾値idx][行動idx]で利用

    reward_list = [[[.90,.10], [.25,.75], [.70,.30], [.20,.80]],  # 低収入である確率
                [[.80,.20], [.85,.15], [.60,.40], [.85,.15]],  # 中収入である確率
                [[.30,.70], [.90,.10], [.70,.30], [.95,.05]]]  # 高収入である確率

    if context[0] >= threshold[0][action]:
        if context[1] >= threshold[1][action]:
            reward = np.random.choice([0,1], p = reward_list[action][0])
        else:
            reward = np.random.choice([0,1], p = reward_list[action][1])
    else:
        if context[2] >= threshold[2][action]:
            reward = np.random.choice([0,1], p = reward_list[action][2])
        else:
            reward = np.random.choice([0,1], p = reward_list[action][3])
    return reward
