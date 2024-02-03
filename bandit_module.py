# Do not edit. These are the only imports permitted.
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier   # for Task 4
from sklearn.base import clone                    # optional for Task 4
import matplotlib.pyplot as plt                   # for Task 5
from sklearn.metrics.pairwise import rbf_kernel   # for Task 6
from sklearn.datasets import make_classification
import random

# MAB基本クラス
class MAB(ABC):
    """Base class for a contextual multi-armed bandit (MAB)

    Parameters
    ----------
    n_arms : int
        Number of arms.
    """
    # initialise and raise input errors
    def __init__(self, n_arms):
        if not type(n_arms)==int:
            raise TypeError("`n_arms` must be an integer")
        if not n_arms >= 0:
            raise ValueError("`n_arms` must be non-negative")
        self.n_arms = n_arms

    @abstractmethod
    # raise input errors
    def play(self, n_arms, n_dims, context):
        """Play a round

        Parameters
        ----------
        context : float numpy.ndarray, shape (n_arms, n_dims), optional
            An array of context vectors presented to the MAB. The 0-th
            axis indexes the arms, and the 1-st axis indexes the features.
            Non-contextual bandits accept a context of None.

        Returns
        -------
        arm : int
            Integer index of the arm played this round. Should be in the set
            {0, ..., n_arms - 1}.
        """
        if not type(context) == np.ndarray:
            raise TypeError("`context` must be numpy.ndarray")
        if not context.shape == (n_arms, n_dims):
            raise TypeError("`context` must have shape (n_arms, n_dims)")
        self.context = context


    @abstractmethod
    # raise input errors
    def update(self, arm, reward, n_arms, n_dims, context):
        """Update the internal state of the MAB after a play

        Parameters
        ----------
        arm : int
            Integer index of the played arm in the set {0, ..., n_arms - 1}.

        reward : float
            Reward received from the arm.

        context : float numpy.ndarray, shape (n_arms, n_dims), optional
            An array of context vectors that was presented to the MAB. The
            0-th axis indexes the arms, and the 1-st axis indexes the
            features. Non-contextual bandits accept a context of None.
        """
        if not (type(arm) == int or arm.dtype == 'int32'):  # 元は64だが，データが32で読み込まれているので，32にしてある
            raise TypeError("`arm` must be int type")
        if not (arm >= 0 and arm <= (n_arms-1)):
            raise ValueError("`arm` must be the the set {0, .., n_arms - 1}")
        if not (type(reward) == float or reward.dtype == 'float64'):
            raise TypeError("`reward` must be float type")
        if not (context.shape == (n_arms, n_dims) and context.dtype == 'float64') :
            raise TypeError("`context` must be float numpy in shape (n_events, n_arms, n_dims)")
        # get the values
        self.arm = arm
        self.reward = reward
        self.context = context

# 推定最適行動が複数ある場合，その中からランダムで行動を選ぶグローバル関数．
def break_tie(_range):
    indices = np.argwhere(_range == np.max(_range))
    index = np.random.randint(0,len(indices))

    return indices[index][0]

# オフ方策評価
def offlineEvaluate(mab, arms, rewards, contexts, n_arms, n_dims, n_events, n_rounds=None):
    """Offline evaluation of a multi-armed bandit

    Parameters
    ----------
    mab : instance of MAB
        MAB to evaluate.

    arms : integer numpy.ndarray, shape (n_events,)
        Array containing the history of pulled arms, represented as integer
        indices in the set {0, ..., mab.n_arms}

    rewards : float numpy.ndarray, shape (n_events,)
        Array containing the history of rewards.

    contexts : float numpy.ndarray, shape (n_events, n_arms, n_dims)
        Array containing the history of contexts presented to the arms.
        The 0-th axis indexes the events in the history, the 1-st axis
        indexes the arms and the 2-nd axis indexed the features.

    n_rounds : int, default=None
        Number of matching events to evaluate the MAB on. If None,
        continue evaluating until the historical events are exhausted.

    Returns
    -------
    out : float numpy.ndarray
        Rewards for the matching events.
    """
    # initialise values and raise input errors
    if not (arms.shape == (n_events,) and arms.dtype == 'int32')  :  # int32でデータが読み込まれてたので，条件をint64ではなくint32にした．
        print(arms.shape)
        print(n_events)
        print(arms.dtype)
        raise TypeError("`arms` must be integer numpy in shape (n_events,)")
    if not rewards.shape == (n_events,) and rewards.dtype == 'float64' :
        raise TypeError("`rewards` must be float numpy in shape (n_events,)")
    if not contexts.shape == (n_events,n_arms, n_dims) and rewards.dtype == 'float64' :
        raise TypeError("`contexts` must be float numpy in shape (n_events, n_arms, n_dims)")
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")

    n_round = 0     # データの行動=方策の行動となり，評価を行うことのできた回数(≠n_rounds: 設定する試行回数)
    R = []          # save the total payoff
    H = []          # save used historical events

    for i in range(n_events):
        if n_round == n_rounds:
            break
        arm = mab.play(contexts[i])
        if arm == arms[i]:                 # if historical data equals to chosen arm
            R.append(rewards[i])           # append the new rewards
            H.append([arms[i], rewards[i], contexts[i]])      # append the used events
            mab.update(arms[i], rewards[i], contexts[i])      # update the information
            n_round += 1

    # return rewards per play
    out = np.array(R)

    return out

# オフ方策評価（決定木の中身を確認できる）
# 詳細を見るために改造
def offlineEvaluate_forPrint(mab, arms, rewards, contexts, n_arms, n_dims, n_events, n_rounds=None):
    """Offline evaluation of a multi-armed bandit

    Parameters
    ----------
    mab : instance of MAB
        MAB to evaluate.

    arms : integer numpy.ndarray, shape (n_events,)
        Array containing the history of pulled arms, represented as integer
        indices in the set {0, ..., mab.n_arms}

    rewards : float numpy.ndarray, shape (n_events,)
        Array containing the history of rewards.

    contexts : float numpy.ndarray, shape (n_events, n_arms, n_dims)
        Array containing the history of contexts presented to the arms.
        The 0-th axis indexes the events in the history, the 1-st axis
        indexes the arms and the 2-nd axis indexed the features.

    n_rounds : int, default=None
        Number of matching events to evaluate the MAB on. If None,
        continue evaluating until the historical events are exhausted.

    Returns
    -------
    out : float numpy.ndarray
        Rewards for the matching events.
    """
    # initialise values and raise input errors
    if not (arms.shape == (n_events,) and arms.dtype == 'int32')  :  # int32でデータが読み込まれてたので，条件をint64ではなくint32にした．
        print(arms.shape)
        print(n_events)
        print(arms.dtype)
        raise TypeError("`arms` must be integer numpy in shape (n_events,)")
    if not rewards.shape == (n_events,) and rewards.dtype == 'float64' :
        raise TypeError("`rewards` must be float numpy in shape (n_events,)")
    if not contexts.shape == (n_events,n_arms, n_dims) and rewards.dtype == 'float64' :
        raise TypeError("`contexts` must be float numpy in shape (n_events, n_arms, n_dims)")
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")

    n_round = 0     # データの行動=方策の行動となり，評価を行うことのできた回数(≠n_rounds: 設定する試行回数)
    R = []          # save the total payoff
    H = []          # save used historical events

    for i in range(n_events):
        if n_round == n_rounds:
            break
        arm = mab.play(contexts[i], n_round)
        if arm == arms[i]:                 # if historical data equals to chosen arm
            R.append(rewards[i])           # append the new rewards
            H.append([arms[i], rewards[i], contexts[i]])      # append the used events
            mab.update(arms[i], rewards[i], contexts[i])      # update the information
            n_round += 1

    # return rewards per play
    out = np.array(R)

    return out
