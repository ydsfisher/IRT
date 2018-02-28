# 双参数二级积分模型进行参数估计

# first define a basic class
from __future__ import print_function, division
import numpy as np
import warnings

class BaseIrt(object):
    def __init__(self, scores=None):
        self.scores = scores

    @staticmethod
    def p(z):
        # 回答正确的概率函数
        e = np.exp(z)
        p = e / (1.0 + e)
        return p

class Irt2PL(BaseIrt):
    ''
    @staticmethod
    def z(slop, threshold, theta):
        # z函数
        _z = slop * theta + threshold
        _z[_z > 35] = 35
        _z[_z < -35] = -35
        return _z
# 由于theta alpha均为未知，可以采用EM算法（当然，也可以用MCMC算法），把theta当缺失数据。E步，计算
# theta下当样本分布（人数）和答对试题对样本量分布（人数），M步，极大似然求解a和b的值。

class Irt2PL(BaseIrt):
    # EM solution
    def __init__(self, init_slop=None, init_threshold=None, max_iter=10000, tol=1e-5,
                 gp_size=11, m_step_method='newton', *args, **kwargs):
        """
        :param init_slop:斜率初值
        :param init_threshold:阈值初值
        :param max_iter:EM算法最大迭代次数
        :param tol:精度
        :param gp_size: Gauss-Hermite 积分点数
        """
        super(Irt2PL, self).__init__(*args, **kwargs)
        # 斜率初值
        if init_slop is not None:
            self._init_slop = init_slop
        else:
            self._init_slop = np.ones(self.scores.shape[1])
        # 阈值初值
        if init_threshold is not None:
            self._init_threshold = init_threshold
        else:
            self._init_threshold = np.zeros(self.scores.shape[1])
        self._max_iter = max_iter
        self._tol = tol
        self._m_step_method = '_{0}'.format(m_step_method)
        self.x_nodes, self.x_weights = self.get_gh_point(gp_size)
        

