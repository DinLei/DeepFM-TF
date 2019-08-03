#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: feature_meta.py
# @time: 2019-08-02 11:36


class FeatureMeta(object):

    def __init__(self, name, f_type, enum=[],
                 max_val=None, min_val=None,
                 avg_val=None, std_val=None, dist=None):
        """
        特征的元信息
        :param name: 特征名
        :param f_type: 特征类型：二值型:bin | 离散型: cat | 数值型: num
        :param enum: 只针对离散型，罗列该特征的所有取值，如：颜色 -> 红、蓝、绿
        :param max_val: 只针对数值型，特征最大值
        :param min_val: 只针对数值型，特征最小值
        :param avg_val: 只针对数值型，特征平均值
        :param std_val: 只针对数值型，特征标准差
        :param dist: 只针对离散型，特征的统计分布
        """
        self.name = name
        self.f_type = f_type
        self.enum = enum
        self.max_val = max_val
        self.min_val = min_val
        self.avg_val = avg_val
        self.std_val = std_val
        self.distribution = dist

    def max_min_scale(self, x):
        if self.min_val == self.max_val:
            return 0
        return (x - self.min_val) / (self.max_val - self.min_val)

    def norm_scale(self, x):
        return (x - self.avg_val) / self.std_val

    def info(self):
        return self.enum
