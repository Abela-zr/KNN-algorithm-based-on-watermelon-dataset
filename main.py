
# -*- coding:utf-8 -*-
# Author: 非鱼子焉
# Creation_time: 2020.11.11
# Content: 基于西瓜数据集的KNN算法实现
# Blog: https://zhu-rui.blog.csdn.net/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# KNN算法类
class KNN(object):
    def __init__(self, x, y, K):  # x:密度；y:含糖率；k:近邻数
        self.x = x
        self.y = y
        self.K = K
        self.n = len(x)

    # 计算距离
    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # 算法实现
    def knn(self, x):
        distance = []
        for i in range(self.n):
            dist = self.distance(x, self.x[i])
            distance.append([self.x[i], self.y[i], dist])
        distance.sort(key=lambda x: x[2])
        neighbors = []
        neighbors_labels = []
        for k in range(self.K):
            neighbors.append(distance[k][0])  # 近邻具体数据
            neighbors_labels.append(distance[k][1])  # 近邻标记
        return neighbors, neighbors_labels

    # 选择多数投票数
    def vote(self, x):
        neighbors, neighbors_labels = self.knn(x)
        vote = {}  # 投票法
        for label in neighbors_labels:
            vote[label] = vote.get(label, 0) + 1
        sort_vote = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        return sort_vote[0][0]  # 返回投票数最多的标记

    # 对应标记
    def fit(self):
        labels = []
        for sample in self.x:
            label = self.vote(sample)
            labels.append(label)
        return labels  # 返回所有样本的标记

    # 计算正确率
    def accuracy(self):
        predict_labels = self.fit()
        real_labels = self.y
        correct = 0
        for predict, real in zip(predict_labels, real_labels):
            if int(predict) == int(real):
                correct += 1
        return correct / self.n


# 读取数据
def getdata(path):
    dataSet = pd.read_csv(path, delimiter=",")
    X = dataSet[['density', 'sugar_rate']].values
    Y = dataSet['label']
    return X, Y


# 进行绘图
def drawpictures(x_positive, y_positive, x_negative, y_negative):
    plt.scatter(x_positive, y_positive, marker='o', color='red', label='1')
    plt.scatter(x_negative, y_negative, marker='o', color='blue', label='0')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    plt.show()


# 训练数据
def train(X, Y):
    for k in range(1, 9):
        print("*****第%d次*****" % k)
        print('本次knn的k值选取为{}'.format(k))
        knn = KNN(X, Y, k)
        predict = knn.fit()
        print('本次knn的正确率为{}'.format(knn.accuracy()))
        x_positive = []
        y_positive = []
        x_negative = []
        y_negative = []
        for i in range(len(X)):
            if int(predict[i]) == 1:
                x_positive.append(X[i][0])
                y_positive.append(X[i][1])
            else:
                x_negative.append(X[i][0])
                y_negative.append(X[i][1])
        drawpictures(x_positive, y_positive, x_negative, y_negative)


if __name__ == '__main__':
    X, Y = getdata('watermelon3_0a.csv')
    train(X, Y)
    print("************程序运行结束************")
