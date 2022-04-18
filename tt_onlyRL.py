# -*- coding: utf-8 -*-
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import newRes
# import resNetAssign

totalNum = 320
random.seed(10)


class newqLearning:
    # # 命中率
    levelAccuracy = np.array([0.98, 0.95, 0.90, 0.85, 0.75, 0.55, 0.30, 0.10, 0.0])
    # # 状态定义
    state = {i: j for i, j in zip(levelAccuracy, [k for k in range(len(levelAccuracy))])}
    # # 动作定义
    edgeCompensate = np.array([i for i in range(int(totalNum / 2))], dtype=int)
    # # Q表
    rewardTable = np.zeros((len(state), int(totalNum / 2)))
    # # 探索的概率
    rateExplore = None
    currentState = None
    currentAction = None
    # 预测模型文件
    resModel = None
    acc_li = []
    batch = None
    cache_node = None

    # 初始化
    def __init__(self):
        self.batch = 1000
        self.cache_node = 10
        self.rateExplore = 0.2
        self.currentState = self.state.get(self.levelAccuracy[8])
        # self.currentAction = random.choice(self.edgeCompensate)
        # self.currentAction = self.edgeCompensate[-1]

    # 状态更新
    def updateState(self, acc):
        assert acc <= 1, 'Error:acc value error!'
        for i in self.levelAccuracy:
            if acc >= i:
                acc = i
                break
        return self.state[acc]

    # 计算奖励值
    def caculateReward(self, next_state, boolDown):
        # # 如果没有从后台下载
        if not boolDown:
            reward = 2 * (totalNum / 2 - self.currentAction) * (10 - next_state)
        else:
            reward = -2 * (self.currentAction) * (next_state)
        # # 奖励值更新
        print(self.currentState, self.currentAction)
        qsa_1 = self.rewardTable[self.currentState, self.currentAction]
        qsa_2 = self.rewardTable[next_state, self.rewardTable[next_state].argmax()]
        # self.rewardTable[self.currentState, self.currentAction] = 0.5 * qsa_1 + 0.5 * (reward + 0.5 * qsa_2)
        self.rewardTable[:, self.currentAction] = 0.5 * qsa_1 + 0.5 * (reward + 0.5 * qsa_2)

    # # 动作选择
    def makeAct(self):
        if random.random() < self.rateExplore:
            self.currentAction = random.choice(self.edgeCompensate)
        else:
            self.currentAction = self.edgeCompensate[self.rewardTable[self.currentState].argmax()]

    # # 由当前动作预测每个缓存的编号
    def predict(self):
        pred = []
        for i in range(10):
            mid = 16 + 32 * i
            aa = max(0, mid - self.currentAction)
            bb = min(totalNum, mid + self.currentAction)
            pred.append([aa, bb])
        return pred

    def getData(self):
        label = []
        for _ in range(self.batch):
            # # carSpeed 一维数组，长度为 10
            carSpeedMiu = 29.44444
            carSpeedSigma = 3.8
            carSpeed = carSpeedSigma * np.random.randn(10) + carSpeedMiu
            netSpeedMiu = 20
            netSpeedSigma = 1
            netSpeed = netSpeedSigma * np.random.randn(10) + netSpeedMiu
            # # rangeEnArea一维数组，长度为 10
            rangeEnArea = np.random.randint(500, 700, 10)
            chunkSize = 10  # # 块大小
            totalSize = 3200  # # 一个缓存的大小
            download = []
            ss = 0
            for j in range(len(carSpeed)):
                # # 字节数不变，content 为块数，传输的块
                content = 0
                if (random.random() > 0.2):
                    content = (
                                      ((rangeEnArea[j] / carSpeed[j]) * netSpeed[j]) / chunkSize) + random.random()
                t = min(content, totalSize / chunkSize)
                areall = int(ss + 0.5)
                areall = np.clip(areall, 0, totalNum)
                arearr = int(ss + t + 0.5)
                arearr = np.clip(arearr, 0, totalNum)
                download.append([areall, arearr])
                ss = ss + t
            label.append(download)
        return label

    def getEvaluateResult(self, dataTrue, dataPred):
        # if (len(dataTrue) != len(dataPred)) or (len(dataTrue[0]) != len(dataPred[0])):
        #     raise Exception("lengthes not matched ")

        sumNeed = 0.0  # # true 下载的块大小
        sumDown = 0.0  # # pred 下载的块大小
        sumAimed = 0.0  # # 重合的块大小
        for i in range(0, len(dataTrue)):
            for j in range(0, len(dataTrue[0])):
                sumNeed += dataTrue[i][j][1] - dataTrue[i][j][0]
                sumDown += dataPred[j][1] - dataPred[j][0]

                areall = max(dataTrue[i][j][0], dataPred[j][0])
                arearr = min(dataTrue[i][j][1], dataPred[j][1])
                # print('-----------', areall, arearr)
                if (areall <= arearr):
                    sumAimed += arearr - areall

        rateNeed = sumAimed / sumNeed
        rateDown = sumAimed / (sumDown + 0.00001)

        # formatStr = "rateNeed:%.4f, rateDownload:%.4f,sumAimed:%.4f,sumDownload:%.4f,sumNeed:%.4f" % (
        #     rateNeed, rateDownload, sumAimed, sumDownload, sumNeed)
        # self.step += 1
        # self.recordDownload.append(rateDownload)
        # self.recordNeed.append(rateNeed)
        # # # run 每 run 100 次，计算一次
        # if self.step % 100 == 0:
        #     self.averDownload.append(sum(self.recordDownload)/100)
        #     self.averNeed.append(sum(self.recordNeed)/100)
        #     self.recordDownload = []
        #     self.recordNeed = []

        self.acc_li.append(rateDown)
        if rateNeed >= 0.98:
            return False, self.acc_li[-1]
        else:
            return True, self.acc_li[-1]

    # # 程序运行
    def run(self):
        # # 动作选择，选择当前的 action，并由 x 预测 pred(1000*10*(a,b,c))
        self.makeAct()
        pred = self.predict()
        label = self.getData()
        # # 命中率大于 0.98 bool 为 False，否则为 True，acc 为下载块的命中率
        boolDown, acc = self.getEvaluateResult(label, pred)
        # 由当前命中率与 boolDown 计算当前状态
        next_state = self.updateState(acc)
        # # 由 boolDown 计算奖励值，并更新 RewardTable
        self.caculateReward(next_state, boolDown)
        self.currentState = next_state
        # print("-----   Current_State   -----   ", self.currentState)
        # return boolDown, self.currentState


if __name__ == '__main__':
    q = newqLearning()
    # e = Environment(1000)
    boolDown = False
    for i in range(1500):
        q.run()
    print("-----   final   -----")
    data = pd.DataFrame(q.rewardTable)
    data.to_csv('./q_reward.csv')
    print(q.rewardTable)
    print(q.acc_li)
