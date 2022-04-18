# coding:utf-8
import numpy as np
import pandas as pd
import resNetAssign
import newRes
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import explained_variance_score, r2_score
import csv

totalNum = 320
# batch = 1000
# totalRewardTable = []


class newqLearning:
    levelAccuracy = [0.99, 0.95, 0.90, 0.85, 0.75, 0.55, 0.30, 0.10, 0.0]
    boolDownloadFromBackhaul = [True, False]
    # 状态定义
    state = {
        (levelAccuracy[0], boolDownloadFromBackhaul[0]): 0,
        (levelAccuracy[0], boolDownloadFromBackhaul[1]): 1,
        (levelAccuracy[1], boolDownloadFromBackhaul[0]): 2,
        (levelAccuracy[1], boolDownloadFromBackhaul[1]): 3,
        (levelAccuracy[2], boolDownloadFromBackhaul[0]): 4,
        (levelAccuracy[2], boolDownloadFromBackhaul[1]): 5,
        (levelAccuracy[3], boolDownloadFromBackhaul[0]): 6,
        (levelAccuracy[3], boolDownloadFromBackhaul[1]): 7,
        (levelAccuracy[4], boolDownloadFromBackhaul[0]): 8,
        (levelAccuracy[4], boolDownloadFromBackhaul[1]): 9,
        (levelAccuracy[5], boolDownloadFromBackhaul[0]): 10,
        (levelAccuracy[5], boolDownloadFromBackhaul[1]): 11,
        (levelAccuracy[6], boolDownloadFromBackhaul[0]): 12,
        (levelAccuracy[6], boolDownloadFromBackhaul[1]): 13,
        (levelAccuracy[7], boolDownloadFromBackhaul[0]): 14,
        (levelAccuracy[7], boolDownloadFromBackhaul[1]): 15,
        (levelAccuracy[8], boolDownloadFromBackhaul[0]): 16,
        (levelAccuracy[8], boolDownloadFromBackhaul[1]): 17}

    edgeCompensate = [i for i in range(int(totalNum / 2))]
    # for i in range(0, int(totalNum/2)):
    #     edgeCompensate.append(i)
    rewardTable = [[0 for i in range(len(edgeCompensate))]
                   for j in range(len(state))]
    rateExplore = None
    currentState = None
    currentReward = 0
    currentAction = None
    # 预测模型文件
    resModel = None
    acc = 0
    trainFlag = True
    # 初始化
    def __init__(self):
        self.fileCheckPoint = './checkPoints/resNetAssign'
        self.fileLog = './logs/'
        self.learnRate = 0.01
        self.rateExplore = 0.2
        self.currentState = self.state.get(
            (self.levelAccuracy[8], self.boolDownloadFromBackhaul[0]))
        self.currentReward = 0
        self.currentAction = self.edgeCompensate[159]
        self.resModel = newRes.resNetModel(
            self.fileCheckPoint, self.fileLog, learnRate=self.learnRate)
    # 状态更新
    def updateState(self, acc, boolDownload):
        assert acc <= 1, 'Error:acc value error!'
        if acc >= 0.98:
            acc = self.levelAccuracy[0]
        elif acc >= 0.95:
            acc = self.levelAccuracy[1]
        elif acc >= 0.90:
            acc = self.levelAccuracy[2]
        elif acc >= 0.85:
            acc = self.levelAccuracy[3]
        elif acc >= 0.75:
            acc = self.levelAccuracy[4]
        elif acc >= 0.55:
            acc = self.levelAccuracy[5]
        elif acc >= 0.30:
            acc = self.levelAccuracy[6]
        elif acc >= 0.10:
            acc = self.levelAccuracy[7]
        else:
            acc = self.levelAccuracy[8]
        self.currentState = self.state[(acc, boolDownload)]
        return self.currentState
    # 计算奖励值
    def caculateReward(self, boolDownload):
        # # 如果没有从后台下载
        if not boolDownload:
            reward = 63 * (totalNum + 1 - self.currentAction)
        else:
            reward = (-64) * (self.currentAction + 1)
        # print('state:', self.currentState, 'action', self.currentAction)
        # print('orireward',
        #       self.rewardTable[self.currentState][self.currentAction])
        self.rewardTable[self.currentState][self.currentAction] = (
            self.rewardTable[self.currentState][self.currentAction] + reward) * 0.5
    # 动作选择
    def makeAct(self, x):
        n = random.randint(0, 9)
        if n < (10*self.rateExplore):
            self.currentAction = random.choice(self.edgeCompensate)
            # print ("random: ##########action%2d###########" %
            #        (self.currentAction))
        else:
            self.currentAction = self.edgeCompensate.index(
                self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
            # print ("select: ##########action%2d###########" %
            #        (self.currentAction))
        # print (self.currentState)
        # print (self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
        # print (max(self.rewardTable[self.currentState]))
        # print (self.edgeCompensate.index(self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState]))))
        # print self.rewardTable
        # assert len(x)%100 == 0,'value of x error!'
        # print("-----1---State_predict--------")
        #print 'x.shape is'+str(x.shape)
        pred = self.resModel.predict(x)
        # pred = self.resModel.predict(x[-100])
        # print("---------resmodel predict target-----------")
        # print(pred)

        if self.currentAction < 159:
            predProcessed = self.edgeProcessed(pred, self.currentAction)
        else:
            predProcessed = self.fullProcessed(pred)
        return predProcessed
    # 通过预测的块数和真实的块数得到真实命中率

    def run(self, x, r, label, boolDownload=True):
        # # 由 boolDown 计算奖励值，并更新 RewardTable
        self.caculateReward(boolDownload)
        if r.isFull():
            train_x, train_y = r.withDrawData()
            if self.acc > 0.99:
                self.trainFlag = False
            if self.trainFlag:
                self.resModel.fit(X_inputs=train_x, Y_targets=train_y, n_epoch=1, shuffle=True, batch_size=50, validation_set=0.1,
                                  snapshot_epoch=True, snapshot_step=10)
            pred = self.resModel.predict(train_x[0:1024])
            dataTrue = self.getDataProcessed(train_y[0:1024])
            dataPred = self.getDataProcessed(pred[0:1024])
            self.acc = self.eva(dataTrue, dataPred)
            print("acc:  ", self.acc, "\n")
        # # 动作选择，选择当前的 action，并由 x 预测 pred(1000*10*(a,b,c))
        pred = self.makeAct(x)
        # # 对 label 进行处理
        label = self.getDataProcessed(label)
        # # 命中率大于 0.98 bool 为 False，否则为 True，acc 为下载块的命中率
        boolDown, acc = e.getEvaluateResult(label, pred)
        # # 由当前命中率与 boolDown 计算当前状态
        current_state = self.updateState(acc, boolDownload)
        print("-----   Current_State   -----   ", current_state)
        return boolDown, current_state

    # 对数据进行处理
    def edgeProcessed(self, data, addN):  # csv file only for 2 dimensons
        data = np.array(data, dtype=float)
        data = data + 0.5
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = int(data.shape[1] / 2)
        dataList = []
        for i in range(0, sizeData):
            enList = []
            for j in range(0, sizeEn):
                data[i, j * 2 + 1] = data[i, j * 2 + 1] + data[i, j * 2]
                start = data[i, j * 2] - addN
                end = data[i, j * 2 + 1] + addN
                if start < 0:
                    start = 0
                if start > totalNum:
                    start = totalNum
                if end > totalNum:
                    end = totalNum
                if end < 0:
                    end = 0
                n = end - start
                enList.append((start, end, n))
            dataList.append(enList)
        return dataList

    def fullProcessed(self, data):  # csv file only for 2 dimensons
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = data.shape[1] / 2
        dataList = []
        for _ in range(0, sizeData):
            enList = []
            for _ in range(0, sizeEn):
                start = 0
                end = totalNum
                n = end - start
                enList.append((start, end, n))
            dataList.append(enList)
        return dataList

    def eva(self, dataTrue, dataPred):
        sumNeed = 0.0
        sumAimed = 0.0
        for i in range(0, len(dataTrue)):
            for j in range(0, len(dataTrue[0])):
                sumNeed += dataTrue[i][j][2]

                areall = max(dataTrue[i][j][0], dataPred[i][j][0])
                arearr = min(dataTrue[i][j][1], dataPred[i][j][1])
                if(areall <= arearr):
                    sumAimed += arearr-areall

        rateNeed = sumAimed / sumNeed
        return rateNeed

    def getDataProcessed(self, data):  # csv file only for 2 dimensons
        data = np.array(data, dtype=float)
        data += 0.5
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = int(data.shape[1] / 2)
        dataList = []
        #####
        data_number = []
        #####
        for i in range(0, sizeData):
            enList = []
            for j in range(0, sizeEn):
                n = data[i, j * 2 + 1]
                data[i, j * 2 + 1] = data[i, j * 2 + 1] + data[i, j * 2]
                enList.append((data[i, j * 2], data[i, j * 2 + 1], n))
                ####
                data_number.append(n)
            dataList.append(enList)
        # turn data into lists of turples:(start,end)
        return dataList


class Recorder:
    num = 0
    dataSetA = []
    dataSetB = []
    fullNum = 20
    rate = 0.2
    # 添加数据
    def append(self, x, y):
        if self.num < int(self.fullNum * self.rate):
            self.dataSetB.append([x, y])
        else:
            self.dataSetA.append([x, y])

        self.num += 1
    # 记录数据

    def withDrawData(self):
        if self.num < self.fullNum:
            return [], []
        t = self.dataSetA + self.dataSetB
        random.shuffle(t)
        self.num = int(self.fullNum * self.rate)
        x = []
        y = []
        for i in t:
            x.extend(i[0])
            y.extend(i[1])

        self.dataSetB = t[0:self.num]
        self.dataSetA = []
        return x, y


    def isFull(self):
        if self.num >= self.fullNum:
            return True
        else:
            return False


class Environment:
    recordDownload = []
    recordNeed = []
    averDownload = []
    averNeed = []
    averStep = []
    acc = []
    step = 0
    batch = 0

    def __init__(self, batch=1000):
        self.batch = batch
    # 这个函数获取的就是获取车辆信息和下载内容的
    # 输入是：x[车辆参数，网络参数]
    # 输出是：y[时间，下载的内容]
    def getCurrentData(self):
        x = []
        y = []
        for _ in range(self.batch):
            carSpeedMiu = 29.44444
            carSpeedSigma = 3.8     
            # # carSpeed 一维数组，长度为 10
            carSpeed = carSpeedSigma * np.random.randn(10) + carSpeedMiu
            netSpeedMiu = 20
            netSpeedSigma = 1
            netSpeed = netSpeedSigma * np.random.randn(10) + netSpeedMiu
            # # rangeEnArea一维数组，长度为 10
            rangeEnArea = np.random.randint(900, 1100, 10)
            chunkSize = 10  # # 块大小
            totalSize = 3200    # # 一个缓存的大小
            enDetails = []
            download = []
            t = 0
            for j in range(0, len(carSpeed)):
                enDetail = []
                enDetail.append(carSpeed[j])
                enDetail.append(carSpeedMiu)
                enDetail.append(carSpeedSigma)
                enDetail.append(netSpeed[j])
                enDetail.append(netSpeedMiu)
                enDetail.append(netSpeedSigma)
                enDetail.append(rangeEnArea[j])
                enDetail.append(chunkSize)
                enDetail.append(totalSize)
                enDetail.append(j)

                download.append(t)
                # # 字节数不变，content 为块数，传输的块？？
                content = (
                    ((rangeEnArea[j] / carSpeed[j]) * netSpeed[j]) / chunkSize) + random.random()

                # # download[0] == 0
                t = min(download[0] + content, totalSize / chunkSize)
                download.append(t)
                enDetails.append(enDetail)
            # # enDetails 10*10, download 10
            x.append(enDetails)
            y.append(download)
        x = np.asarray(x)
        # # x 100*10*10*1
        x = x.reshape((self.batch, 10, 10, 1))
        return x, y

    def getDataProcessed(self, data):  # csv file only for 2 dimensons
        data = np.array(data, dtype=float)
        data += 0.5
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = int(data.shape[1] / 2)
        dataList = []
        for i in range(0, sizeData):
            enList = []
            for j in range(0, sizeEn):
                n = data[i, j * 2 + 1]
                data[i, j * 2 + 1] = data[i, j * 2 + 1] + data[i, j * 2]
                enList.append((data[i, j * 2], data[i, j * 2 + 1], n))
            dataList.append(enList)
        # turn data into lists of turples:(start,end)
        return dataList
    # 获得比较的结果
    def getEvaluateResult(self, dataTrue, dataPred):
        lenTrue = len(dataTrue)
        lenTrueEn = len(dataTrue[0])
        lenPred = len(dataPred)
        lenPredEn = len(dataPred[0])
        if (lenTrue != lenPred) or (lenPredEn != lenTrueEn):
            raise Exception("lengthes not matched ")

        sumNeed = 0.0
        sumDownload = 0.0
        sumAimed = 0.0

        for i in range(0, lenTrue):
            for j in range(0, lenTrueEn):
                sumNeed += dataTrue[i][j][2]
                sumDownload += dataPred[i][j][2]

                areall = max(dataTrue[i][j][0], dataPred[i][j][0])
                arearr = min(dataTrue[i][j][1], dataPred[i][j][1])
                if(areall <= arearr):
                    sumAimed += arearr - areall
        rateNeed = sumAimed / sumNeed
        rateDownload = sumAimed / (sumDownload+0.0001)

        formatStr = "rateNeed:%.4f, rateDownload:%.4f,sumAimed:%.4f,sumDownload:%.4f,sumNeed:%.4f" % (
            rateNeed, rateDownload, sumAimed, sumDownload, sumNeed)
        self.step += 1
        self.recordDownload.append(rateDownload)
        self.recordNeed.append(rateNeed)
        # # run 每 run 100 次，计算一次
        if self.step % 100 == 0:
            self.averDownload.append(sum(self.recordDownload)/100)
            self.averNeed.append(sum(self.recordNeed)/100)
            self.averStep.append(self.step)
            self.recordDownload = []
            self.recordNeed = []

        self.acc = rateDownload
        if rateNeed >= 0.98:
            return False, self.acc
        else:
            return True, self.acc


if __name__ == '__main__':
    q = newqLearning()
    e = Environment(1000)
    r = Recorder()
    boolDown = False
    for i in range(1000):
        # # x(1000*10*10*1) 车辆状态，label(1000*20) 下载内容块的大小
        x, label = e.getCurrentData()
        # # 由 x, label 传入进行训练
        boolDown, acc = q.run(x=x, r=r, label=label, boolDownload=boolDown)
        # 如果不从backhalu中下载的话
        if not boolDown:
            r.append(x=x, y=label)
    print("-----   final   -----")
    data = pd.DataFrame(q.rewardTable)
    data.to_csv('./q_reward.csv')
    print(q.rewardTable)
