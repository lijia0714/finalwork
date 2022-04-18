#-*-coding=utf-8-*-
import numpy as np
import resNetAssign
import newRes
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import explained_variance_score,r2_score

totalNum = 320
batch = 1000
totalRewardTable = []
# class qLearning:
#
#     levelAccuracy = [0.99,0.95,0.90,0.85,0.75,0.55,0.30,0.10,0.0]
#     boolDownloadFromBackhaul = [True,False]
#     state = {
#              (levelAccuracy[0], boolDownloadFromBackhaul[0]): 0,
#              (levelAccuracy[0], boolDownloadFromBackhaul[1]): 1,
#              (levelAccuracy[1], boolDownloadFromBackhaul[0]): 2,
#              (levelAccuracy[1], boolDownloadFromBackhaul[1]): 3,
#              (levelAccuracy[2], boolDownloadFromBackhaul[0]): 4,
#              (levelAccuracy[2], boolDownloadFromBackhaul[1]): 5,
#              (levelAccuracy[3], boolDownloadFromBackhaul[0]): 6,
#              (levelAccuracy[3], boolDownloadFromBackhaul[1]): 7,
#              (levelAccuracy[4], boolDownloadFromBackhaul[0]): 8,
#              (levelAccuracy[4], boolDownloadFromBackhaul[1]): 9,
#              (levelAccuracy[5], boolDownloadFromBackhaul[0]): 10,
#              (levelAccuracy[5], boolDownloadFromBackhaul[1]): 11,
#              (levelAccuracy[6], boolDownloadFromBackhaul[0]): 12,
#              (levelAccuracy[6], boolDownloadFromBackhaul[1]): 13,
#              (levelAccuracy[7], boolDownloadFromBackhaul[0]): 14,
#              (levelAccuracy[7], boolDownloadFromBackhaul[1]): 15,
#              (levelAccuracy[8], boolDownloadFromBackhaul[0]): 16,
#              (levelAccuracy[8], boolDownloadFromBackhaul[1]): 17}
#
#     edgeCompensate = []
#     for i in range(0,int(totalNum/2)):
#         edgeCompensate.append(i)
#     rewardTable = [[0 for i in range(len(edgeCompensate))] for j in range(len(state))]
#     rateExplore = None
#     currentState = None
#     currentReward = 0
#     currentAction = None
#     lastAction = None
#     fileCheckPoint = 'checkPoint/resNetAssign'
#     fileLog = 'logs/'
#     learnRate = 0.002
#     resModel = None
#     storeX = []
#     storeY = []
#     storeNum = 0
#     def __init__(self):
#         self.rateExplore = 0.2
#         self.currentState = self.state.get((self.levelAccuracy[8],self.boolDownloadFromBackhaul[0]))
#         self.currentReward = 0
#         self.currentAction = self.edgeCompensate[159]
#         self.lastAction = self.currentAction
#         self.resModel = resNetAssign.resNetModel(self.fileCheckPoint, self.fileLog, learnRate=self.learnRate)
#
#     def updateState(self,acc,boolDownload):
#         assert acc<=1,'Error:acc value error!'
#         if acc>=0.98:
#             acc = self.levelAccuracy[0]
#         elif acc>=0.95:
#             acc = self.levelAccuracy[1]
#         elif acc >= 0.90:
#             acc = self.levelAccuracy[2]
#         elif acc >= 0.85:
#             acc = self.levelAccuracy[3]
#         elif acc >= 0.75:
#             acc = self.levelAccuracy[4]
#         elif acc >= 0.55:
#             acc = self.levelAccuracy[5]
#         elif acc >= 0.30:
#             acc = self.levelAccuracy[6]
#         elif acc >= 0.10:
#             acc = self.levelAccuracy[7]
#         else:
#             acc = self.levelAccuracy[8]
#
#         self.currentState = self.state[(acc,boolDownload)]
#
#     def caculateReward(self,boolDownload):
#         if not boolDownload:
#             reward = 63 * (totalNum+1 - self.lastAction)
#         else:
#             reward = (-64) * (self.lastAction+1)
#             print ('reward',reward)
#         print ('state:',self.currentState,'action',self.lastAction)
#         print ('orireward',self.rewardTable[self.currentState][self.lastAction])
#         self.rewardTable[self.currentState][self.lastAction] = (self.rewardTable[self.currentState][self.lastAction]+reward)*0.5
#         print ('curreward', self.rewardTable[self.currentState][self.lastAction])
#
#     def makeAct(self,x):
#         n = random.randint(0,9)
#         if n < (10*self.rateExplore):
#             print ('random')
#             self.currentAction = random.choice(self.edgeCompensate)
#         else:
#             print ('select')
#             self.currentAction = self.edgeCompensate.index(self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
#         print ("##########action%2d###########"%(self.currentAction))
#         print (self.currentState)
#         print (self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
#         print (max(self.rewardTable[self.currentState]))
#         print (self.edgeCompensate.index(self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState]))))
#         print self.rewardTable
#         assert len(x)%100 == 0,'value of x error!'
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                #         pred = self.resModel.predict(x)
#
#         if self.currentAction < 159:
#             predProcessed = self.edgeProcessed(pred, self.currentAction)
#         else:
#             predProcessed = self.fullProcessed(pred)
#         self.lastAction = self.currentAction
#         return predProcessed
#
#     def run(self,x,r,e,acc = 0,boolDownload = True):
#         self.caculateReward(boolDownload)
#         self.updateState(acc, boolDownload)
#         if r.isFull():
#             train_x,train_y = r.withDrawData()
#             self.resModel.fit(X_inputs=train_x, Y_targets=train_y, n_epoch=1, batch_size=100, validation_set=0.15,
#                               snapshot_epoch=True, snapshot_step=100)
#             # if e.step < 2000:
#             #     self.resModel.fit(X_inputs=train_x,Y_targets=train_y,n_epoch=1,batch_size=100,validation_set=0.15,snapshot_epoch=True,snapshot_step=100)
#             #     print ('step:',e.step)
#         pred = self.makeAct(x)
#
#         return pred
#
#     def edgeProcessed(self,data,addN):  # csv file only for 2 dimensons
#         data = np.array(data, dtype=float)
#         data = data + 0.5
#         data = np.array(data, dtype=int)
#         sizeData = data.shape[0]
#         sizeEn = int(data.shape[1] / 2)
#         dataList = []
#         for i in range(0, sizeData):
#             enList = []
#             for j in range(0, sizeEn):
#                 data[i, j * 2 + 1] = data[i, j * 2 + 1] + data[i, j * 2]
#                 start = data[i, j * 2] - addN
#                 end = data[i, j * 2 + 1] + addN
#                 if start < 0:
#                     start = 0
#                 if start > totalNum:
#                     start = totalNum
#                 if end > totalNum:
#                     end = totalNum
#                 if end < 0:
#                     end = 0
#                 n = end - start
#                 enList.append((start, end, n))
#             dataList.append(enList)
#         return dataList
#
#
#     def fullProcessed(self,data):  # csv file only for 2 dimensons
#         data = np.array(data, dtype=int)
#         sizeData = data.shape[0]
#         sizeEn = data.shape[1] / 2
#         dataList = []
#         for i in range(0, sizeData):
#             enList = []
#             for j in range(0, sizeEn):
#                 start = 0
#                 end = totalNum
#                 n = end - start
#                 enList.append((start, end, n))
#             dataList.append(enList)
#         # turn data into lists of turples:(start,end)
#         return dataList

class newqLearning:

    levelAccuracy = [0.99,0.95,0.90,0.85,0.75,0.55,0.30,0.10,0.0]
    boolDownloadFromBackhaul = [True,False]
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

    edgeCompensate = []
    for i in range(0,int(totalNum/2)):
        edgeCompensate.append(i)
    rewardTable = [[0 for i in range(len(edgeCompensate))] for j in range(len(state))]
    rateExplore = None
    currentState = None
    currentReward = 0
    currentAction = None
    lastAction = None
    fileCheckPoint = 'checkPoints/resNetAssign'
    fileLog = 'logs/'
    learnRate = 0.01
    resModel = None
    storeX = []
    storeY = []
    storeNum = 0
    acc = 0
    trainFlag = True
    bingoTimes = 0
    def __init__(self):
        self.rateExplore = 0.2
        self.currentState = self.state.get((self.levelAccuracy[8],self.boolDownloadFromBackhaul[0]))
        self.currentReward = 0
        self.currentAction = self.edgeCompensate[159]
        #self.resModel = newRes.resNetModel(self.fileCheckPoint, self.fileLog, learnRate=self.learnRate)
        # self.resModel = newRes.oldResNetModel(self.fileCheckPoint, self.fileLog, learnRate=self.learnRate)
        self.resModel = newRes.myResNetModel(self.fileCheckPoint, self.fileLog, learnRate=self.learnRate)
    def updateState(self,acc,boolDownload):
        assert acc<=1,'Error:acc value error!'
        if acc>=0.98:
            acc = self.levelAccuracy[0]
        elif acc>=0.95:
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

        self.currentState = self.state[(acc,boolDownload)]

    def caculateReward(self,boolDownload):
        if not boolDownload:
            reward = 63 * (totalNum+1 - self.currentAction)
        else:
            reward = (-64) * (self.currentAction+1)
            #print ('reward',reward)
        #print ('state:',self.currentState,'action',self.currentAction)
        #print ('orireward',self.rewardTable[self.currentState][self.currentAction])
        self.rewardTable[self.currentState][self.currentAction] = (self.rewardTable[self.currentState][self.currentAction]+reward)*0.5
        #print ('curreward', self.rewardTable[self.currentState][self.currentAction])

    def makeAct(self,x):
        n = random.randint(0,9)
        if n < (10*self.rateExplore):
            self.currentAction = random.choice(self.edgeCompensate)
            print ("random: ##########action%2d###########" % (self.currentAction))
        else:
            self.currentAction = self.edgeCompensate.index(self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
            print ("select: ##########action%2d###########" % (self.currentAction))
        # print (self.currentState)
        # print (self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState])))
        # print (max(self.rewardTable[self.currentState]))
        # print (self.edgeCompensate.index(self.rewardTable[self.currentState].index(max(self.rewardTable[self.currentState]))))
        # print self.rewardTable
        assert len(x)%100 == 0,'value of x error!'


        pred = self.resModel.predict(x)

        if self.currentAction < 159:
            predProcessed = self.edgeProcessed(pred, self.currentAction)
        else:
            predProcessed = self.fullProcessed(pred)
        return predProcessed

    def run(self,x,r,boolDownload = True):
        self.caculateReward(boolDownload)
        if r.isFull():
            train_x,train_y = r.withDrawData()
            if self.acc>0.99:
            #     self.bingoTimes+=1
            # if self.bingoTimes >= 3:
                self.trainFlag = False
            if self.trainFlag:
                self.resModel.summary()
                self.resModel.fit(X_inputs=train_x, Y_targets=train_y, n_epoch=1, shuffle=True,batch_size=256, validation_set=0.1,
                                  snapshot_epoch=True, snapshot_step=10)
            pred = self.resModel.predict(train_x[0:1024])
            dataTrue = self.getDataProcessed(train_y[0:1024])
            dataPred = self.getDataProcessed(pred[0:1024])
            print dataTrue
            print dataPred
            self.acc = self.eva(dataTrue,dataPred)
            print "\n acc:",self.acc,"\n"

        self.updateState(self.acc, boolDownload)
        pred = self.makeAct(x)

        return pred,self.acc

    def edgeProcessed(self,data,addN):  # csv file only for 2 dimensons
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


    def fullProcessed(self,data):  # csv file only for 2 dimensons
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = data.shape[1] / 2
        dataList = []
        for i in range(0, sizeData):
            enList = []
            for j in range(0, sizeEn):
                start = 0
                end = totalNum
                n = end - start
                enList.append((start, end, n))
            dataList.append(enList)
        return dataList

    def eva(self,dataTrue,dataPred):
        sumNeed = 0.0
        sumAimed = 0.0
        for i in range(0, len(dataTrue)):
            for j in range(0, len(dataTrue[0])):
                sumNeed += dataTrue[i][j][2]

                if (dataTrue[i][j][0] >= dataPred[i][j][0]) and (dataTrue[i][j][1] <= dataPred[i][j][1]):
                    sumAimed += dataTrue[i][j][2]

                elif (dataTrue[i][j][0] <= dataPred[i][j][0]) and (dataTrue[i][j][1] >= dataPred[i][j][1]):
                    sumAimed += dataPred[i][j][2]

                elif (dataTrue[i][j][0] < dataPred[i][j][0]) and (dataTrue[i][j][1] < dataPred[i][j][1]):
                    if ((dataTrue[i][j][1] - dataPred[i][j][0]) >= 0):
                        sumAimed += (dataTrue[i][j][1] - dataPred[i][j][0])

                elif (dataTrue[i][j][0] > dataPred[i][j][0]) and (dataTrue[i][j][1] > dataPred[i][j][1]):
                    if ((dataPred[i][j][1] - dataTrue[i][j][0]) >= 0):
                        sumAimed += (dataPred[i][j][1] - dataTrue[i][j][0])

        rateNeed = sumAimed / sumNeed
        return rateNeed

    def getDataProcessed(self,data):  # csv file only for 2 dimensons
        data = np.array(data, dtype=float)
        data += 0.5
        data = np.array(data, dtype=int)
        sizeData = data.shape[0]
        sizeEn = int(data.shape[1] / 2)
        dataList = []
        #####
        data_number=[]
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
class Recorder2:

    num = 0
    dataSetA = []
    dataSetB = []
    fullNum = 20
    rate = 0.2

    def append(self,x,y):
        if self.num < int(self.fullNum * self.rate):
            a = []
            a.append(x)
            a.append(y)
            self.dataSetB.append(a)
        else:
            a = []
            a.append(x)
            a.append(y)
            self.dataSetA.append(a)

        self.num += 1

    def withDrawData(self):

        if self.num >= self.fullNum:
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
            return x,y
        else:
            return [],[]


    def isFull(self):
        if self.num >= self.fullNum:
            return True
        else:
            return False

class Environment:
    xTotal = None
    yTotal = None
    dataSize = None
    position = 0
    recordDownload = []
    recordNeed = []
    averDownload = []
    averNeed = []
    averStep = []
    acc = []
    step = 0
    batch = 0

    def __init__(self,batch = 1000):
        self.batch = batch
    #这个函数获取的就是获取车辆信息和下载内容的
    #输入是：x[车辆速度参数和网络参数]
    #输出是：y[时间，下载的内容]
    def getCurrentData(self): #获取批量车辆的数据信息
        carSpeedMiu = 29.44444
        carSpeedSigma = 3.8
        carSpeed = carSpeedSigma * np.random.randn(10) + carSpeedMiu
        netSpeedMiu = 20
        netSpeedSigma = 1
        netSpeed = netSpeedSigma * np.random.randn(10) + netSpeedMiu
        rangeEnArea = np.random.randint(900,1100,10)
        chunkSize = 10
        totalSize = 3200
        x = []
        y = []
        newX = []
        for i in range(0,self.batch):
            enDetails = []
            download = []
            t = 0
            for j in range(0,len(carSpeed)):
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
                if j == 0:
                    download.append(0)
                else:
                    download.append(t)

                #计算每一辆车在缓存节点下的下载数量
                content = (((rangeEnArea[j] / carSpeed[j]) * netSpeed[j]) / chunkSize) + random.random()

                if ( (download[0] + content) > (totalSize / chunkSize)):
                    download.append(int(totalSize / chunkSize))
                    t = int(totalSize / chunkSize)
                else:
                    download.append(download[0] + content)
                    t = download[0] + content

                enDetails.append(enDetail)

            x.append(enDetails)
            y.append(download)
        x = np.asarray(x)
        print(x.shape)
        # x = x.reshape((self.batch,10,10,1))
        return x,y

    def getDataProcessed(self,data):  # csv file only for 2 dimensons
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

    def getEvaluteResult(self,dataTrue, dataPred,acc = 0):
        dataTrue = self.getDataProcessed(dataTrue)
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

                if (dataTrue[i][j][0] >= dataPred[i][j][0]) and (dataTrue[i][j][1] <= dataPred[i][j][1]):
                    sumAimed += dataTrue[i][j][2]

                elif (dataTrue[i][j][0] <= dataPred[i][j][0]) and (dataTrue[i][j][1] >= dataPred[i][j][1]):
                    sumAimed += dataPred[i][j][2]

                elif (dataTrue[i][j][0] < dataPred[i][j][0]) and (dataTrue[i][j][1] < dataPred[i][j][1]):
                    if ((dataTrue[i][j][1] - dataPred[i][j][0]) >= 0):
                        sumAimed += (dataTrue[i][j][1] - dataPred[i][j][0])

                elif (dataTrue[i][j][0] > dataPred[i][j][0]) and (dataTrue[i][j][1] > dataPred[i][j][1]):
                    if ((dataPred[i][j][1] - dataTrue[i][j][0]) >= 0):
                        sumAimed += (dataPred[i][j][1] - dataTrue[i][j][0])

        rateNeed = sumAimed / sumNeed
        rateDownload = sumAimed / (sumDownload+0.0001)

        formatStr = "rateNeed:%.4f, rateDownload:%.4f,sumAimed:%.4f,sumDownload:%.4f,sumNeed:%.4f"% (rateNeed, rateDownload,sumAimed,sumDownload,sumNeed)
        self.step+=1
        self.recordDownload.append(rateDownload)
        self.recordNeed.append(rateNeed)
        if self.step%100 == 0:
            self.averDownload.append(sum(self.recordDownload)/100)
            self.averNeed.append(sum(self.recordNeed)/100)
            self.averStep.append(self.step)
            self.recordDownload = []
            self.recordNeed = []
            self.acc.append(acc)

        if rateNeed >= 0.98:
            # print 'pred',dataPred
            # print 'true',dataTrue
            print formatStr
            #return formatStr,rateNeed,False
            return formatStr, False
        else:
            #return formatStr,rateNeed,True
            return formatStr, True

if __name__ == '__main__':
    q = newqLearning()#产生一个强化学习的对象
    e = Environment(10)#产生一个环境对象
    r = Recorder2()#产生一个计数器对象
    boolDown = False#默认从缓存节点下载
    for i in range(50):
        x, label = e.getCurrentData()
        print(x.shape)
        pred,acc= q.run(x=x,r=r,boolDownload=boolDown)
        str,boolDown = e.getEvaluteResult(label, pred,acc)
        if not boolDown:
            r.append(x=x,y=label)
        if i % 1000 == 0:
            totalRewardTable.extend(q.rewardTable)
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.plot(e.averStep, e.averNeed)
    # plt.xlabel('step')
    # plt.ylabel('cash hit ratio')
    #
    # plt.subplot(222)
    # plt.plot(e.averStep, e.averDownload)
    # plt.xlabel('step')
    # plt.ylabel('effective cashing ratio')
    #
    # plt.subplot(223)
    # plt.plot(e.averStep, e.acc)
    # plt.xlabel('step')
    # plt.ylabel('explained_variance_score')
    #
    # plt.show()
    #
    # a = np.array(totalRewardTable)
    # print(a)
    # np.savetxt('/home/frank/work/finalResMapAssign/pic/rewardTable.txt',a)
    # b = np.array(e.averStep)
    # np.savetxt('/home/frank/work/finalResMapAssign/pic/step.txt', b)
    # c = np.array(e.averNeed)
    # np.savetxt('/home/frank/work/finalResMapAssign/pic/need.txt', c)
    # d = np.array(e.averDownload)
    # np.savetxt('/home/frank/work/finalResMapAssign/pic/download.txt', d)
