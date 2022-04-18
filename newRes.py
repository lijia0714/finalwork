# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import csv
import tflearn
from tflearn.layers.core import input_data, fully_connected, flatten, activation, dropout
from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
# from tflearn.optimizers import Momentumz
from tflearn.optimizers import Momentum
from tflearn.metrics import R2
import tensorflow as tf
import os
import numpy as np
import pandas
from sklearn import preprocessing as pr


fileLog = 'logs/'
fileData = 'data/dataTrain.csv'
nb_epoch = 50
batch_size = 64
numOfEn = 10
numOfAtt = 12
sizeConv = 1
nb_filters = 32
sizePool = 3
stridesPool = 1


def getTrainData(fileDataTrain=fileData):
    dataframe = pandas.read_csv(fileDataTrain, delimiter=',', header=None)
    datasets = dataframe.values
    sizeData = len(datasets)
    scaler = pr.MinMaxScaler().fit(datasets)
    scaler.transform(datasets)
    x = np.zeros((sizeData, numOfEn, numOfAtt - 2, 1))
    y = np.zeros((sizeData, numOfEn * 2))
    xSource = datasets[:, numOfEn * 2:]
    for i in range(0, sizeData):
        testAll = sum(xSource[i, 0:numOfEn])
        if testAll == 0:
            print("error i:", i)
        for j in range(0, numOfEn):
            for k in range(0, numOfAtt - 2):
                x[i, j, k] = xSource[i, j + k * (numOfAtt - 2)]
            for l in range(0, numOfEn):
                y[i, l * 2] = datasets[i, l + numOfEn]
                y[i, l * 2 + 1] = datasets[i, l]
    x_train = x[:, :, :]
    y_train = y[:, :]
    print(y_train)
    return x_train, y_train


def resNetModel(fileCheckPoint, fileLogs, learnRate=0.001):
    # with tf.name_scope('Inputs'):
    # tflearn.config.init_graph(gpu_memory_fraction=0.9)
    n = 9
    net = tflearn.input_data(shape=[None, 10, 10, 1])
    net = tflearn.conv_2d(net, 16, 3)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32)
    net = tflearn.residual_block(net, n - 1, 32)
    net = tflearn.residual_block(net, 1, 64)
    net = tflearn.residual_block(net, n - 1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    loss = fully_connected(net, numOfEn * 2, activation='relu')
    # loss = dropout(loss,0.7)
    momentum = Momentum(learning_rate=learnRate, lr_decay=0.5, decay_step=30000)
    r2 = R2()
    network = regression(loss, optimizer=momentum, learning_rate=learnRate, loss='mean_square', metric=r2)
    model = tflearn.DNN(network, checkpoint_path=fileCheckPoint, max_checkpoints=1, tensorboard_verbose=0,
                        tensorboard_dir=fileLogs)

    ckptFile = tf.train.latest_checkpoint(fileCheckPoint.split('/')[0])
    print(ckptFile)
    if ckptFile:
        print("find a model!")
        model.load(ckptFile, weights_only=False)
        print("finish model loading!")

    return model


def oldResNetModel(fileCheckPoint, fileLogs, learnRate=0.001):
    # with tf.name_scope('Inputs'):
    # tflearn.config.init_graph(gpu_memory_fraction=0.9)

    network = input_data(shape=[None, numOfEn, numOfAtt - 2, 1], name="input")
    # with tf.name_scope('conv1'):
    network = conv_2d(network, nb_filter=16, filter_size=1, strides=1, activation='relu', name='conv1')
    network = batch_normalization(network, name='BN1')
    network = activation(network, activation='relu', name='relu')
    # with tf.name_scope('maxpool'):
    network = max_pool_2d(network, kernel_size=1, strides=1, name='maxpool')
    # with tf.name_scope('convblock1'):
    network = convBlock(network, [16, 16, 64], name='convblock1')
    # with tf.name_scope('identifyBlock1'):
    network = idenBlock(network, [16, 16, 64], name='identifyBlock1')
    # with tf.name_scope('identifyBlock2'):
    network = idenBlock(network, [16, 16, 64], name='identifyBlock2')

    # with tf.name_scope('convblock2'):
    network = convBlock(network, [32, 32, 128], name='convBlock2')
    # with tf.name_scope('identifyBlock3'):
    network = idenBlock(network, [32, 32, 128], name='identifyBlock3')
    # with tf.name_scope('identifyBlock4'):
    network = idenBlock(network, [32, 32, 128], name='identifyBlock4')
    # with tf.name_scope('identifyBlock5'):
    network = idenBlock(network, [32, 32, 128], name='identifyBlock5')

    # with tf.name_scope('convblock3'):
    network = convBlock(network, [64, 64, 256], name='convBlock3')
    # with tf.name_scope('identifyBlock6'):
    network = idenBlock(network, [64, 64, 256], name='identifyBlock6')
    # with tf.name_scope('identifyBlock7'):
    network = idenBlock(network, [64, 64, 256], name='identifyBlock7')
    # with tf.name_scope('identifyBlock8'):
    network = idenBlock(network, [64, 64, 256], name='identifyBlock8')
    # with tf.name_scope('identifyBlock9'):
    network = idenBlock(network, [64, 64, 256], name='identifyBlock9')
    # with tf.name_scope('identifyBlock10'):
    network = idenBlock(network, [64, 64, 256], name='identifyBlock10')

    # with tf.name_scope('convblock4'):
    network = convBlock(network, [128, 128, 512], name='convBlock4')
    # with tf.name_scope('identifyBlock11'):
    network = idenBlock(network, [128, 128, 512], name='identifyBlock11')
    # with tf.name_scope('identifyBlock12'):
    network = idenBlock(network, [128, 128, 512], name='identifyBlock12')
    # with tf.name_scope('avepool'):

    network = avg_pool_2d(network, kernel_size=1, strides=1, name='avePool')

    # with tf.name_scope('flatten'):
    # network = flatten(network,name='flatten')
    # with tf.name_scope('output'):

    loss = fully_connected(network, numOfEn * 2, activation='relu')
    # loss = dropout(loss,0.7)
    momentum = Momentum(learning_rate=learnRate, lr_decay=0.5, decay_step=30000)
    r2 = R2()
    network = regression(loss, optimizer=momentum, learning_rate=learnRate, loss='mean_square', metric=r2)
    model = tflearn.DNN(network, checkpoint_path=fileCheckPoint, max_checkpoints=1, tensorboard_verbose=0,
                        tensorboard_dir=fileLogs)

    ckptFile = tf.train.latest_checkpoint(fileCheckPoint.split('/')[0])
    print(ckptFile)
    if ckptFile:
        print("find a model!")
        model.load(ckptFile, weights_only=False)
        print("finish model loading!")

    return model


def idenBlock(x, nbFilters, kernelSize=1, name='identifyBlock'):
    k1, k2, k3 = nbFilters
    nameCov1 = name + '_branch1_conv1'
    nameBn1 = name + '_branch1_Bn1'
    nameRelu1 = name + '_branch1_relu1'
    nameCov2 = name + '_branch1_conv2'
    nameBn2 = name + '_branch1_Bn2'
    nameRelu2 = name + '_branch1_relu1'
    nameCov3 = name + '_branch1_conv3'
    nameBn3 = name + '_branch1_Bn3'
    nameOutM = name + '_out_merge'
    nameOutA = name + '_out_relu'

    out = conv_2d(x, k1, 1, 1, name=nameCov1)
    out = batch_normalization(out, name=nameBn1)
    out = activation(out, activation='relu', name=nameRelu1)

    out = conv_2d(out, k2, 3, 1, name=nameCov2)
    out = batch_normalization(out, name=nameBn2)
    out = activation(out, 'relu', name=nameRelu2)

    out = conv_2d(out, k3, 1, 1, name=nameCov3)
    out = batch_normalization(out, name=nameBn3)

    out = merge([out, x], mode='concat', axis=0, name=nameOutM)
    out = activation(out, activation='relu', name=nameOutA)

    return out


def convBlock(x, nbFilters, kernelSize=1, name='convBlock'):
    k1, k2, k3 = nbFilters
    nameCov1 = name + '_branch1_conv1'
    nameBn1 = name + '_branch1_Bn1'
    nameRelu1 = name + '_branch1_relu1'
    nameCov2 = name + '_branch1_conv2'
    nameBn2 = name + '_branch1_Bn2'
    nameRelu2 = name + '_branch1_relu1'
    nameCov3 = name + '_branch1_conv3'
    nameBn3 = name + '_branch1_Bn3'
    nameCov4 = name + '_branch2_conv'
    nameBn4 = name + '_branch2_Bn'
    nameOutM = name + '_out_merge'
    nameOutA = name + '_out_relu'
    out = conv_2d(x, k1, filter_size=1, strides=1, name=nameCov1)
    out = batch_normalization(out, name=nameBn1)
    out = activation(out, activation='relu', name=nameRelu1)

    out = conv_2d(out, k2, 3, 1, name=nameCov2)
    out = batch_normalization(out, name=nameBn2)
    out = activation(out, 'relu', name=nameRelu2)

    out = conv_2d(out, k3, 1, 1, name=nameCov3)
    out = batch_normalization(out, name=nameBn3)

    x = conv_2d(x, k3, 1, 1, name=nameCov4)
    x = batch_normalization(x, name=nameBn4)

    out = merge([out, x], mode='concat', axis=0, name=nameOutM)
    out = activation(out, activation='relu', name=nameOutA)

    return out


def myResNetModel(fileCheckPoint, fileLogs, learnRate=0.001):
    # with tf.name_scope('Inputs'):
    # tflearn.config.init_graph(gpu_memory_fraction=0.9)

    network = input_data(shape=[None, numOfEn, numOfAtt - 2, 1], name="input")
    # with tf.name_scope('conv1'):
    network = conv_2d(network, nb_filter=16, filter_size=1, strides=1, activation='relu', name='conv1')
    network = batch_normalization(network, name='BN1')
    network = activation(network, activation='relu', name='relu')
    # with tf.name_scope('maxpool'):
    network = max_pool_2d(network, kernel_size=1, strides=1, name='maxpool')
    # with tf.name_scope('convblock1'):
    network = myconvBlock(network, [16, 16, 64], name='convblock1')
    # with tf.name_scope('identifyBlock1'):
    network = myidenBlock(network, [16, 16, 64], name='identifyBlock1')
    # with tf.name_scope('identifyBlock2'):
    network = myidenBlock(network, [16, 16, 64], name='identifyBlock2')

    # with tf.name_scope('convblock2'):
    network = myconvBlock(network, [32, 32, 128], name='convBlock2')
    # with tf.name_scope('identifyBlock3'):
    network = myidenBlock(network, [32, 32, 128], name='identifyBlock3')
    # with tf.name_scope('identifyBlock4'):
    network = myidenBlock(network, [32, 32, 128], name='identifyBlock4')
    # with tf.name_scope('identifyBlock5'):
    network = myidenBlock(network, [32, 32, 128], name='identifyBlock5')

    # with tf.name_scope('convblock3'):
    network = myconvBlock(network, [64, 64, 256], name='convBlock3')
    # with tf.name_scope('identifyBlock6'):
    network = myidenBlock(network, [64, 64, 256], name='identifyBlock6')
    # with tf.name_scope('identifyBlock7'):
    network = myidenBlock(network, [64, 64, 256], name='identifyBlock7')
    # with tf.name_scope('identifyBlock8'):
    network = myidenBlock(network, [64, 64, 256], name='identifyBlock8')
    # with tf.name_scope('identifyBlock9'):
    network = myidenBlock(network, [64, 64, 256], name='identifyBlock9')
    # with tf.name_scope('identifyBlock10'):
    network = myidenBlock(network, [64, 64, 256], name='identifyBlock10')

    # with tf.name_scope('convblock4'):
    network = myconvBlock(network, [128, 128, 512], name='convBlock4')
    # with tf.name_scope('identifyBlock11'):
    network = myidenBlock(network, [128, 128, 512], name='identifyBlock11')
    # with tf.name_scope('identifyBlock12'):
    network = myidenBlock(network, [128, 128, 512], name='identifyBlock12')
    # with tf.name_scope('avepool'):

    network = avg_pool_2d(network, kernel_size=1, strides=1, name='avePool')

    # with tf.name_scope('flatten'):
    # network = flatten(network,name='flatten')
    # with tf.name_scope('output'):

    loss = fully_connected(network, numOfEn * 2, activation='relu')
    # loss = dropout(loss,0.7)
    momentum = Momentum(learning_rate=learnRate, lr_decay=0.5, decay_step=30000)
    r2 = R2()
    network = regression(loss, optimizer=momentum, learning_rate=learnRate, loss='mean_square', metric=r2)
    print('i am')
    model = tflearn.DNN(network, checkpoint_path=fileCheckPoint, max_checkpoints=1, tensorboard_verbose=0,
                        tensorboard_dir=fileLogs)
    print('pass')
    ckptFile = tf.train.latest_checkpoint(fileCheckPoint.split('/')[0])
    print(ckptFile)
    if ckptFile:
        print("find a model!")
        model.load(ckptFile, weights_only=False)
        print("finish model loading!")

    return model


def myidenBlock(x, nbFilters, kernelSize=1, name='identifyBlock'):
    k1, k2, k3 = nbFilters
    nameCov1 = name + '_branch1_conv1'
    nameBn1 = name + '_branch1_Bn1'
    nameRelu1 = name + '_branch1_relu1'
    nameCov2 = name + '_branch1_conv2'
    nameBn2 = name + '_branch1_Bn2'
    nameRelu2 = name + '_branch1_relu1'
    nameCov3 = name + '_branch1_conv3'
    nameBn3 = name + '_branch1_Bn3'
    nameOutM = name + '_out_merge'
    nameOutA = name + '_out_relu'

    out = conv_2d(x, k1, 1, 1, name=nameCov1)
    out = batch_normalization(out, name=nameBn1)
    out = activation(out, activation='relu', name=nameRelu1)

    out = conv_2d(out, k2, 1, 1, name=nameCov2)
    out = batch_normalization(out, name=nameBn2)
    out = activation(out, 'relu', name=nameRelu2)

    out = conv_2d(out, k3, 1, 1, name=nameCov3)
    out = batch_normalization(out, name=nameBn3)

    out = merge([out, x], mode='concat', axis=0, name=nameOutM)
    out = activation(out, activation='relu', name=nameOutA)

    return out


def myconvBlock(x, nbFilters, kernelSize=1, name='convBlock'):
    k1, k2, k3 = nbFilters
    nameCov1 = name + '_branch1_conv1'
    nameBn1 = name + '_branch1_Bn1'
    nameRelu1 = name + '_branch1_relu1'
    nameCov2 = name + '_branch1_conv2'
    nameBn2 = name + '_branch1_Bn2'
    nameRelu2 = name + '_branch1_relu1'
    nameCov3 = name + '_branch1_conv3'
    nameBn3 = name + '_branch1_Bn3'
    nameCov4 = name + '_branch2_conv'
    nameBn4 = name + '_branch2_Bn'
    nameOutM = name + '_out_merge'
    nameOutA = name + '_out_relu'
    out = conv_2d(x, k1, filter_size=1, strides=1, name=nameCov1)
    out = batch_normalization(out, name=nameBn1)
    out = activation(out, activation='relu', name=nameRelu1)

    out = conv_2d(out, k2, 1, 1, name=nameCov2)
    out = batch_normalization(out, name=nameBn2)
    out = activation(out, 'relu', name=nameRelu2)

    out = conv_2d(out, k3, 1, 1, name=nameCov3)
    out = batch_normalization(out, name=nameBn3)

    x = conv_2d(x, k3, 1, 1, name=nameCov4)
    x = batch_normalization(x, name=nameBn4)

    out = merge([out, x], mode='concat', axis=0, name=nameOutM)
    out = activation(out, activation='relu', name=nameOutA)

    return out


def getDataProcessed(data):  # csv file only for 2 dimensons

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


totalNum = 320


def edgeProcessed(data, rate):  # csv file only for 2 dimensons
    data = np.array(data, dtype=float)
    data = data + 0.5
    data = np.array(data, dtype=int)
    sizeData = data.shape[0]
    sizeEn = int(data.shape[1] / 2)
    dataList = []
    for i in range(0, sizeData):
        enList = []
        for j in range(0, sizeEn):
            n = data[i, j * 2 + 1]
            addN = n * rate
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
    # turn data into lists of turples:(start,end)
    return dataList


def eva(model, x, y):
    dataTrue = getDataProcessed(y)
    # dataPred = getDataProcessed(model.predict(x))
    dataPred = edgeProcessed(model.predict(x), 1)
    print(dataTrue)
    print(dataPred)

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

            if (dataTrue[i][j][0] == dataPred[i][j][0]) and (dataTrue[i][j][1] == dataPred[i][j][1]):
                pass

    rateNeed = sumAimed / sumNeed
    rateDownload = sumAimed / sumDownload

    averNeed = sumNeed / lenTrue
    averDownload = sumDownload / lenTrue
    averAimed = sumAimed / lenTrue
    print("------ACC-----------")
    print(rateNeed)
    return rateNeed, rateDownload, averNeed, averDownload, averAimed


if __name__ == "__main__":
    fileModel = 'checkPoints/'
    fileDataTrains = "/Users/a/Desktop/finalResMapAssign/data/dataTrain_"
    fileCheckPoints = 'checkPoints/resNetAssign'
    fileLog = 'logs/'
    learnRate = 0.01

    x_train, y_train = getTrainData(fileDataTrains + '01.csv')
    x_train = x_train.reshape(250000, 10, 10, 1)
    # print (y_train[5][0:10])
    # for i in range(2, 7):
    #     fileDataTrain = fileDataTrains + '0' + str(i) + '.csv'
    #     xMid, yMid= getTrainData(fileDataTrain)
    #     x_train = np.concatenate((x_train,xMid))
    #     y_train = np.concatenate((y_train,yMid))
    print(x_train.shape, y_train.shape)
    fileLogs = fileLog
    # model = oldResNetModel( fileCheckPoints, fileLogs, learnRate)
    model = resNetModel(fileCheckPoints, fileLogs, learnRate)
    training = True
    if training:
        model.fit(x_train, y_train, n_epoch=2, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64,
                  snapshot_step=500, snapshot_epoch=False, run_id='resNetAssign')
    else:
        # eva(model,x_train[110000:111000],y_train[110000:111000])
        from qLearning import Environment

        q = Environment(1000)
        x, y = q.getCurrentData()
        print(x)
        print(y)
