import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from GetData import DataProcess
from Unet import Unet

test_num = 50


class DataPredict(object):
    def __init__(self, predict_path="./Predict/", predict_npy_path="./Predict/predict.npy",
                 test_npy_path="./Predict/test.npy", result_npy_path="./Predict/result.npy",
                 test_name_list_post="./Predict/testPostName.csv"):
        self.predict_npy_path = predict_npy_path
        self.predict_path = predict_path
        self.test_npy_path = test_npy_path
        self.result_npy_path = result_npy_path
        self.test_name_list_post = test_name_list_post

    """
    predict_npy_path:one-hot编码的得到的值
    new_predict_npy_path:预测图片的值shape=(num,width,height,1)
    data_csv_paht:Predict图片和Result图片，即预测值和真实值
    result_csv_path:评估结果，包括准确率召回率，IOU等数据
    """

    def createData(self, trainPath="./NpyData/train.npy", labelPath="./NpyData/label.npy", k=test_num):
        """
        随机生成k张测试图片和对应的标签，保存到Predict/test和Predict/result
        每次测试都会覆盖掉原来的test和result的npy数据
        :param num:
        :return:
        """
        myData = DataProcess()
        train_npy = np.load(trainPath).astype('float32')
        label_npy = np.load(labelPath).astype('float32')

        """
        train:(5598, 256, 256, 3)
        test:(5598, 65536, 3)
        """

        polulation = np.arange(train_npy.shape[0])
        randomList = random.sample(set(polulation), k)

        test_npy = np.ndarray((k, myData.img_height, myData.img_width, myData.channel_num), dtype=np.uint8)
        result_npy = np.ndarray((k, myData.img_height * myData.img_width, myData.one_hot_num), dtype=np.uint8)

        """
        test_npy.shape=(k,256,256,3)
        result_npy.shape=(k,256*256,3)
        """
        testNameList = []
        dfTrain = pd.read_csv(myData.train_name_path)
        trainTemp = dfTrain['trainName']
        trainNameList = list(trainTemp)

        num = 0
        for i in randomList:
            test_npy[num] = train_npy[int(i)]
            result_npy[num] = label_npy[int(i)]
            testNameList.append(trainNameList[i])
            num += 1

        predictdir = os.path.split(self.predict_path)[0]
        if not os.path.isdir(predictdir):
            os.makedirs(predictdir)

        dfTest = pd.DataFrame(testNameList, columns=['testName'])
        dfTest.to_csv(self.test_name_list_post)

        np.save(self.test_npy_path, test_npy)
        np.save(self.result_npy_path, result_npy)

        return test_npy, result_npy

    def predictData(self, model_name, test_npy_path="./NpyData/test.npy",
                    result_npy_path="./NpyData/result.npy"):
        """

        :param model_name: 模型名称
        :param mode: 0表示预测从训练集中随机抽取的数据，默认状态是0；
        1表示单独拿出来做测试集的数据预测其泛化能力，若输入1，则需要修改test和result的path
        :param test_npy_path: 在mode==1情况下表示测试集的地址
        :param result_npy_path: 在mode==1情况下表示测试集标签的地址
        :return:
        """
        global test_npy, result_npy
        myUnet = Unet()

        test_npy = np.load(test_npy_path).astype('float32')
        result_npy = np.load(result_npy_path).astype('float32')

        test_npy /= 255
        model = myUnet.createUnet()
        model.load_weights(Unet().model_file_path + model_name)
        print("开始预测")
        predict_npy = model.predict(test_npy, batch_size=2, verbose=0)
        print("预测完成")

        np.save(self.predict_npy_path, predict_npy)

        return test_npy, result_npy, predict_npy

    #
    def evaluateData(self,testPath,labelPath, evaluate_name, target=1):
        """
        test_npy, result_npy, predict_npy=self.predictData(model_name)

        将predict变成三通道one-hot编码，基于概率最大选择
                                pred
        confusion_matrix= true  TP  FN
                                FP  TN

        IOU/Jaccard = TP / (TP + FN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_measure = 2 * TP / (2 * TP + FP + FN)

        target:要检测的目标在one-hot之后切片的位置

        evaluate_name:测试结果保存到csv文件的path

        """
        myData = DataProcess()
        myUnet = Unet()
        testList = os.listdir(testPath)
        labelList = os.listdir(labelPath)

        testList.sort()
        labelList.sort()

        test_npy = np.ndarray((len(testList), myData.img_height, myData.img_width, myData.channel_num), dtype=np.float)
        label_npy = np.ndarray((len(testList), myData.img_height * myData.img_width, myData.one_hot_num), dtype=np.float)

        k = 0
        for i in testList:
            path = testPath + i
            img = image.load_img(path, target_size=(myData.img_height, myData.img_width))
            data = image.img_to_array(img)
            data /= 255
            test_npy[k] = data
            k += 1

        k = 0
        for i in labelList:
            path = labelPath + i
            img = image.load_img(path, target_size=(myData.img_height, myData.img_width))
            data = image.img_to_array(img)
            data = data[:, :, 0]
            temp = data.flatten()
            label_npy[k] = to_categorical(temp, myData.one_hot_num)
            k += 1
        """
        npy导入
        """
        # test_npy = np.load("./NpyData/test.npy").astype('float32')
        # label_npy = np.load("./NpyData/result.npy").astype('float32')
        # test_npy /= 255

        print(test_npy.shape)
        print(label_npy.shape)

        model = myUnet.createUnet()
        model.load_weights("./Model/unet_batchsize_2_epochs_13_0511_21_04.h5")
        predict_npy = model.predict(test_npy, batch_size=2, verbose=0)
        print("预测完成")

        # myData = DataProcess()
        # # result_npy = np.load("./").astype("float32")
        # predict_npy = np.load(self.predict_npy_path).astype("float32")

        # dfTest = pd.read_csv(self.test_name_list_post)
        # testTemp = dfTest['testName']

        # name = pd.read_csv("./ImgData/testName.csv")
        # name = name['testName']

        predict = np.argmax(predict_npy, axis=2)
        print("log:", predict.shape)
        """
        选取三个通道里概率值比较大的那个通道作为我们的预测值
        """

        # data = np.zeros((test_num, 3))
        # for i in range(test_num):
        #     """
        #     统计predict图片0，1，2的数量
        #     """
        #     data[i, 0] = len(list(filter(lambda x: x == 0, predict[i, :])))
        #     data[i, 1] = len(list(filter(lambda x: x == 1, predict[i, :])))
        #     data[i, 2] = len(list(filter(lambda x: x == 2, predict[i, :])))
        #
        # df = pd.DataFrame(data, columns=['0', '1', '2'], index=testNameList)
        # df.to_csv("./Predict/predictData.csv")

        predict_result = np.ndarray((predict.shape[0], predict.shape[1], myData.one_hot_num), dtype=np.uint8)

        evaluateList = np.ndarray((test_num, 8), dtype=np.float32)

        y_true = np.ndarray((predict.shape[1],1),dtype=np.float)
        y_pred = np.ndarray((predict.shape[1], 1), dtype=np.float)

        for i in range(predict.shape[0]):
            predict_result[i] = to_categorical(predict[i], myData.one_hot_num)

            y_true[:,0] = label_npy[i, :, target]
            y_pred[:,0] = predict_result[i, :, target]
            # y_true=np.reshape(y_true,predict.shape[0])
            # y_pred = np.reshape(y_pred, predict.shape[0])
            # y_true=np.array(y_true)
            # y_pred=np.array(y_pred)
            # print(y_true.shape)
            # print(y_pred.shape)
            if np.sum(y_true==0)==predict.shape[1] and np.sum(y_pred==0)==predict.shape[1]:
                TN, FP, FN, TP = predict.shape[1],0,0,0
            else:
                TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

            evaluateList[i, 0] = TP
            evaluateList[i, 1] = FN
            evaluateList[i, 2] = FP
            evaluateList[i, 3] = TN
            if TP==0 and FN==0 and FP==0:
                Iou=0
                precision=0
                recall=0
                F1_measure=0
            else:
                Iou = TP / (TP + FN + FP)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                F1_measure = 2 * TP / (2 * TP + FP + FN)

            evaluateList[i, 4] = Iou
            evaluateList[i, 5] = precision
            evaluateList[i, 6] = recall
            evaluateList[i, 7] = F1_measure

        columns = ['TP', 'FN', 'FP', 'TN', 'IOU/Jaccard', 'Precision', 'Recall', 'F1_measure']
        evaluateDF = pd.DataFrame(evaluateList, columns=columns)
        print("保存评估结果")
        evaluateDF.to_csv(self.predict_path + evaluate_name)

        np.save("./Predict/describe.npy", predict_result)


#
#     def evaluateData(self):
#         result_npy = np.load(self.result_npy_path).astype("float32")
#         predict_npy = np.load(self.predict_npy_path).astype("float32")
#
#         predict = np.argmax(predict_npy, axis=2)
#         result = np.argmax(result_npy, axis=2)
#
#         # dfTest = pd.read_csv(self.test_name_list_post)
#         # testTemp = dfTest['testName']
#         # testNameList = list(testTemp)
#
#         data = np.zeros((test_num, 3))
#         for i in range(test_num):
#             """
#             统计predict图片0，1，2的数量
#             """
#             data[i, 0] = len(list(filter(lambda x: x == 0, predict[i, :])))
#             data[i, 1] = len(list(filter(lambda x: x == 1, predict[i, :])))
#             data[i, 2] = len(list(filter(lambda x: x == 2, predict[i, :])))
#
#         # df = pd.DataFrame(data, columns=['0', '1', '2'], index=testNameList)
#         # df.to_csv("./Predict/predictData.csv")
#
#         data = np.zeros((test_num, 3))
#         for i in range(test_num):
#             data[i, 0] = len(list(filter(lambda x: x == 0, result[i, :])))
#             data[i, 1] = len(list(filter(lambda x: x == 1, result[i, :])))
#             data[i, 2] = len(list(filter(lambda x: x == 2, result[i, :])))
#         df = pd.DataFrame(data, columns=['0', '1', '2'])
#         df.to_csv("./Predict/resultData.csv")
#         """
#         predict=(100,256*256),0,1,2
#         result=(100,256*256),0,1,2
#         """
#         predict[predict == 2] = 0
#         result[result == 2] = 0
#
#         evaluateList = np.ndarray((test_num + 1, 9), dtype=np.float32)
#
#         sumIOU = 0
#         existTumorNum = 0
#         for i in range(predict.shape[0]):
#             TN, FP, FN, TP = confusion_matrix(result[i, :], predict[i, :]).ravel()
#
#             evaluateList[i, 0] = TP
#             evaluateList[i, 1] = FN
#             evaluateList[i, 2] = FP
#             evaluateList[i, 3] = TN
#
#             Iou = TP / (TP + FN + FP)
#             precision = TP / (TP + FP)
#             recall = TP / (TP + FN)
#             F1_measure = 2 * TP / (2 * TP + FP + FN)
#
#             evaluateList[i, 4] = Iou
#             evaluateList[i, 5] = precision
#             evaluateList[i, 6] = recall
#             evaluateList[i, 7] = F1_measure
#             """
#             1代表该图片存在肿瘤
#             0代表该图片不存在肿瘤
#             阈值是100个像素点
#             """
#             if TP + FN < 100:
#                 evaluateList[i, 8] = 0
#             else:
#                 evaluateList[i, 8] = 1
#
#                 existTumorNum += 1
#                 sumIOU += Iou
#
#         """
#         计算存在肿瘤的图片的IOU平均值
#         """
#         # evaluateList[test_num, :]=np.zeros(9)
#         # evaluateList[test_num, 4] = sumIOU / existTumorNum
#         #
#         # testNameList.append("average")
#
#         columns = ['TP', 'FN', 'FP', 'TN', 'IOU/Jaccard', 'Precision', 'Recall', 'F1_measure', 'existTumor']
#         evaluateDF = pd.DataFrame(evaluateList, columns=columns)
#         print("保存评估结果")
#         evaluateDF.to_csv("./Predict/info.csv")
#
#
# def seeRGB(path):
#     """
#     可以通过debug查看图片像素
#     :return:
#     """
#     img = Image.open(path)
#     data = np.array(img)
#     print(data)
#
#
# def seeNpy():
#     """
#     可以通过debug查看测试集标签和预测结果的值
#     :return:
#     """
#     result_npy = np.load("./Predict/result.npy").astype("float32")
#     predict_npy = np.load("./Predict/new_predict.npy").astype("float32")
#     print(result_npy.shape)
#     print(predict_npy.shape)
#
#
if __name__ == '__main__':
    myPredict = DataPredict()
    """
    随机抽取测试数据集
    """
    # myPredict.createData()
    # myPredict.predictData("unet_batchsize_2_epochs_10.h5")
    myPredict.evaluateData("ImgData/TestT/","ImgData/ResultT","report_50_0513_Train.csv")
    # seeRGB("./ImgData/AugSee/_0_2441.png")
    # seeNpy()

# import numpy as np
#
# arr = np.arange(24).reshape((2,3,4))
# arr =np.reshape(arr,(24,1))
# print(arr.shape)
