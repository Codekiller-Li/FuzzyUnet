# -*- coding: UTF-8 -*-
import glob
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

width = 64
height = 64
channel_num = 3
one_hot_num = 3
aug_num = 5
test_aug_num = 5
test_T_rate = 0.2
test_F_rate = 0.2
"""
    width: 图片Resize的宽度
    height:图片Resize的高度
    channel_num:训练样本通道数
    one_hot_num:图像类别数
    aug_num:对每张训练图片扩充的数量(具有随机性)
    test_aug_num:对每张测试图片扩充的数量(具有随机性)
    test_T_rate:正例样本比例
    test_F_rate:反例样本比例
"""


class DataProcess(object):
    def __init__(self, trainT_path="./ImgData/TrainT/", trainF_path="./ImgData/TrainF/",
                 labelT_path="./ImgData/LabelT/", labelF_path="./ImgData/LabelF/",
                 trainT_aug_path="./ImgData/TrainAugT/", trainF_aug_path="./ImgData/TrainAugF/",
                 labelT_aug_path="./ImgData/LabelAugT/", labelF_aug_path="./ImgData/LabelAugF/",
                 testT_path="./ImgData/TestT/", resultT_path="./ImgData/ResultT/",
                 testF_path="./ImgData/TestF/", resultF_path="./ImgData/ResultF/",
                 img_type="png", img_width=width, img_height=height, npy_path="./NpyData/",
                 train_npy_path="./NpyData/train.npy", label_npy_path="./NpyData/label.npy",
                 channel_num=channel_num, one_hot_num=one_hot_num):
        self.trainT_path = trainT_path
        self.trainF_path = trainF_path
        self.labelT_path = labelT_path
        self.labelF_path = labelF_path
        self.trainT_aug_path = trainT_aug_path
        self.trainF_aug_path = trainF_aug_path
        self.labelT_aug_path = labelT_aug_path
        self.labelF_aug_path = labelF_aug_path
        self.testT_path = testT_path
        self.testF_path = testF_path
        self.resultT_path = resultT_path
        self.resultF_path = resultF_path
        self.img_type = img_type
        self.img_width = img_width
        self.img_height = img_height
        self.npy_path = npy_path
        self.train_npy_path = train_npy_path
        self.label_npy_path = label_npy_path
        self.channel_num = channel_num
        self.one_hot_num = one_hot_num
        self.data_gen = image.ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.03,
            height_shift_range=0.03,
            shear_range=0.03,
            zoom_range=0.03,
            fill_mode='constant',
            horizontal_flip=True,
            vertical_flip=True,
        )
        '''
        rotation_range: 旋转范围, 随机旋转(0 - 180)度;
        width_shift and height_shift: 随机沿着水平或者垂直方向，以图像的长宽小部分百分比为变化范围进行平移;
        rescale: 对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0 - 1
        之间，通常为1 / 255;
        shear_range: 水平或垂直投影变换, 参考这里
        zoom_range: 按比例随机缩放图像尺寸;
        horizontal_flip: 水平翻转图像;
        fill_mode: 填充像素, 出现在旋转或平移之后．constant=黑边,nearest=自动补齐
        horizontal_flip=True水平翻转
        vertical_flip=True竖直反转
        详情请参考Keras图像预处理部分文档
        '''

    def augImage(self, target_width=width, target_height=height):
        """
            输入：
            target_width:resize的宽度
            target_height:resize的高度
            """
        global randomList_T, randomList_F
        self.createFile(self.trainT_aug_path)
        self.createFile(self.trainF_aug_path)
        self.createFile(self.labelT_aug_path)
        self.createFile(self.labelF_aug_path)
        self.createFile(self.testT_path)
        self.createFile(self.testF_path)
        self.createFile(self.resultT_path)
        self.createFile(self.resultF_path)

        trainTList = os.listdir(self.trainT_path)
        trainFList = os.listdir(self.trainF_path)
        labelTList = os.listdir(self.labelT_path)
        labelFList = os.listdir(self.labelF_path)

        trainTList.sort()
        trainFList.sort()
        labelTList.sort()
        labelFList.sort()

        test_T_num = int(test_T_rate * len(trainTList))
        test_F_num = int(test_F_rate * len(trainFList))

        if test_T_rate > 1 or test_F_rate > 1:
            print("测试数据数量超过样本总数，请修改测试数据数量")
        else:
            polulation_T = np.arange(test_T_num)
            randomList_T = random.sample(set(polulation_T), test_T_num)
            randomList_T = np.sort(randomList_T)

            polulation_F = np.arange(test_F_num)
            randomList_F = random.sample(set(polulation_F), test_F_num)
            randomList_F = np.sort(randomList_F)

        train_T_npy = np.ndarray((len(trainTList) - test_T_num, height, width, channel_num), dtype=np.uint8)
        train_F_npy = np.ndarray((len(trainFList) - test_F_num, height, width, channel_num),
                                 dtype=np.uint8)
        label_T_npy = np.ndarray((len(labelTList) - test_T_num, height, width, 1), dtype=np.uint8)
        label_F_npy = np.ndarray((len(labelFList) - test_F_num, height, width, 1), dtype=np.uint8)
        test_T_npy = np.ndarray((test_T_num, height, width, channel_num), dtype=np.uint8)
        test_F_npy = np.ndarray((test_F_num, height, width, channel_num), dtype=np.uint8)
        result_T_npy = np.ndarray((test_T_num, height, width, 1), dtype=np.uint8)
        result_F_npy = np.ndarray((test_F_num, height, width, 1), dtype=np.uint8)

        trainTName = []
        testTName = []
        p = 0
        k = 0
        for i in zip(trainTList, labelTList):
            trainPath = self.trainT_path + i[0]
            labelPath = self.labelT_path + i[1]

            if channel_num == 1:
                trainImage = image.load_img(trainPath, grayscale=True, target_size=(target_height, target_width))
            else:
                trainImage = image.load_img(trainPath, grayscale=False, target_size=(target_height, target_width))

            labelImage = image.load_img(labelPath, grayscale=True, target_size=(target_height, target_width))

            trainData = image.img_to_array(trainImage)
            labelData = image.img_to_array(labelImage)

            if p != test_T_num:
                if k == randomList_T[p]:
                    if channel_num == 1:
                        test_T_npy[p] = trainData[:, :, :1]
                        result_T_npy[p] = labelData
                    else:
                        test_T_npy[p] = trainData
                    result_T_npy[p] = labelData
                    testTName.append(i[0])
                    p += 1
                else:
                    if channel_num == 1:
                        train_T_npy[k] = trainData[:, :, :1]
                    else:
                        train_T_npy[k] = trainData
                    label_T_npy[k] = labelData
                    trainTName.append(i[0])
                    k += 1
            else:
                if channel_num == 1:
                    train_T_npy[k] = trainData[:, :, :1]
                else:
                    train_T_npy[k] = trainData
                label_T_npy[k] = labelData
                trainTName.append(i[0])
                k += 1

        trainFName = []
        testFName = []
        p = 0
        k = 0
        for i in zip(trainFList, labelFList):
            trainPath = self.trainF_path + i[0]
            labelPath = self.labelF_path + i[1]

            if channel_num == 1:
                trainImage = image.load_img(trainPath, grayscale=True, target_size=(target_height, target_width))
            else:
                trainImage = image.load_img(trainPath, grayscale=False, target_size=(target_height, target_width))

            labelImage = image.load_img(labelPath, grayscale=True, target_size=(target_height, target_width))

            trainData = image.img_to_array(trainImage)
            labelData = image.img_to_array(labelImage)

            if p != test_F_num:
                if k == randomList_F[p]:
                    if channel_num == 1:
                        test_F_npy[p] = trainData[:, :, :1]
                        result_F_npy[p] = labelData
                    else:
                        test_F_npy[p] = trainData
                    result_F_npy[p] = labelData
                    testFName.append(i[0])
                    p += 1
                else:
                    if channel_num == 1:
                        train_F_npy[k] = trainData[:, :, :1]
                    else:
                        train_F_npy[k] = trainData
                    label_F_npy[k] = labelData
                    trainFName.append(i[0])
                    k += 1
            else:
                if channel_num == 1:
                    train_F_npy[k] = trainData[:, :, :1]
                else:
                    train_F_npy[k] = trainData
                label_F_npy[k] = labelData
                trainFName.append(i[0])
                k += 1

        dfTrainT = pd.DataFrame(trainTName, columns=['trainTName'])
        dfTrainF = pd.DataFrame(trainFName, columns=['trainFName'])
        dfTestT = pd.DataFrame(testTName, columns=['testTName'])
        dfTestF = pd.DataFrame(testFName, columns=['testFName'])
        dfTrainT.to_csv("./ImgData/trainTName.csv")
        dfTrainF.to_csv("./ImgData/trainFName.csv")
        dfTestT.to_csv("./ImgData/testTName.csv")
        dfTestF.to_csv("./ImgData/testFName.csv")

        print("测试集采集完成")
        print("正例训练集 {} 张 ，反例训练集 {} 张".format(train_T_npy.shape[0], train_F_npy.shape[0]))
        print("正例测试集 {} 张 ，反例测试集 {} 张".format(test_T_npy.shape[0], test_F_npy.shape[0]))
        a = 0
        for i in zip(train_T_npy, label_T_npy):
            trainData = i[0]
            labelData = i[1]

            trainData = trainData[np.newaxis, :, :, :]
            labelData = labelData[np.newaxis, :, :, :]

            random_seed = random.randint(0, 9999)
            self.dataGen(trainData, self.trainT_aug_path, aug_num, random_seed)
            self.dataGen(labelData, self.labelT_aug_path, aug_num, random_seed)
            if a % 100 == 0:
                print("第{}张正例训练集完成".format(a))
            a += 1
        b = 0
        for i in zip(train_F_npy, label_F_npy):
            trainData = i[0]
            labelData = i[1]

            trainData = trainData[np.newaxis, :, :, :]
            labelData = labelData[np.newaxis, :, :, :]

            random_seed = random.randint(0, 9999)
            self.dataGen(trainData, self.trainF_aug_path, aug_num, random_seed)
            self.dataGen(labelData, self.labelF_aug_path, aug_num, random_seed)
            if b % 100 == 0:
                print("第{}张反例训练集完成".format(b))
            b += 1

        c = 0
        for i in zip(test_T_npy, result_T_npy):
            trainData = i[0]
            labelData = i[1]

            trainData = trainData[np.newaxis, :, :, :]
            labelData = labelData[np.newaxis, :, :, :]

            random_seed = random.randint(0, 9999)
            self.dataGen(trainData, self.testT_path, test_aug_num, random_seed)
            self.dataGen(labelData, self.resultT_path, test_aug_num, random_seed)
            if c % 100 == 0:
                print("第{}张正例测试集完成".format(c))
            c += 1

        d = 0
        for i in zip(test_F_npy, result_F_npy):
            trainData = i[0]
            labelData = i[1]

            trainData = trainData[np.newaxis, :, :, :]
            labelData = labelData[np.newaxis, :, :, :]

            random_seed = random.randint(0, 9999)
            self.dataGen(trainData, self.testF_path, test_aug_num, random_seed)
            self.dataGen(labelData, self.resultF_path, test_aug_num, random_seed)
            if d % 100 == 0:
                print("第{}张反例测试集完成".format(d))
            d += 1

        print("数据增强完成")

        self.convertImg(self.labelT_aug_path)
        self.convertImg(self.labelF_aug_path)
        self.convertImg(self.resultT_path)
        self.convertImg(self.resultF_path)
        print("数据转换完成")

        print("共生成训练数据：正例{} ， 反例{}".format(a, b))
        print("共生成训练数据：正例{} ， 反例{}".format(c, d))

    def createFile(self, filepath):
        dir = os.path.split(filepath)[0]
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def dataGen(self, data, file, imgnum, random_seed):
        i = 0
        for _ in self.data_gen.flow(data, save_to_dir=file,
                                    seed=random_seed,
                                    save_format=self.img_type):
            i += 1
            if i >= imgnum:
                break

    def convertImg(self, filepath):
        fileList = os.listdir(filepath)
        for i in fileList:
            path = filepath + i

            img = image.load_img(path, grayscale=False)
            data = image.img_to_array(img)

            data[data == 255] = 2
            data[data == 127] = 1
            data[(data != 0) & (data != 1) & (data != 2)] = 2

            changeImg = Image.fromarray(np.uint8(data[:, :, 0]))
            changeImg.save(path)

    def seeLabel(self, filepath):
        """
        filepath:可视化文件目录
        """
        newPath = filepath[:(len(filepath) - 1)] + "_See/"
        self.createFile(newPath)

        fileList = os.listdir(filepath)
        fileList.sort()

        for i in fileList:
            path = filepath + i
            img = image.load_img(path, grayscale=True)
            data = image.img_to_array(img)

            data[data == 1] = 127
            data[data == 2] = 255

            changeImg = Image.fromarray(np.uint8(data[:, :, 0]))
            changeImg.save(newPath + i)

        print("图像可视化完成")

    def createNpy(self, trainTPath, labelTPath, trainFPath, labelFPath, one_hot=one_hot_num):
        """
            将图片数据转化为npy数据

            :param trainPath: 训练集地址
            :param labelPath: 标签集地址
            :param one_hot:图像标签类别数量，one-hot编码，为了计算损失函数的softmax
            :return:
            """
        self.createFile(self.npy_path)

        trainTList = os.listdir(trainTPath)
        labelTList = os.listdir(labelTPath)
        trainFList = os.listdir(trainFPath)
        labelFList = os.listdir(labelFPath)


        trainTList.sort()
        labelTList.sort()
        trainFList.sort()
        labelFList.sort()
        if trainTList[0]=='.DS_Store':
            trainTList.remove('.DS_Store')
        if labelTList[0]=='.DS_Store':
            labelTList.remove('.DS_Store')

        # print(trainTList)
        # print(trainFList)
        # print(labelTList)
        # print(labelFList)

        # train_npy = np.ndarray((len(trainTList) + len(trainFList), self.img_height, self.img_width, channel_num),
        #                        dtype=np.uint8)
        # label_npy = np.ndarray((len(labelTList) + len(labelFList), self.img_height * self.img_width, one_hot),
        #                        dtype=np.uint8)
        train_npy = np.ndarray((20, self.img_height, self.img_width, channel_num),
                               dtype=np.uint8)
        label_npy = np.ndarray((20, self.img_height * self.img_width, one_hot),
                               dtype=np.uint8)
        k = 0
        for i in zip(trainTList, labelTList):

            img_train = image.load_img(trainTPath + i[0], target_size=(height, width))
            img_label = image.load_img(labelTPath + i[1], target_size=(height, width))
            data_train = image.img_to_array(img_train)
            data_label = image.img_to_array(img_label)
            if channel_num == 1:
                train_npy[k] = data_train[:, :, :1]
            else:
                train_npy[k] = data_train

            data_label = data_label[:, :, 0]
            temp = data_label.flatten()
            label_npy[k] = to_categorical(temp, one_hot_num)
            k += 1
            if k>9:
                break
            if k % 100 == 0:
                print("第{}个保存完成".format(k))

        print("正例样本{}个".format(k))
        num = k

        for i in zip(trainFList, labelFList):
            img_train = image.load_img(trainFPath + i[0], target_size=(height, width))
            img_label = image.load_img(labelFPath + i[1], target_size=(height, width))
            data_train = image.img_to_array(img_train)
            data_label = image.img_to_array(img_label)
            if channel_num == 1:
                train_npy[k] = data_train[:, :, :1]
            else:
                train_npy[k] = data_train
            data_label = data_label[:, :, 0]
            temp = data_label.flatten()
            label_npy[k] = to_categorical(temp, one_hot_num)
            k += 1
            if (k-num)>9:
                break
            if (k - num) % 100 == 0:
                print("第{}个保存完成".format(k - num))

        print("反例样本{}个".format(k - num))

        np.save(self.train_npy_path, train_npy)
        np.save(self.label_npy_path, label_npy)
        print(train_npy.shape)
        print(label_npy.shape)
        print("保存训练数据完成")

    #     for i in zip(trainTList,trainFList):
    #         imgT = image.load_img(trainTPath + i[0], target_size=(height, width))
    #         imgF = image.load_img(trainFPath+i[1],target_size=(height, width))
    #         dataT = image.img_to_array(imgT)
    #         dataF = image.img_to_array(imgF)
    #         if channel_num == 1:
    #             train_npy[k] = img[:, :, :1]
    #         else:
    #             train_npy[k] = img
    #         trainNameList.append(i)
    #         if p != test_num:
    #             if k == randomList[p]:
    #                 if channel_num == 1:
    #                     test_npy[p] = img[:, :, :1]
    #                 else:
    #                     test_npy[p] = img
    #                 testNameList.append(i)
    #                 p += 1
    #             else:
    #                 if channel_num == 1:
    #                     train_npy[k] = img[:, :, :1]
    #                 else:
    #                     train_npy[k] = img
    #                 trainNameList.append(i)
    #                 k += 1
    #         else:
    #             if channel_num == 1:
    #                 train_npy[k] = img[:, :, :1]
    #             else:
    #                 train_npy[k] = img
    #             trainNameList.append(i)
    #             k += 1
    #         t += 1
    #         if t % 100 == 0:
    #             print("第{}张图片转化完成".format(t))
    #
    #     np.save(self.train_npy_path, train_npy)
    #     np.save(self.test_npy_path, test_npy)
    #     print("Train和Test数据保存完成")
    #
    #     k = 0
    #     p = 0
    #     t = 0
    #     for i in labelList:
    #         img = image.load_img(labelPath + i, target_size=(height, width))
    #         img = image.img_to_array(img)
    #         """
    #         img为三通道图,one-hot=3
    #         """
    #         img = img[:, :, 0]
    #         img = np.resize(img, (self.img_height, self.img_width))
    #         """
    #         one-hot
    #         """
    #         temp = img.flatten()
    #         if p != test_num:
    #             if k == randomList[p]:
    #                 result_npy[p] = to_categorical(temp, one_hot_num)
    #                 p += 1
    #             else:
    #                 label_npy[k] = to_categorical(temp, one_hot_num)
    #                 k += 1
    #         else:
    #             label_npy[k] = to_categorical(temp, one_hot_num)
    #             k += 1
    #         t += 1
    #         if t % 100 == 0:
    #             print("第{}张图片转化完成".format(t))
    #
    #     np.save(self.label_npy_path, label_npy)
    #     np.save(self.result_npy_path, result_npy)
    #     print("Label和Result数据保存完成")
    #
    #     """
    #     这部分是黄老师要求的，保存扩充数据集的文件名，便于查找图片的对应关系，使用了pandas库
    #     """
    #     dfTrain = pd.DataFrame(trainNameList, columns=['trainName'])
    #     dfTest = pd.DataFrame(testNameList, columns=['testName'])
    #     dfTrain.to_csv(self.train_name_path)
    #     dfTest.to_csv(self.test_name_path)
    #
    # def testData(self):
    #     train_npy = np.load(self.train_npy_path).astype('float32')
    #     label_npy = np.load(self.label_npy_path).astype('float32')
    #     test_npy = np.load(self.test_npy_path).astype('float32')
    #     result_npy = np.load(self.result_npy_path).astype('float32')
    #     print(train_npy.shape)
    #     print(label_npy.shape)
    #     print(test_npy.shape)
    #     print(result_npy.shape)


if __name__ == '__main__':
    myData = DataProcess()
    # """
    # 扩充图片样本
    # """
    # myData.augImage()

    # myData.seeLabel(myData.labelT_aug_path)
    # """
    # 将label图片可视化，主要作用是方便后期找错和调试
    # 分别将扩充label和原始label保存到AugSee文件夹和See文件夹
    #
    # """
    # myData.seeLabel()
    # # """
    # # 将图片数据保存为npy数据，并流出一部分作为测试数据
    # # """
    myData.createNpy("./ImgData/TrainDataT/", "./ImgData/LabelDataT/", "./ImgData/TrainDataF/", "./ImgData/LabelDataF/")
    # myData.testData()
