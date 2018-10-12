from PIL import Image
from keras.models import *
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Reshape, Activation, concatenate,normalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import datetime
from keras.preprocessing import image
import os
import numpy as np
from GetData import DataProcess
import GetData

from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers import merge
from keras.utils import plot_model
from keras.layers import *
from keras.models import Model


class Unet(object):
    def __init__(self, img_width=DataProcess().img_width, img_height=DataProcess().img_height,
                 model_file_path="./Model/"):
        self.img_width = img_width
        self.img_height = img_height
        self.model_file_path = model_file_path

    def createUnet(self):
        model=Sequential()
        inputs = Input(shape=(self.img_height, self.img_width, GetData.channel_num))
        # print("inputs_1:",inputs.shape)
        # inputs=tf.reshape(inputs,[-1,self.img_height*self.img_width, GetData.channel_num])
        # print("inputs_2:",inputs.shape)
        # x=(65536,3)
        # # x=(inputs.shape[1],inputs.shape[2])
        # print(x)
        #print("slice:",tf.slice(inputs,[0,0,0,0],[-1,-1,-1,1]))


        myLayer=MyLayer((self.img_height,self.img_width,GetData.one_hot_num))(inputs)
        print("myLayer_shape:",myLayer)

        myLayer = BatchNormalization(epsilon=1e-6, momentum=0.99)(myLayer)
        # # myLayer=BatchNormalization(axis=3)(myLayer)
        # print("myLayer_shape:",myLayer)
        #
        # merge1 = concatenate([inputs, myLayer], axis=3)
        # print("merge1:",merge1)
        # myLayer =MyLayer()(inputs)

        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(myLayer)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        print("conv1:",conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(GetData.one_hot_num, (1, 1))(conv9)
        reshape1 = Reshape((self.img_width * self.img_height, GetData.one_hot_num))(
            conv10)  # conv10=(?,width*height,class_num)
        reshape2 = Activation(activation='softmax')(reshape1)

        model = Model(inputs=inputs, outputs=reshape2)
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def trainUnet(self, model_temp_name, model_name, trainName, labelName,
                  batch_size, epochs):
        """

        :param model_temp_name: 模型缓存名称，每完成一个epoch会保存一次模型
        eg:比如说我训练5个epoch，当我训练第3个epoch的时候会将前两次epoch训练好的模型保存起来
        :param model_name: 模型的名称
        :param trainName: 训练集的名称，目录在./NpyData/下
        :param labelName: 标签集的名称，目录在./NpyData/下
        :return:
        """
        myData = DataProcess()

        # trainList = os.listdir("./ImgData/TrainData/")
        # labelList = os.listdir("./ImgData/LabelData/")
        #
        # trainList.sort()
        # labelList.sort()
        #
        # train_npy = np.ndarray((len(trainList),myData.img_height,myData.img_width,3),dtype=np.float)
        # train_npy = np.ndarray((len(trainList), myData.img_height*myData.img_width, 1), dtype=np.float)
        #
        # k = 0
        # for i in trainList:
        #     path="./ImgData/TrainData/"+i
        #     img=Image.open(path)
        #     data=image.img_to_array(img)
        #     data /= 255
        #     train_npy[k]=data
        #
        # k = 0
        # for i in labelList:
        #     path="./ImgData/LabelData/"+i
        #     img=Image.open(path)

        #     data=image.img_to_array(img)
        #
        #     train_npy[k]=data

        # myData = DataProcess()
        train_npy = np.load(
            myData.npy_path + trainName).astype('float32')
        train_npy /= 255
        label_npy = np.load(myData.npy_path + labelName).astype('float32')
        print("数据加载完成")
        model = self.createUnet()
        print("网络创建完成")

        modeldir = os.path.split(self.model_file_path)[0]
        if not os.path.isdir(modeldir):
            os.makedirs(modeldir)

        model_checkpoint = ModelCheckpoint(self.model_file_path + model_temp_name, monitor='val_loss', verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

        model.fit(train_npy, label_npy, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1,
                  shuffle=True,
                  callbacks=[model_checkpoint])
        model.save(self.model_file_path + model_name)

    def loadModel(self, model_name, new_model_temp_name, new_model_name, trainName, labelName,
                  batch_size, epochs):
        myData = DataProcess()
        train_npy = np.load(myData.npy_path + trainName).astype('float32')
        train_npy /= 255
        label_npy = np.load(myData.npy_path + labelName).astype('float32')
        print("数据加载完成")
        model = self.createUnet()
        print("网络创建完成")
        model.load_weights(self.model_file_path + model_name)
        model_checkpoint = ModelCheckpoint(self.model_file_path + new_model_temp_name, monitor='val_loss', verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

        model.fit(train_npy, label_npy, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1,
                  shuffle=True,
                  callbacks=[model_checkpoint])

        model.save(self.model_file_path + new_model_name)

import numpy as np
from numpy.linalg import  *
import math

tfd = tf.contrib.distributions

from scipy.stats import multivariate_normal

class MyLayer(Layer):


    def __init__(self, output_dim, **kwargs):
        self.img_height = 64
        self.img_width = 64

        self.output_dim = output_dim
        # 因为重写了init方法，因此需要调用父类的方法
        super(MyLayer, self).__init__(**kwargs)

	# 2. 重写build方法，主要是定义权重.也就是self.kernel
    def build(self, input_shape):
        # self.input_spec = [InputSpec(shape=input_shape)]
        # 1：InputSpec(dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None)
        #Docstring:
        #Specifies the ndim, dtype and shape of every input to a layer.
        #Every layer should expose (if appropriate) an `input_spec` attribute:a list of instances of InputSpec (one per input tensor).
        #A None entry in a shape is compatible with any dimension
        #A None shape is compatible with any shape.

        # 2:self.input_spec: List of InputSpec class instances
        # each entry describes one required input:
        #     - ndim
        #     - dtype
        # A layer with `n` input tensors must have
        # an `input_spec` of length `n`.



        # # print("input_shape:",input_shape[1])
        # shape=self.output_dim
        # print(shape)
        # # print("output_shape:",shape[1])
        # # Create a trainable weight variable for this layer.
        # # self.input_spec = [InputSpec(shape=input_shape)]
        # # print("aa:",self.input_spec)
        # self.kernel = self.add_weight(name='kernel',
        #                          shape=(input_shape[1], shape[1],),
        #                          initializer='uniform',
        #                          trainable=True)
        # # self.trainable_weights=[]
        # 这个方法必须设置`self.built=True'，继承父类方法即可
        print("input_shape:",input_shape)
        print("output_dim:",self.output_dim)
        self.scale=self.add_weight(name='v1',shape=(GetData.one_hot_num,3,3), initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2',shape=(GetData.one_hot_num,self.img_height*self.img_width,3), initializer='uniform', trainable=True)
        # self.gamma = K.variable(tf.ones(shape=(3,3,3)),name='v1')
        # self.beta = K.variable(tf.ones(shape=(3,self.img_width*self.img_height,3)),name='v2')
        #
        #
        #
        # self.trainable_weights = [self.gamma, self.beta]
        #
        # if self.initial_weights is not None:
        #     self.set_weights(self.initial_weights)
        #     del self.initial_weights
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    		# call和compute方法都不建议使用self绑定，因为可能会重名
    # 3. call(x) 主要是运算部分，只需要传入inputs，返回运算结果
    def call(self, x):
        # multivariate_normal.pdf(x, mean=mean, cov=cov)
        print("scale:",self.scale)
        print("mean:",self.mean)
        print("x:",x)
        #
        # # result=tf.placeholder(dtype='int32', shape=[None,1])
        # # temp1=tf.placeholder(dtype='int32', shape=[None,1])
        # # temp2=tf.placeholder(dtype='int32', shape=[None,1])
        # # # temp3=tf.placeholder(dtype='int32', shape=[None,1,1])
        # # a=tf.stack([temp1,temp2],axis=2)
        # # print("temp:",a)
        # # print("temp1:",tf.concat([a,temp2],axis=2))
        # # a=tf.concat([temp1,temp2],axis=2)
        # # print("aa:",tf.concat([a,temp3],axis=2))
        # # result=[]
        # # print("result:",result.shape)
        # #test=tf.placeholder(dtype='int32', shape=[None,1])
        # test=[]
        #
        # for i in range(x.shape[1]):
        #     aaa=[]
        #     new_tensor=tf.slice(x,[0,i,0],[-1,1,3])
        #     for k in range(4):
        #         # print("k:",k)
        #         # print("i:",i)
        #
        #         X_T=tf.matmul(new_tensor[:,0,:],tf.matrix_inverse(self.gamma[k]))
        #         X=tf.matmul(tf.reshape(X_T,[-1,1,3]),tf.reshape(new_tensor,[-1,3,1]))
        #         res=tf.reshape(tf.exp(-0.5*X),[-1,1])
        #
        #         # print("res:",res.shape)
        #         # print("res_type",tf.rank(res))
        #         aaa.append(res)
        #         # print(result[:,i:i+1,k:k+1].shape)
        #         # result=result[:,i:i+1,k:k+1].assign(res)
        #         # result.append(res)
        #         # result[:,i:i+1,k:k+1]=res
        #         # result[:,i:i+1,k:k+1]=tf.assign(tf.slice(result,[0,i,k],[-1,1,1]),res)
        #     temp_aa=tf.transpose(tf.convert_to_tensor(aaa))
        #     temp_aa=tf.reshape(temp_aa,[-1,4])
        #     add_sum=tf.reduce_sum(temp_aa,axis=1)
        #     add_sum=tf.reshape(add_sum,[-1,1])
        #     # print("add_sum:",tf.reshape(add_sum,[-1,1]))
        #     for k in range(4):
        #         aaa[k]=aaa[k]/add_sum
        #     # print("aaa:",aaa)
        #     # if i>3:
        #     #     break
        #     test.append(aaa)
        #     print("num:",i)
        # output_temp=tf.transpose(tf.convert_to_tensor(test))
        # print("test_list:",tf.convert_to_tensor(test))
        # bb=tf.convert_to_tensor(test)
        # # bb=tf.reshape(bb,[-1,11,4])
        #
        # output=tf.reshape(tf.transpose(bb),[-1,self.img_height, self.img_width,output_temp.shape[2]])
        # print("output:",output.shape)
        # print("over")
        # # output = tf.reshape(tf.stack(result), shape=temp.shape)
        # # print("output:",output)
        # # print("result_shape",result)
        #
        #         # if i>0:
        #         #     break
        #         # print("gamma_new_shape:",self.gamma.shape)
        #         # print("log_aa:",new_tensor.shape)
        #         # print("v1:",new_tensor-self.beta[0])
        #         # new_tensor=new_tensor-self.beta[0]
        #         # print("v1_trans:",tf.reshape(new_tensor,[-1,3]))
        #         # print("log:",tf.matmul(new_tensor[:,0,:],tf.matrix_inverse(self.gamma[0])))
        #         # temp=tf.matmul(new_tensor[:,0,:],tf.matrix_inverse(self.gamma[0]))
        #         #
        #         # print("ni:",tf.matmul(tf.reshape(temp,[-1,1,3]),tf.reshape(new_tensor,[-1,3,1])))
        #         # print("exp:",tf.exp(-0.5*tf.matmul(tf.reshape(temp,[-1,1,3]),tf.reshape(new_tensor,[-1,3,1])))/
        #         #       tf.pow(2*math.pi,1.5)*tf.sqrt(tf.matrix_determinant(self.gamma[0])))
        #         # print("num:",)
        #
        # # new_gamma=tf.variables([4,3,3,65536],name="nv1")
        # # new_beta=tf.Variable(np.ones(65536,4,3),name="nv2")
        # # print("new_beta:",new_beta.shape)
        #
        # # print(new_beta.shape)
        # # for i in range(new_beta.shape[1]):
        # #     new_beta[:,i,:]=self.beta
        #
        # # y=math.exp(-0.5*(x-self.beta[0])*(tf.matrix_inverse(self.gamma[0]))*(tf.transpose(x-self.beta[0])))/\
        # #   pow(2*math.pi)*math.sqrt(det(self.gamma[0]))
        # # x *= self.kernel
        # # print("test:",x)
        # # print(x.shape)
        # # print(self.kernel.shape)
        # # x *= self.kernel
        # # print(result)
        # return output
        x=tf.reshape(x,[-1,self.img_height*self.img_width,3])
        output=[]
        # for i in range(x.shape[1]):
        #     temp=[]
        #     for num in range(3):
        #         print("num:",num)

        for i in range(GetData.one_hot_num):
            mvn = tfd.MultivariateNormalTriL(
                    loc=self.mean[i],
                    scale_tril=self.scale[i])
            gauss=mvn.prob(x)
            # mvn.prob(x)
            # print("i:,",i)
            # print("prob",mvn.prob(x))
            for j in range(GetData.channel_num):
                print("i:{} , j:{} , gauss:{} , x:{}".format(i,j,gauss,tf.reshape(tf.slice(x,[0,0,j],[-1,-1,1]),[-1,self.img_height*self.img_width])))
                temp=tf.multiply(gauss,tf.reshape(tf.slice(x,[0,0,j],[-1,-1,1]),[-1,self.img_height*self.img_width]))
                # print("tag:",mvn.prob(x))
                output.append(temp)

        # new_output=[]
        # for i in range(x.shape[1]):
        #     print("num:",i)
        #     sum=[]
        #     new_sum=[]
        #     for num in range(3):
        #         temp=tf.slice(output[num],[-1,i],[-1,1])
        #         sum.append(temp)
        #     for num in range(3):
        #         temp=tf.slice(output[num],[-1,i],[-1,1])
        #         new_sum.append(temp/sum[num])
        #     new_output.append(new_sum)
        # output=tf.reshape(tf.convert_to_tensor(new_output),[-1,self.img_height,self.img_width,3])

        output=tf.reshape(tf.convert_to_tensor(output),[-1,self.img_height,self.img_width,GetData.one_hot_num*GetData.channel_num])
        return output

	# 4. 如果输入与输出的shape不一致，这里应该定义shaoe变化的逻辑，折让keras能够自动推断各层的形状
    def compute_output_shape(self, input_shape):
        # print("output_shape",self.output_dim)
        output_shape=(input_shape[0],self.output_dim[0],self.output_dim[1],GetData.one_hot_num*GetData.channel_num)
        print("myLayer_input_shape:",output_shape)
        print("myLayer_input_shape:",type(output_shape))
        # print("result:",input_shape)
        return output_shape




if __name__ == '__main__':
    myUnet = Unet()
    batch_size = 2
    epochs = 10

    nowTime = datetime.datetime.now().strftime('%m%d_%H_%M')
    myUnet.trainUnet("temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs,nowTime),
                     "unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs,nowTime), "train.npy", "label.npy"
                     , batch_size, epochs)
    # myUnet.loadModel("unet_batchsize_2_epochs_8_0511_13_22.h5",
    #                  "temp_batchsize_2_epochs_13_{}.h5".format(nowTime),
    #                  "unet_batchsize_2_epochs_13_{}.h5".format(nowTime), "train.npy", "label.npy"
    #                  , batch_size, epochs)
    """
    载入已经训练好的Model，继续训练
    myUnet.loadModel("unet.h5","new_temp.h5","new_unet.h5","train.npy", "label.npy")
    """

    # x=[[1,4,5],[1,2,3]]
    # mean=[[1,1,1],[1,1,1]]
    # cov=[[1,0,0],[0,1,0],[0,0,1]]
    #
    # y = multivariate_normal.pdf(x, mean=mean, cov=cov)
    # print(y)

    # train_npy = np.load("./NpyData/train.npy").astype('float32')
    # train_npy /= 255
    # train_npy=train_npy[0]
    # train_npy=np.reshape(train_npy,[256*256,3])
    #
    # mean=[1,1,1]
    # cov=[[1,0,0],[0,1,0],[0,0,1]]
    # for i in range(256*256):
    #     x=train_npy[i]
    #     y=multivariate_normal.pdf(x, mean=mean, cov=cov)
    #     print(y)
    #     # print(i)

#     tfd = tf.contrib.distributions
#
# # Initialize a single 3-variate Gaussian.
# mu = [[1., 2., 3.],[8.,4.,7.]]
# cov = [[ 0.36,  0.12,  0.06],
#        [ 0.12,  0.29, -0.13],
#        [ 0.06, -0.13,  0.26]]
# scale = tf.cholesky(cov)
# # ==> [[ 0.6,  0. ,  0. ],
# #      [ 0.2,  0.5,  0. ],
# #      [ 0.1, -0.3,  0.4]])
# mvn = tfd.MultivariateNormalTriL(
#     loc=mu,
#     scale_tril=scale)
# a=tf.constant([[2.,1.,6.],[3.,5.,2.]],name="a")
#
# sess=tf.Session()
# with sess.as_default():
#     print(mvn.prob(a).eval())

# mvn.mean().eval()
# # ==> [1., 2, 3]
#
# # Covariance agrees with cholesky(cov) parameterization.
# mvn.covariance().eval()
# # ==> [[ 0.36,  0.12,  0.06],
# #      [ 0.12,  0.29, -0.13],
# #      [ 0.06, -0.13,  0.26]]
#
# # Compute the pdf of an observation in `R^3` ; return a scalar.
# mvn.prob([-1., 0, 1]).eval()  # shape: []
#
# # Initialize a 2-batch of 3-variate Gaussians.
# mu = [[1., 2, 3],
#       [11, 22, 33]]              # shape: [2, 3]
# tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
# mvn = tfd.MultivariateNormalTriL(
#     loc=mu,
#     scale_tril=tril)
#
# # Compute the pdf of two `R^3` observations; return a length-2 vector.
# x = [[-0.9, 0, 0.1],
#      [-10, 0, 9]]     # shape: [2, 3]
# mvn.prob(x).eval()    # shape: [2]
#
# # Instantiate a "learnable" MVN.
# dims = 4
# with tf.variable_scope("model"):
#   mvn = tfd.MultivariateNormalTriL(
#       loc=tf.get_variable(shape=[dims], dtype=tf.float32, name="mu"),
#       scale_tril=tfd.fill_triangular(
#           tf.get_variable(shape=[dims * (dims + 1) / 2],
#                           dtype=tf.float32, name="chol_Sigma")))
