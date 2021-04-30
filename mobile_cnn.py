#-*- coding: utf-8 for training model-*-
#C:\Users\vegetableclean\.keras\models
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Convolution2D, BatchNormalization,AveragePooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import keras
from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE
from keras.callbacks import History 
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Input, Dropout
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
class Dataset:
    def __init__(self, path_name):
            #訓練集
        self.train_images = None
        self.train_labels = None

            #驗證集
        self.valid_images = None
        self.valid_labels = None

            #測試集
        self.test_images  = None            
        self.test_labels  = None

            #數據加載路徑
        self.path_name    = path_name

            #當前採用的維度順序
        self.input_shape = None

        #加載數據集並按照交叉驗證的原則劃分數據集並進行相關預處理理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 3, nb_classes = 2):
            #加在數據集到内存
        images, labels = load_dataset(self.path_name)        

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.4, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))                

            #當前的維度順序如果為'th'，則輸入照片的順序為：channels,rows,cols，否則:rows,cols,channels
            #這部分代碼就是根據keras酷要求的維度順序重組訓練數集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            

                #輸出訓練集、驗證集、測試集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

                #我们的模型使用categorical_crossentropy作為損失函數，因此需要根據類別數量nb_classes將
                #類別標前進行one-hot编码使其向量化，在這裡我们的類別只有兩種，經過轉化後標籤數據變為二維
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        

                #像素數據浮點化以便歸一化
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

                #將其歸一化,圖像的各像素值歸一化到0~1區間
            train_images /= 255
            valid_images /= 255
            test_images /= 255            

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels
                
#CNN網絡模型類
class Model_:
    def __init__(self):
        self.model = None

        #建立模型
    def build_model(self, dataset, nb_classes = 2):
            #構建一個空的網絡模型，它是一個線性堆疊模型，各神經網絡層會被順序添加，專業名稱為序貫模型或線性堆疊模型
        
        target_size=128
        input_tensor = Input(shape=(target_size, target_size, 3))
        base_model = MobileNetV2(
        include_top=True,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

        for layer in base_model.layers:
            layer.trainable = True#trainable has to be false in order to freeze the layers
        
        op = Dense(256, activation='relu')(base_model.output)
        #op = Dropout(.05)(op)
    #
        output_tensor = Dense(2, activation='softmax')(op)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)
		#self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])       
        self.model.summary()
          
    
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be 
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.

    #訓練模型
    def train(self, dataset, batch_size = 25, nb_epoch = 10, data_augmentation = True):
        #sgd = SGD(lr = 0.01, decay = 1e-3,
                    #  momentum = 0.9, nesterov = True) #採用SGD+momentum的優化器進行訓練，首先生成一個優化器對象
        self.model.compile(loss='categorical_crossentropy',  
                           optimizer='adam',
                           metrics=['accuracy'])   #完成實際的模型配置工作
        
            #不使用數據提升，所謂的提升就是從我們提供的訓練數據中利用旋轉、翻轉、加噪聲等方法創造新的
            #訓練數據，有意識的提升訓練數據規模，增加模型訓練量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
            #使用實時數據提升
        else:
                #定義數據生成器用於數據提升，其返回一個生成器對象datagen，datagen每被調用一
               #次其生成一組數據（順序生成），節省內存，其實就是python的數據生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使輸入數據去中心化（均值為0），
                samplewise_center  = False,             #是否使輸入數據的每個樣本均值為0
                featurewise_std_normalization = False,  #是否數據標準化（輸入數據除以數據集的標準差）
                samplewise_std_normalization  = False,  #是否將每個樣本數據除以自身的標準差
                zca_whitening = False,                  #是否對輸入數據施以ZCA白化
                rotation_range = 20,                    #數據提升時圖片隨機轉動的角度(範圍為0～180)
                width_shift_range  = 0.2,               #數據提升時圖片水平偏移的幅度（單位為圖片寬度的佔比，0~1之間的浮點數）
                height_shift_range = 0.2,               #同上，只不過這裡是垂直
                horizontal_flip = True,                 #是否進行隨機水平翻轉
                vertical_flip = False)                  #是否進行隨機垂直翻轉

                #計算整個訓練樣本集的數量以用於特徵值歸一化、ZCA白化等處理
            datagen.fit(dataset.train_images)
            
            self.history = History()
                #利用生成器開始訓練模型
            train_history=self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels),shuffle = True)
            #圖形化loss and accc 
            plt.plot(train_history.history['loss'])      #圖示化acc 與val-acc的關係 以及 loss 與val_loss的關係
            plt.plot(train_history.history['val_loss'])  
            plt.title('Train History')  
            plt.ylabel('loss')  
            plt.xlabel('Epoch')  
            plt.legend(['loss', 'val_loss'], loc='upper left')  
            plt.show() 
            plt.plot(train_history.history['acc'])  
            plt.plot(train_history.history['val_acc'])  
            plt.title('Train History')  
            plt.ylabel('acc')  
            plt.xlabel('Epoch')  
            plt.legend(['acc', 'val_acc'], loc='upper left')  
            plt.show() 
              
    MODEL_PATH = './model/gender.mobilev2.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
 
	#识别人脸
    def face_predict(self, image):    
            
#依然是根據後端系統確定維度順序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                             #尺寸必須與訓練集一致都應該是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #與模型訓練不同，這次只是針對1張圖片進行預測 
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    

            #浮点并归一化
        image = image.astype('float32')
        image /= 255
        
            #給出輸入屬於各個類別的概率，我們是二值類別，則該函數會給出輸入圖像屬於0和1的概率各為多少
        result = self.model.predict_proba(image)
        print('result:', result)
        
            #给出類別預測：0或者1
        result = self.model.predict_classes(image)        

            #返回類別預測結果
        return result[0]
		
if __name__ == '__main__':
    dataset = Dataset(r'D:\data_128')
    dataset.load()

    #訓練模型
    
    model = Model_()
    
    model.build_model(dataset)
    
    model.train(dataset)
    model.save_model(file_path = './model/gender.mobilev2.model.h5')#儲存model
    #評估模型
    model = Model_()
    model.load_model(file_path = './model/gender.mobilev2.model.h5')
    model.evaluate(dataset)

    
    
    