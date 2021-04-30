#-*- coding: utf-8 -*-

import cv2
import sys
import gc
from testcnn import Model
import numpy as np
from keras import backend as K
#test image set
IMAGE_SIZE = 150
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #获取图像尺寸
    h, w, _  = image.shape
    
    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)    
    
    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB颜色
    BLACK = [0, 0, 0]
    
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

if __name__ == '__main__':
    #if len(sys.argv) != 2:
    #    print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    #    sys.exit(0)
    from keras.models import load_model
    path_to_model=r'./model/acc85.h5'
    model = load_model(path_to_model)
        #加载模型
    #model = Model()
    #model.load_model(file_path = './model/gender.mobile.model.h5')    

        #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    url=r'http://192.168.50.66:8081'
        #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(url)

        #人脸识别分类器本地存储路径
    cascade_path = (r"C:\Users\vegetableclean\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")   

        #循环检测识别人脸
    while True:
        ret, frame = cap.read()   #读取一帧视频

            #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                
 
            #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                #nop = np.array([None])
                if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
                    image = resize_image(image)                             #尺寸必須與訓練集一致都應該是IMAGE_SIZE x IMAGE_SIZE
                    image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #與模型訓練不同，這次只是針對1張圖片進行預測 
                elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
                    image = resize_image(image)
                    image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    

            #浮点并归一化
                    image = image.astype('float32')
                    image /= 255
                #a=np.asarray(image)
                pred = model.predict(image)
                print(pred)
                    #截取脸部图像提交给模型识别这是谁
                predict=int(np.argmax(pred))
                print(predict)   

                    #如果是“我”
                if predict == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255,0,0), thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'Man', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,0),                           #颜色
                                2)                                     #字的线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,0,255), thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'Woman', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,0,255),                           #颜色
                                2)   

        cv2.imshow("Gender Recognition", frame)

            #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
            #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

        #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()