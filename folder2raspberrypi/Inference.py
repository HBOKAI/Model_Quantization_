import tensorflow as tf
import numpy as np
import cv2
import time
print(f"tensorflow version: {tf.__version__}")

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = np.pad(train_images,[[0,0],[2,2],[2,2]],"constant",constant_values=0)
train_images = np.expand_dims(train_images,-1)
train_images = train_images.astype(np.float32) / 255.0
# print(train_images.dtype)

test_images = np.pad(test_images,[[0,0],[2,2],[2,2]],"constant",constant_values=0)
test_images = np.expand_dims(test_images,-1)
test_images = test_images.astype(np.float32) / 255.0
# print(train_images.dtype)

def call(interpreter_path,img):
    #加载模型并分配张量
    interpreter = tf.lite.Interpreter(model_path=interpreter_path)
    interpreter.allocate_tensors()

    #获得输入输出张量.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(output_details)
    index = input_details[0]['index']
    shape = input_details[0]['shape']
    # print("當前輸入圖片格式: ",img.shape)
    # print("當前輸入圖片所需格式: ",shape)
    interpreter.set_tensor(index, img.reshape(shape).astype("float32"))
    interpreter.invoke()
    if output_details[0]['shape'].shape==(2,) and output_details[0]['shape'][1]==10:
        predict = interpreter.get_tensor(output_details[0]['index'])
        try:
            params = interpreter.get_tensor(output_details[1]['index'])
            return predict,params
        except IndexError:
            return predict,0
    else:
        predict = interpreter.get_tensor(output_details[1]['index'])
        params = interpreter.get_tensor(output_details[0]['index'])
        return predict,params


def model_call(folder_address,img):
    start_time=np.zeros(7)
    end_time=np.zeros(7)
    for i in range(1,7):
        print(f"第{i}層")
        if i == 1:
            start_time[i] = time.perf_counter()
            predict, params = call(f"{folder_address}/Exit_Model_{i}.tflite", img)
            end_time[i] = time.perf_counter()
            
        else:
            start_time[i] = time.perf_counter()
            predict, params = call(f"{folder_address}/Exit_Model_{i}.tflite", params)
            end_time[i] = time.perf_counter()
            
        max_softmax = np.max(predict,-1)
        predict_num = np.argmax(predict,-1)
        print(f"最大softmax: {max_softmax},  此層運算時間: {(end_time[i]-start_time[i])*1000}ms")
        print(f"預測數字: {predict_num}")
        if(max_softmax>=0.9):
            break      
    print(f"總運算時間: {(end_time.sum()-start_time.sum())*1000}ms")
    return predict_num

def show_xy(event,x,y,flags,param):
    global dots, draw,img_gray                    # 定義全域變數
    if flags == 1:
        if event == 1:
            dots.append([x,y])            # 如果拖曳滑鼠剛開始，記錄第一點座標
        if event == 4:
            dots = []                     # 如果放開滑鼠，清空串列內容
        if event == 0 or event == 4:
            dots.append([x,y])            # 拖曳滑鼠時，不斷記錄座標
            x1 = dots[len(dots)-2][0]     # 取得倒數第二個點的 x 座標
            y1 = dots[len(dots)-2][1]     # 取得倒數第二個點的 y 座標
            x2 = dots[len(dots)-1][0]     # 取得倒數第一個點的 x 座標
            y2 = dots[len(dots)-1][1]     # 取得倒數第一個點的 y 座標
            cv2.line(draw,(x1,y1),(x2,y2),(255,255,255),20)  # 畫直線
        cv2.imshow('img', draw)#draw


dots = []   # 建立空陣列記錄座標
w = 320
h = 320
draw = np.zeros((h,w,3), dtype='uint8')   # 建立 420x240 的 RGBA 黑色畫布

while True:
    cv2.imshow('img', draw)
    cv2.setMouseCallback('img', show_xy)
    keyboard = cv2.waitKey(5)                    # 每 5 毫秒偵測一次鍵盤事件
    if keyboard == ord('q'):
        break                                    # 按下 q 就跳出

    if keyboard == ord('n'):
        img_gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)   # 轉為灰度圖
        img = cv2.resize(img_gray,(32,32))                          # 變更圖片尺寸
        cv2.imwrite(".\images\gray.png",img)
        img = img/255
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,-1)
        
        model_call("./TFLITE_Models/float16_Models",img)
        print("\n")
        draw = np.zeros((h,w,3), dtype='uint8')
    if keyboard == ord('r'):
        draw = np.zeros((h,w,3), dtype='uint8')  # 按下 r 就變成原本全黑的畫布
        cv2.imshow('img', draw)