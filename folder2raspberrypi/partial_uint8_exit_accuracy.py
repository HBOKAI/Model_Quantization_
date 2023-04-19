import tensorflow as tf 
import numpy as np
import cv2
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    
def model_call(folder_address,img,select_layer=None):
    if(select_layer != None):
        layernums = select_layer+1
    elif(select_layer == None):
        layernums = 7
    start_time=np.zeros(7)
    end_time=np.zeros(7)
    use_time = np.zeros(7)
    layer_use_counter = np.zeros(7)
    for i in range(1,layernums):
        # print(f"第{i}層")
        if i == 1:
            start_time[i] = time.perf_counter()
            predict, params = call(f"{folder_address}/Exit_Model_{i}.tflite", img)
            end_time[i] = time.perf_counter()
            
        else:
            start_time[i] = time.perf_counter()
            predict, params = call(f"{folder_address}/Exit_Model_{i}.tflite", params)
            end_time[i] = time.perf_counter()
            
        use_time[i] = end_time[i]-start_time[i]
        max_softmax = np.max(predict,-1)
        predict_num = np.argmax(predict,-1)
        # print(f"最大softmax: {max_softmax},  此層運算時間: {(end_time[i]-start_time[i])*1000}ms")
        # print(f"預測數字: {predict_num}")
        if select_layer==None:
            if(max_softmax>=0.9):
                layer_use_counter[i] += 1
                break      
    # print(f"總運算時間: {(end_time.sum()-start_time.sum())*1000}ms")
    return predict_num, use_time, layer_use_counter


def evaluate(folder_address,images,labels,select_layer=None):
    accuracy_count = 0
    image_count = images.shape[0]
    use_time=0
    layer_use_counter=0
    for i in tqdm(range(image_count)):
        predict_num, use_time_def, layer_use_counter_def= model_call(folder_address,images[i],select_layer)
        use_time += use_time_def
        layer_use_counter += layer_use_counter_def
        if predict_num == labels[i]:
            accuracy_count += 1
    return accuracy_count/image_count, use_time, layer_use_counter


if __name__=='__main__':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = np.pad(test_images,[[0,0],[2,2],[2,2]],"constant",constant_values=0)
    test_images = np.expand_dims(test_images,-1)
    test_images = test_images.astype(np.float32) / 255.0
    # print(train_images.dtype)

    partial_uint8_accuracy, use_time0, layer_use_counter0 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels)
    partial_uint8_accuracy_exit1, use_time1, layer_use_counter1 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,1)
    partial_uint8_accuracy_exit2, use_time2, layer_use_counter2 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,2)
    partial_uint8_accuracy_exit3, use_time3, layer_use_counter3 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,3)
    partial_uint8_accuracy_exit4, use_time4, layer_use_counter4 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,4)
    partial_uint8_accuracy_exit5, use_time5, layer_use_counter5 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,5)
    partial_uint8_accuracy_exit6, use_time6, layer_use_counter6 = evaluate("./TFLITE_Models/Partial_Uint8_Models",test_images,test_labels,6)
    
    # 各出口準確度
    x=[1,2,3,4,5,6,7]
    y=[round((partial_uint8_accuracy*100),4), round((partial_uint8_accuracy_exit1*100),4), round((partial_uint8_accuracy_exit2*100),4), round((partial_uint8_accuracy_exit3*100),4), round((partial_uint8_accuracy_exit4*100),4), round((partial_uint8_accuracy_exit5*100),4), round((partial_uint8_accuracy_exit6*100),4)]
    plt.bar(x,y,tick_label=["Auto","C1","S2","C3","S4","F5","Final Output"])
    plt.xlabel("Exit layer")
    plt.ylabel("Accuracy(%)")  
    plt.title("Accuracy of each exit")
    for i in range(len(x)):
        plt.text(x[i],y[i],f"{y[i]}%",ha="center",va="bottom")
    plt.show()


    # 各出口耗時
    t1 = (use_time0.sum())/test_images.shape[0]
    t2 = (use_time1.sum())/test_images.shape[0]
    t3 = (use_time2.sum())/test_images.shape[0]
    t4 = (use_time3.sum())/test_images.shape[0]
    t5 = (use_time4.sum())/test_images.shape[0]
    t6 = (use_time5.sum())/test_images.shape[0]
    t7 = (use_time6.sum())/test_images.shape[0]

    x=[1,2,3,4,5,6,7]
    y=[round(t1*1000,4), round(t2*1000,4), round(t3*1000,4), round(t4*1000,4), round(t5*1000,4), round(t6*1000,4), round(t7*1000,4)]
    plt.bar(x,y,tick_label=["Auto","C1","S2","C3","S4","F5","Final Output"])
    plt.xlabel("Exit layer")
    plt.ylabel("Time(ms)")  
    plt.title("Time using of each exit")
    for i in range(len(x)):
        plt.text(x[i],y[i],f"{y[i]}ms",ha="center",va="bottom")
    plt.show()
    

    # 各層耗時
    t1 = (use_time1[1])/test_images.shape[0]
    t2 = (use_time2[2])/test_images.shape[0]
    t3 = (use_time3[3])/test_images.shape[0]
    t4 = (use_time4[4])/test_images.shape[0]
    t5 = (use_time5[5])/test_images.shape[0]
    t6 = (use_time6[6])/test_images.shape[0]
    x=[1,2,3,4,5,6]
    y=[round(t1*1000,4), round(t2*1000,4), round(t3*1000,4), round(t4*1000,4), round(t5*1000,4), round(t6*1000,4)]
    plt.bar(x,y,tick_label=["C1","S2","C3","S4","F5","Final Output"])
    plt.xlabel("Exit layer")
    plt.ylabel("Time(ms)")  
    plt.title("Time using of each layer")
    for i in range(len(x)):
        plt.text(x[i],y[i],f"{y[i]}ms",ha="center",va="bottom")
    plt.show()


    # Auto時各層退出分布(圓餅圖)
    ratio = layer_use_counter0[1]/test_images.shape[0]*100
    ratio1 = layer_use_counter0[2]/test_images.shape[0]*100
    ratio2 = layer_use_counter0[3]/test_images.shape[0]*100
    ratio3 = layer_use_counter0[4]/test_images.shape[0]*100
    ratio4 = layer_use_counter0[5]/test_images.shape[0]*100
    ratio5 = layer_use_counter0[6]/test_images.shape[0]*100
    x = [ratio,ratio1,ratio2,ratio3,ratio4,ratio5]
    plt.pie(x,
            radius=1.5,
            labels=["C1","S2","C3","S4","F5","Final Output"],
            pctdistance=0.8,
            autopct='%.1f%%',
            wedgeprops={'linewidth':3,'edgecolor':'w'})   # %.1f%% 表示顯示小數點一位的浮點數，後方加上百分比符號
    plt.title("Auto exit distribution Map",loc="left")
    plt.show()
    print(f"{ratio}%, {ratio1}%, {ratio2}%, {ratio3}%, {ratio4}%, {ratio5}%")
