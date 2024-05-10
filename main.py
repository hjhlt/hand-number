# author:Hurricane
# date:  2020/11/6
# E-mail:hurri_cane@qq.com

import cv2 as cv
import numpy as np
import os
from Pre_treatment import get_number as g_n
from Pre_treatment import get_roi
import predict as pt
from time import time
from Pre_treatment import softmax
import tkinter
from tkinter import filedialog
from tkinter import font
from PIL import Image, ImageTk

# 实时检测视频
# capture = cv.VideoCapture(0, cv.CAP_DSHOW)
# capture.set(3, 1280)
# capture.set(4, 1080)
net = pt.get_net()

# img_path = r'.\5.jpg'
# frame = cv.imread(img_path)
# img_bw = g_n(frame)
# img_bw_sg = get_roi(img_bw)
# print(img_bw)
# cv.imshow("img", img_bw_sg)
# cv.waitKey(0)

root = tkinter.Tk()
root.geometry("1600x900")
root.resizable(False, False)
root.title('手写数字识别系统')
video = cv.VideoCapture(0)
res = video.set(3, 1280)
label1 = tkinter.Label(root, text='原图像')
label2 = tkinter.Label(root, text='预处理后的图像')

label1.place(x=0, y=0)
label2.place(x=900, y=0)
# 创建label标签
image1 = tkinter.Label(root, text=' ', width=700, height=700)
image1.place(x=0, y=25, width=700, height=700)

image2 = tkinter.Label(root, text=' ', width=700, height=700)
image2.place(x=900, y=25, width=700, height=700)

my_font = font.Font(size=15)
ans = tkinter.Label(root, text='未识别到数字', font=my_font)
ans.place(x=10, y=775, width=200, height=100)

cnt = 0
mode = 0
first = 1


def pre(img):
    x, y, z = img.shape
    return img[x // 2 - 350:x // 2 + 349, y // 2 - 350:y // 2 + 349]


def enlarge(img):
    result = np.zeros((700, 700))
    for i in range(28):
        for j in range(28):
            result[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = img[i][j]
    return result


def maxPool(img):
    img = np.array(img)
    x, y = img.shape
    result = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            result[i, j] = np.max(img[i * x // 28:(i + 1) * x // 28, j * y // 28:(j + 1) * y // 28])
    return result


def open_image():
    global video
    global root
    global image1
    global image2
    global ans
    global cnt
    global mode
    # 使用filedialog打开文件选择对话框
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        mode = 0
        # 使用Pillow读取图片
        img = Image.open(file_path)
        # 使用BICUBIC作为重采样过滤器
        img = img.resize((700, 700), Image.BICUBIC)
        # 将PIL Image对象转换为Tkinter可以显示的PhotoImage对象
        photo = ImageTk.PhotoImage(img)

        img = np.array(img)
        img2 = g_n(img)
        # img2 = get_roi(img2)
        # img2 = cv.resize(img2, (28,28))
        img2 = maxPool(img2)
        # img_in = img2
        img2 = enlarge(img2)
        image = Image.fromarray(img2)
        image = ImageTk.PhotoImage(image)

        img_in = cv.resize(img2, (28, 28))
        result_org = pt.predict(img_in, net)
        result = softmax(result_org)
        best_result = result.argmax(dim=1).item()
        best_result_num = max(max(result)).cpu().detach().numpy()
        if best_result_num <= 0.5:
            best_result = None
        if best_result is not None:
            ans.config(text="数字为" + str(best_result) + str(best_result_num))
        else:
            ans.config(text="未识别到数字")

        # 将上传的图片显示到窗口中
        image1.image = photo
        image1['image'] = photo
        image2.image = image
        image2['image'] = image

        # if answer > -0.001:
        #     ans.config(text="数字为" + str(answer))
        # else:
        #     ans.config(text="未识别到数字")


def imshow():
    global video
    global root
    global image1
    global image2
    global ans
    global cnt
    global mode
    r, img = video.read()
    img = pre(img)

    if r and mode == 1:
        # 将adarray转化为image
        img1 = Image.fromarray(img)
        # 显示图片到label
        img1 = ImageTk.PhotoImage(img1)

        img2 = g_n(img)
        # img2 = get_roi(img)
        img2 = maxPool(img2)
        img2 = enlarge(img2)
        img2 = Image.fromarray(img2)
        img2 = ImageTk.PhotoImage(img2)
        image1.image = img1
        image1['image'] = img1
        image2.image = img2
        image2['image'] = img2
        img_in = g_n(img)
        img_in = maxPool(img_in)
        result_org = pt.predict(img_in, net)
        result = softmax(result_org)
        best_result = result.argmax(dim=1).item()
        best_result_num = max(max(result)).cpu().detach().numpy()
        if best_result_num > 0.9:
            ans.config(text="数字为" + str(best_result))
            cnt = 0
        else:
            cnt += 1
        if cnt > 60:
            ans.config(text="未识别到数字")

    # 创建一个定时器，每10ms进入一次函数
    root.after(50, imshow)

def start():
    global mode
    global first
    mode = 1
    if first == 1:
        imshow()
        first = 0
btn_open = tkinter.Button(root, text="上传图片", command=open_image)
btn_open.place(x=1200, y=800, width=100, height=50)
btn_video = tkinter.Button(root, text="使用摄像头捕捉", command=start)
btn_video.place(x=1350, y=800, width=100, height=50)

# while (True):
#     ret, frame = capture.read()
#     since = time()
#     if ret:
#         frame = cv.imread(img_path)
#
#         img_bw = g_n(frame)
#         img_bw_sg = get_roi(img_bw)
#         # 展示图片
#         cv.imshow("img", img_bw_sg)
#         c = cv.waitKey(1) & 0xff
#         if c == 27:
#             capture.release()
#             break
#         img_in = cv.resize(img_bw_sg, (28, 28))
#         result_org = pt.predict(img_in, net)
#         result = softmax(result_org)
#         best_result = result.argmax(dim=1).item()
#         best_result_num = max(max(result)).cpu().detach().numpy()
#         if best_result_num <= 0.5:
#             best_result = None
#
#         # 显示结果
#         img_show = cv.resize(frame, (600, 600))
#         end_predict = time()
#         fps = round(1/(end_predict-since))
#         font = cv.FONT_HERSHEY_SIMPLEX
#         cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
#         cv.putText(img_show, "Probability is:" + str(best_result_num), (1, 60), font, 1, (0, 255, 0), 2)
#         cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)
#         cv.imshow("result", img_show)
#         cv.waitKey(1)
#         print(result)
#         print("*" * 50)
#         print("The number is:", best_result)
#
#
#     else:
#         print("please check camera!")
#         break


root.mainloop()

# 释放video资源
video.release()
