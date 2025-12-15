import time
import numpy as np
import torch
import torch.nn.functional as F
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_models import *

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
####################################################
from torchvision.transforms import InterpolationMode
####################################################

import os
import datetime
from skimage import morphology
import cv2

from settings import set_value
from PySide6 import QtCore, QtGui, QtWidgets
from MAIN_DETECTOR_MOV_GUI import Ui_MainWindow
from getRectanglePos import getRectanglePos #２点の何れかが選択領域の開始点（左上）になり、終点（左下）になるか判定し、さらに終点が指定した範囲にあるかるか確認するライブラリ
import sys
import glob

import statistics





LOG_THETA = torch.nn.LogSigmoid()

PICTURE_PATH = ""
RESULT_PATH = "./result"
WORK_PATH = ""
OK_PATH = ""
NG_PATH = ""

ENCODER = ""
POOL_LAYERS = ""
DECODERS = ""

THRESHOLD_KERNEL = 0
THRESHOLD_DETECTION = 0
THRESHOLD_AREA = 0

DETECTION_STARTED = 0

NON_SIMILER_THRESHOLD = 0.5
NON_SIMILER_VALUE = 10.0

DETECT_COUNTER = 0
NG_COUNTER = 0
DETECTION_LOG =""

RUN_MODE = 0
PIC_DIR = ""

TAKE_PIC_MODE = 0

#####グローバル変数########################################
CAP = 0 #キャプチャー画像取得用変数
FOURCC = cv2.VideoWriter_fourcc(*'MJPG') #ビデオ保存用コーデック

sStartFlag = 0 #領域選択開始フラグ
mX1 = 0 #マウスボタンを押した時の横方向の座標
mY1 = 0 #マウスボタンを押した時の縦方向の座標
mX2 = 0 #マウスボタンを離した時の横方向の座標
mY2 = 0 #マウスボタンを離した時の縦方向の座標
ssX = 0 #選択領域開始点（左上）の横方向座標
ssY = 0 #選択領域開始点（左上）の縦方向座標
seX = 0 #選択領域終点（右下）の横方向座標（デフォルトではフレームワークで未使用）
seY = 0 #選択領域終点（右下）の縦方向座標（デフォルトではフレームワークで未使用）
sXL = 0 #選択領域の横方向座標の長さ（開始点＋長さで終点を求める場合は１を引く）
sYL = 0 #選択領域の縦方向座標の長さ（開始点＋長さで終点を求める場合は１を引く）

######フレームワーク以外のグローバル変数変数########################################
#TW = 512 #トリムモード用Ｗｉｄｔｈ
#TH = 512 #トリムモード用Ｈｅｉｇｈｔ
DTW = 512 #ＩＭＡＧＥ　ＳＩＺＥ用Ｗｉｄｔｈ
DTH = 512 #ＩＭＡＧＥ　ＳＩＺＥ用Ｈｅｉｇｈｔ
CAP_WIDTH = 320 #キャプチャー用Ｗｉｄｔｈ
CAP_HIGHT = 240 #キャプチャー用Ｈｅｉｇｈｔ

DETETED_PICTURE = ""






def DETECTION_START():
    global c
    global ENCODER
    global POOL_LAYERS
    global DECODERS
    global WORK_PATH
    global OK_PATH
    global NG_PATH
    global CAP
    global CAP_WIDTH
    global CAP_HIGHT
    #global TW
    #global TH
    global DETECTION_STARTED
    global DETETED_PICTURE
    global DTW
    global DTH
    #res = win.ui.comboBox1.currentText() #get capture
    #TW, TH = res.split('x') #input capture size
    #TW = int(TW)
    #TH = int(TH)
    if RUN_MODE == 0:
        L = c.pool_layers # number of pooled layers(default=3)
        #WConsole('Number of pool layers = ' + str(L))
        ENCODER, POOL_LAYERS, pool_dims = load_encoder_arch(c, L)
        ENCODER = ENCODER.to(c.device).eval()
        # NF decoder
        DECODERS = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
        DECODERS = [decoder.to(c.device) for decoder in DECODERS]
        load_weights(ENCODER, DECODERS, c.checkpoint)
        if DETECTION_STARTED == 1:
            WConsole("[DETECTION STARTED]")
            WConsole("[" + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S]\r\n"))
            if win.ui.checkBox3.isChecked() == True:
                WORK_PATH = RESULT_PATH + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if not os.path.isdir(WORK_PATH):
                    os.makedirs(WORK_PATH, exist_ok=True)
                OK_PATH = WORK_PATH + "/" + "OK"
                if not os.path.isdir(OK_PATH):
                    os.makedirs(OK_PATH, exist_ok=True)
                NG_PATH = WORK_PATH + "/" + "NG"
                if not os.path.isdir(NG_PATH):
                    os.makedirs(NG_PATH, exist_ok=True)
        res = win.ui.comboBox4.currentText() #get capture
        DTW, DTH = res.split('x') #input capture size
        DTW = int(DTW)
        DTH = int(DTH)
        c.input_size = DTW
        c.img_size = (c.input_size, c.input_size)  # HxW format
        c.crp_size = (c.input_size, c.input_size)  # HxW format
        c.img_dims = [3] + list(c.img_size)
        DETETED_PICTURE = np.zeros((DTH, DTW), np.uint8)
        INIT_DETECTION()
    res = win.ui.comboBox2.currentText() #get capture
    rx, ry = res.split('x') #input capture size
    camNum = int(win.ui.comboBox3.currentText()) #select cam
    CAP = cv2.VideoCapture(camNum, cv2.CAP_DSHOW) #create caputtr objrct
    CAP.set(cv2.CAP_PROP_BUFFERSIZE, 2) #set buffer size
    CAP.set(3, int(rx)) #set capure width
    CAP.set(4, int(ry)) #set caputure height
    CAP.set(6, cv2.VideoWriter_fourcc(*'MJPG'))
    CAP.set(5, 10)


    CAP.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) #カメラによる露出制御のON／OFF 0 = Manual Mode, 1= Auto Mode
    print(CAP.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    CAP.set(cv2.CAP_PROP_EXPOSURE, -2) #露出
    print(CAP.get(cv2.CAP_PROP_EXPOSURE))
    #CAP.set(cv2.CAP_PROP_GAIN, 144)  # ゲイン値を0に設定
    #print(CAP.get(cv2.CAP_PROP_GAIN))


    CAP_WIDTH = int(rx)
    CAP_HIGHT = int(ry)
    if DETECTION_STARTED == 0:
        DETECTION_STARTED = 1 #ループ中とする
    MAIN_LOOP()





def INIT_DETECTION(): #初回の検出が遅いので、事前に検出を一回実行
    global c
    # task data

    picture = np.zeros((DTW, DTH, 3), np.uint8) #####picture = np.zeros((512, 512, 3), np.uint8)

    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB) #convert from cv2 to pil
    picture = Image.fromarray(picture) #convert from cv2 to pil
    test_dataset  = DATASET_WITH_A_PICTURE(c, picture)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    _, _, _ = test_meta_epoch(c, test_dataset, ENCODER, DECODERS, POOL_LAYERS, N)





def DETECTION(PICTURE_PATH):
    global c
    global DETECT_COUNTER
    start = time.time()
    # task data
    original_pic = PICTURE_PATH
    picture = original_pic
    #picture = cv2.medianBlur(picture, 3) ##############################NOIZE REDUCTION
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB) #convert from cv2 to pil
    picture = Image.fromarray(picture) #convert from cv2 to pil
    test_dataset  = DATASET_WITH_A_PICTURE(c, picture)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    height, width, test_dist = test_meta_epoch(c, test_dataset, ENCODER, DECODERS, POOL_LAYERS, N)
    # PxEHW
    #print('Heights/Widths', height, width)
    test_map = [list() for p in POOL_LAYERS]
    for l, p in enumerate(POOL_LAYERS):
        test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
        test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
        test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
        test_mask = test_prob.reshape(-1, height[l], width[l])
        test_mask = test_prob.reshape(-1, height[l], width[l])
        # upsample
        test_map[l] = F.interpolate(test_mask.unsqueeze(1),size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
    # score aggregation
    score_map = np.zeros_like(test_map[0]) #####get the structure as first one out of 3
    average_map = 0
    for l, p in enumerate(POOL_LAYERS):
        if win.ui.checkBox2.isChecked() == True:
            WConsole("AVERAGE VALUE OF THE LAYER[" + str(l) + "] IS " + str(np.mean(test_map[l])))
        average_map += np.mean(test_map[l]) #####detect picture that is totally different from selected category(each value of test_map is very small)
        score_map += test_map[l] #####add 3 layers to one.
    if win.ui.checkBox2.isChecked() == True:
        WConsole("AVERAGE VALUE OF THE RESULT(SUM OF THE LAYERS) IS " + str(average_map))
    score_mask = score_map
    if average_map < NON_SIMILER_THRESHOLD : #####If detected picture is totally different from selected category, set score as defect
        super_mask = score_mask * 0 + NON_SIMILER_VALUE
        if win.ui.checkBox2.isChecked() == True:
            WConsole("*****AVERAGE VALUE OF THE RESULT IS SMALLER THAN NON SIMILER THRESHOLD*****")
            WConsole("*****EACH VALUE OF THE RESULT ARE CHANGED TO " + str(NON_SIMILER_VALUE) + "*****")
    else:
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
    # show elapsed time
    if win.ui.checkBox2.isChecked() == True:
        average_map = np.mean(super_mask)
        WConsole("AVERAGE VALUE OF THE INVERTED RESULT IS " + str(average_map))
        WConsole("MAX VALUE OF THE INVERTED RESULT IS " + str(super_mask.max()))
        WConsole("MIN VALUE OF THE INVERTED RESULT IS " + str(super_mask.min()))
        elapsed_time = int((time.time() - start) * 1000 + 0.5) / 1000
        WConsole("ELAPSED TIME : " + str(elapsed_time) + " SECONDS")
    DETECT_COUNTER += 1
    win.ui.lineEdit6.setText(str(DETECT_COUNTER))
    # export visualuzations
    ret = export_test_images(c, super_mask, original_pic)
    return ret





class DATASET_WITH_A_PICTURE(Dataset):
    def __init__(self, c, picture):
        self.picture = picture
        self.cropsize = c.crp_size
        # load dataset
        y =[[]]
        self.x = list(y)
        # set transforms
        #self.transform_x = T.Compose([T.Resize(c.img_size, Image.ANTIALIAS),T.CenterCrop(c.crp_size),T.ToTensor()]) ####################gets warning####################
        self.transform_x = T.Compose([T.Resize(c.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),T.CenterCrop(c.crp_size),T.ToTensor()])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
    def __getitem__(self, idx):
        x = self.x[idx]
        x = self.picture
        x = self.normalize(self.transform_x(x))
        x =  x.unsqueeze(0) ####################convert tensor dimention####################
        return x
    def __len__(self):
        return len(self.x)





def test_meta_epoch(c, loader, encoder, decoders, pool_layers, N):
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        image = loader[0]
        # data
        image = image.to(c.device) # single scale
        _ = encoder(image)  # BxCxHxW
        # test decoder
        for l, layer in enumerate(pool_layers):
            e = activation[layer]  # BxCxHxW
            #
            B, C, H, W = e.size()
            S = H*W
            E = B*S
            #
            height.append(H)
            width.append(W)
            #
            p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
            c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
            e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
            decoder = decoders[l]
            FIB = E//N + int(E%N > 0)  # number of fiber batches
            for f in range(FIB):
                if f < (FIB-1):
                    idx = torch.arange(f*N, (f+1)*N)
                else:
                    idx = torch.arange(f*N, E)
                #
                c_p = c_r[idx]  # NxP
                e_p = e_r[idx]  # NxC
                #
                z, log_jac_det = decoder(e_p, [c_p,])
                #
                decoder_log_prob = get_logp(C, z, log_jac_det)
                log_prob = decoder_log_prob / C  # likelihood per dim
                loss = -LOG_THETA(log_prob)
                test_loss += t2np(loss.sum())
                test_count += len(loss)
                test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    return height, width, test_dist





def export_test_images(c, scores, pic):
    global NG_COUNTER
    global DETECTION_LOG
    kernel = morphology.disk(THRESHOLD_KERNEL)
    score_seg = np.zeros_like(scores)
    score_seg[scores >=  THRESHOLD_DETECTION] = 1.0 ####################THRESHOLD####################
    score_seg = morphology.opening(score_seg, kernel)
    score_seg = (255.0 * score_seg).astype(np.uint8)

    vis = cv2.cvtColor(score_seg, cv2.COLOR_GRAY2BGR)
    if win.ui.lineEdit8.text() != '':
        vis = cv2.resize(vis,(int(win.ui.lineEdit10.text()), int(win.ui.lineEdit11.text())))
    else:
        th, tw, _ = pic.shape[:3]
        vis = cv2.resize(vis,(tw, th))

    #####CONTOURS TO DETECT EDGE
    reg = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(reg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if win.ui.checkBox1.isChecked() == True:
        contours = list(filter(lambda x: cv2.contourArea(x) >= THRESHOLD_AREA, contours)) #leave area bigger than threshold
    WConsole("[" + str(DETECT_COUNTER) +"]NUMBER OF DEFECTS : " + str(len(contours)))
    if win.ui.checkBox3.isChecked() == True:
        DETECTION_LOG += str(DETECT_COUNTER) +"," + str(len(contours)) + "\n"
    if len(contours) > 0:
        NG_COUNTER += 1
        win.ui.lineEdit7.setText(str(NG_COUNTER))
    external_contours = np.zeros(reg.shape)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(external_contours, contours, i - 1, 255, 2) #last value is thickness of line
    external_contours = (external_contours).astype(np.uint8)
    external_contours = cv2.cvtColor(external_contours, cv2.COLOR_GRAY2BGR)
    #BLUE
    #external_contours[:, :, 1] = 0  # G
    #external_contours[:, :, 2] = 0  # R
    #GREEN
    #external_contours[:, :, 0] = 0  # B
    #external_contours[:, :, 2] = 0  # R
    #RED
    external_contours[:, :, 0] = 0  # B
    external_contours[:, :, 1] = 0  # G

    dst = cv2.addWeighted(pic, 1.0, vis, 0.3, 0)
    dst = cv2.addWeighted(dst, 1.0, external_contours, 1.0, 0)
    if win.ui.checkBox3.isChecked() == True:
        if len(contours) > 0:
            cv2.imwrite(NG_PATH + "/" + str(DETECT_COUNTER) + ".png", dst)
        else:
            cv2.imwrite(OK_PATH + "/" + str(DETECT_COUNTER) + ".png", dst)
    #cv2.imshow("DETECTION", dst)
    #cv2.imshow("PIC", pic)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    #if win.ui.lineEdit8.text() != '':
        #dst = cv2.resize(dst,(int(win.ui.lineEdit10.text()), int(win.ui.lineEdit11.text())))
    return dst





def MAIN_LOOP():
    global DETETED_PICTURE
    while(True):
        ret, frame = CAP.read()
        frame = cv2.edgePreservingFilter(frame, cv2.NORMCONV_FILTER)
        if ret == True and DETECTION_STARTED == 1:
            frameB = np.copy(frame)
            if DETECTION_STARTED == 1:
                if win.ui.lineEdit8.text() != '':
                    frameB = frameB[int(win.ui.lineEdit9.text()):int(win.ui.lineEdit9.text()) + int(win.ui.lineEdit11.text()), int(win.ui.lineEdit8.text()):int(win.ui.lineEdit8.text()) + int(win.ui.lineEdit10.text())] #指定したサイズに画像をトリム
                elif sStartFlag == 1: #領域選択開始後の処理
                    frameB = cv2.rectangle(frameB, (ssX, ssY), (sXL, sYL), (0, 0, 255), 1)
                cvKey = cv2.waitKey(1)
                if cvKey == 32: ##########SPACE KEY##########
                    if win.ui.lineEdit8.text() != '' and RUN_MODE == 0:
                        DETETED_PICTURE = DETECTION(frameB)
                    elif win.ui.lineEdit8.text() != '' and RUN_MODE == 1:
                        #frameC = cv2.medianBlur(frameB, 3) ##############################NOIZE REDUCTION
                        cv2.imwrite(PIC_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", frameB)
                        WConsole("SAVED " + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
                    else:
                        msgbox = QtWidgets.QMessageBox()
                        msgbox.setWindowTitle("cflow-ad GUI")
                        msgbox.setText("Please select detection position.")
                        ret = msgbox.exec()
            if DETECTION_STARTED == 1:
                cv2.imshow("CURRENT", frameB)
                cv2.setMouseCallback("CURRENT", onMouse)
                if RUN_MODE == 0:
                    cv2.imshow("DETECTED", DETETED_PICTURE)
            app.processEvents()
        else:
            break





def WConsole(cmd):
    cursor = win.ui.plainTextEdit1.textCursor()
    cursor.movePosition(QtGui.QTextCursor.End)
    cursor.insertText(cmd + "\r\n")
    win.ui.plainTextEdit1.setTextCursor(cursor)





def CHECK_DIGITS(string):
    if float(string) <= 0:
        return False
    string = string.replace(".", "")
    string = string.replace("+", "")
    string = string.replace("-", "")
    if string.isdigit() == True:
        return True
    else:
        return False





def CHECK_MAX():
    global c
    # task data
    original_pic = cv2.imread(PICTURE_PATH)
    picture = original_pic
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB) #convert from cv2 to pil
    picture = Image.fromarray(picture) #convert from cv2 to pil
    test_dataset  = DATASET_WITH_A_PICTURE(c, picture)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    height, width, test_dist = test_meta_epoch(c, test_dataset, ENCODER, DECODERS, POOL_LAYERS, N)
    # PxEHW
    #print('Heights/Widths', height, width)
    test_map = [list() for p in POOL_LAYERS]
    for l, p in enumerate(POOL_LAYERS):
        test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
        test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
        test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
        test_mask = test_prob.reshape(-1, height[l], width[l])
        test_mask = test_prob.reshape(-1, height[l], width[l])
        # upsample
        test_map[l] = F.interpolate(test_mask.unsqueeze(1),size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
    # score aggregation
    score_map = np.zeros_like(test_map[0]) #####get the structure as first one out of 3
    average_map = 0
    for l, p in enumerate(POOL_LAYERS):
        average_map += np.mean(test_map[l]) #####detect picture that is totally different from selected category(each value of test_map is very small)
        score_map += test_map[l] #####add 3 layers to one.
    score_mask = score_map
    if average_map < NON_SIMILER_THRESHOLD : #####If detected picture is totally different from selected category, set score as defect
        super_mask = score_mask * 0 + NON_SIMILER_VALUE
    else:
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
    WConsole("MAX VALUE OF THE INVERTED RESULT IS " + str(super_mask.max()))
    app.processEvents()
    return super_mask.max()





class MainWindow1(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow1, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.ui.comboBox1.addItems(["256x256","512x512", "1024x1024"])
        #self.ui.comboBox1.setCurrentIndex(0)
        self.ui.comboBox2.addItems(["640x480", "800x600", "1024x768", "1280x960", "1400x1050", "2448x2048", "2592x1944", "640x360", "1280x720", "1600x900", "1920x1080"]) #コンボボックスにアイテムを追加
        self.ui.comboBox2.setCurrentIndex(0)
        self.ui.comboBox3.addItems(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.ui.comboBox3.setCurrentIndex(0)
        self.ui.comboBox4.addItems(["256x256","512x512", "1024x1024", "2048x2048"])
        self.ui.comboBox4.setCurrentIndex(0)
        self.ui.pushButton1.clicked.connect(self.pushButton1_clicked)
        self.ui.pushButton2.clicked.connect(self.pushButton2_clicked)
        self.ui.pushButton3.clicked.connect(self.pushButton3_clicked)
        self.ui.pushButton4.clicked.connect(self.pushButton4_clicked)
        self.ui.pushButton5.clicked.connect(self.pushButton5_clicked)
        self.ui.pushButton6.clicked.connect(self.pushButton6_clicked)
        self.ui.pushButton7.clicked.connect(self.pushButton7_clicked)
        self.ui.pushButton8.clicked.connect(self.pushButton8_clicked)
        self.ui.pushButton9.clicked.connect(self.pushButton9_clicked)
        self.ui.pushButton10.clicked.connect(self.pushButton10_clicked)
        if os.path.isfile("SETTINGS.cfg"):
            f = open("SETTINGS.cfg", "r")
            text = f.readlines()
            f.close()
            self.ui.lineEdit1.setText(text[0].replace("\n", ""))
            self.ui.lineEdit2.setText(text[1].replace("\n", ""))
            self.ui.lineEdit3.setText(text[2].replace("\n", ""))
            self.ui.lineEdit4.setText(text[3].replace("\n", ""))
            self.ui.lineEdit5.setText(text[4].replace("\n", ""))
            self.ui.lineEdit8.setText(text[5].replace("\n", ""))
            self.ui.lineEdit9.setText(text[6].replace("\n", ""))
            if int(text[7]) == 0:
                self.ui.checkBox1.setChecked(False)
            else:
                self.ui.checkBox1.setChecked(True)
            if int(text[8]) == 0:
                self.ui.checkBox2.setChecked(False)
            else:
                self.ui.checkBox2.setChecked(True)
            if int(text[9]) == 0:
                self.ui.checkBox3.setChecked(False)
            else:
                self.ui.checkBox3.setChecked(True)
            #self.ui.comboBox1.setCurrentIndex(int(text[10]))
            self.ui.comboBox2.setCurrentIndex(int(text[10]))
            self.ui.comboBox3.setCurrentIndex(int(text[11]))
            self.ui.comboBox4.setCurrentIndex(int(text[12]))
            self.ui.lineEdit10.setText(text[13].replace("\n", ""))
            self.ui.lineEdit11.setText(text[14].replace("\n", ""))
        if not os.path.isdir(RESULT_PATH):
            os.makedirs(RESULT_PATH, exist_ok=True)

    def pushButton1_clicked(self):
        global c
        global THRESHOLD_KERNEL
        global THRESHOLD_DETECTION
        global THRESHOLD_AREA
        global DETECTION_STARTED
        global NON_SIMILER_THRESHOLD
        global DETECT_COUNTER
        global NG_COUNTER
        global RUN_MODE
        THRESHOLD_KERNEL = self.ui.lineEdit2.text()
        THRESHOLD_DETECTION = self.ui.lineEdit3.text()
        THRESHOLD_AREA = self.ui.lineEdit4.text()
        NON_SIMILER_THRESHOLD = self.ui.lineEdit5.text()
        filepath = self.ui.lineEdit1.text()
        if  filepath == "" and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("PT file is not selected.")
            ret = msgbox.exec()
        elif not os.path.exists(filepath) and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("The selected PT file does not exist.")
            ret = msgbox.exec()
        elif not os.path.exists(filepath) and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Selected PT file dose not exist.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(THRESHOLD_KERNEL) is False and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to KERNEL SIZE.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(THRESHOLD_DETECTION) is False and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to THRESHOLD FOR DETECTION.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(THRESHOLD_AREA) is False and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to THRESHOLD FOR AREA SIZE.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(NON_SIMILER_THRESHOLD) is False and RUN_MODE == 0:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to THRESHOLD FOR AREA SIZE.\r\nValue must be above 0.")
            ret = msgbox.exec()
        else:
            THRESHOLD_KERNEL = int(THRESHOLD_KERNEL)
            THRESHOLD_DETECTION = float(THRESHOLD_DETECTION)
            THRESHOLD_AREA = int(THRESHOLD_AREA)
            NON_SIMILER_THRESHOLD = float(NON_SIMILER_THRESHOLD)
            DETECTION_STARTED = 1
            DETECT_COUNTER = 0
            NG_COUNTER = 0
            self.ui.lineEdit6.setText(str(DETECT_COUNTER))
            self.ui.lineEdit7.setText(str(NG_COUNTER))
            self.ui.plainTextEdit1.setPlainText("")
            c.checkpoint = filepath
            
            self.ui.pushButton1.setEnabled(False)
            self.ui.pushButton2.setEnabled(True)
            self.ui.pushButton3.setEnabled(False)
            self.ui.pushButton4.setEnabled(False)
            self.ui.pushButton5.setEnabled(False)
            self.ui.pushButton6.setEnabled(False)
            self.ui.pushButton7.setEnabled(False)
            self.ui.pushButton8.setEnabled(False)
            self.ui.pushButton9.setEnabled(False)
            if TAKE_PIC_MODE == 0:
                self.ui.pushButton10.setEnabled(True)
            else:
                self.ui.pushButton10.setEnabled(False)
            self.ui.lineEdit2.setEnabled(False)
            self.ui.lineEdit3.setEnabled(False)
            self.ui.lineEdit4.setEnabled(False)
            self.ui.lineEdit5.setEnabled(False)
            self.ui.checkBox1.setEnabled(False)
            self.ui.checkBox3.setEnabled(False)
            self.ui.comboBox2.setEnabled(False)
            self.ui.comboBox3.setEnabled(False)
            self.ui.comboBox4.setEnabled(False)
            DETECTION_START()

    def pushButton2_clicked(self):
        global DETECTION_STARTED
        global CAP
        global RUN_MODE
        global TAKE_PIC_MODE
        DETECTION_STARTED = 0
        self.ui.pushButton1.setEnabled(True)
        self.ui.pushButton2.setEnabled(False)
        self.ui.pushButton3.setEnabled(True)
        self.ui.pushButton4.setEnabled(True)
        self.ui.pushButton5.setEnabled(True)
        self.ui.pushButton6.setEnabled(True)
        self.ui.pushButton7.setEnabled(True)
        self.ui.pushButton8.setEnabled(True)
        self.ui.pushButton9.setEnabled(True)
        self.ui.pushButton10.setEnabled(False)
        self.ui.lineEdit2.setEnabled(True)
        self.ui.lineEdit3.setEnabled(True)
        self.ui.lineEdit4.setEnabled(True)
        self.ui.lineEdit5.setEnabled(True)
        self.ui.checkBox1.setEnabled(True)
        self.ui.checkBox3.setEnabled(True)
        self.ui.comboBox2.setEnabled(True)
        self.ui.comboBox3.setEnabled(True)
        self.ui.comboBox4.setEnabled(True)
        if self.ui.checkBox3.isChecked() == True and RUN_MODE == 0:
            f = open(WORK_PATH + "/log.csv", "w")
            f.writelines(DETECTION_LOG)
            f.close()
            text = "COUNTER=" + self.ui.lineEdit6.text() + "\n" + "NG=" + self.ui.lineEdit7.text() + "\n"
            f = open(WORK_PATH + "/log.txt", "w")
            f.writelines(text)
            f.close()
        CAP.release() #release capture object
        cv2.destroyAllWindows() #close all cv windows
        RUN_MODE = 0
        TAKE_PIC_MODE = 0
    
    def pushButton3_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'pt File (*.pt)')
        if filepath:
            self.ui.lineEdit1.setText(filepath)

    def pushButton4_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'cfg File (*.cfg)')
        if filepath:
            f = open(filepath, "r")
            text = f.readlines()
            f.close()
            self.ui.lineEdit1.setText(text[0].replace("\n", ""))
            self.ui.lineEdit2.setText(text[1].replace("\n", ""))
            self.ui.lineEdit3.setText(text[2].replace("\n", ""))
            self.ui.lineEdit4.setText(text[3].replace("\n", ""))
            self.ui.lineEdit5.setText(text[4].replace("\n", ""))
            self.ui.lineEdit8.setText(text[5].replace("\n", ""))
            self.ui.lineEdit9.setText(text[6].replace("\n", ""))
            if int(text[7]) == 0:
                self.ui.checkBox1.setChecked(False)
            else:
                self.ui.checkBox1.setChecked(True)
            if int(text[8]) == 0:
                self.ui.checkBox2.setChecked(False)
            else:
                self.ui.checkBox2.setChecked(True)
            if int(text[9]) == 0:
                self.ui.checkBox3.setChecked(False)
            else:
                self.ui.checkBox3.setChecked(True)
            #self.ui.comboBox1.setCurrentIndex(int(text[10]))
            self.ui.comboBox2.setCurrentIndex(int(text[10]))
            self.ui.comboBox3.setCurrentIndex(int(text[11]))
            self.ui.comboBox4.setCurrentIndex(int(text[12]))
            self.ui.lineEdit10.setText(text[13].replace("\n", ""))
            self.ui.lineEdit11.setText(text[14].replace("\n", ""))
            f = open("SETTINGS.cfg", "w")
            f.writelines(text)
            f.close()

    def pushButton5_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "",'cfg File (*.cfg)')
        if filepath:
            text = self.ui.lineEdit1.text() + "\n" + \
            self.ui.lineEdit2.text() + "\n" + \
            self.ui.lineEdit3.text() + "\n" + \
            self.ui.lineEdit4.text() + "\n" + \
            self.ui.lineEdit5.text() + "\n" + \
            self.ui.lineEdit8.text() + "\n" + \
            self.ui.lineEdit9.text() + "\n"
            if self.ui.checkBox1.isChecked() == True:
                text = text + "1\n"
            else:
                text = text + "0\n"
            if self.ui.checkBox2.isChecked() == True:
                text = text + "1\n"
            else:
                text = text + "0\n"
            if self.ui.checkBox3.isChecked() == True:
                text = text + "1\n"
            else:
                text = text + "0\n"
            #text = text + str(self.ui.comboBox1.currentIndex()) + "\n"
            text = text + str(self.ui.comboBox2.currentIndex()) + "\n"
            text = text + str(self.ui.comboBox3.currentIndex()) + "\n"
            text = text + str(self.ui.comboBox4.currentIndex()) + "\n"
            text = text +self.ui.lineEdit10.text() + "\n"
            text = text +self.ui.lineEdit11.text() + "\n"
            f = open(filepath, "w")
            f.writelines(text)
            f.close()
            f = open("SETTINGS.cfg", "w")
            f.writelines(text)
            f.close()

    def pushButton6_clicked(self):
        self.ui.lineEdit1.setText("")
        self.ui.lineEdit2.setText("4")
        self.ui.lineEdit3.setText("2.0")
        self.ui.lineEdit4.setText("500")
        self.ui.lineEdit5.setText("0.5")
        self.ui.lineEdit8.setText("")
        self.ui.lineEdit9.setText("")
        self.ui.lineEdit10.setText("")
        self.ui.lineEdit11.setText("")
        self.ui.checkBox1.setChecked(True)
        self.ui.checkBox2.setChecked(False)
        self.ui.checkBox3.setChecked(True)
        #self.ui.comboBox1.setCurrentIndex(0)
        self.ui.comboBox2.setCurrentIndex(0)
        self.ui.comboBox3.setCurrentIndex(0)
        self.ui.comboBox4.setCurrentIndex(0)

    def pushButton7_clicked(self):
        global c
        global ENCODER
        global POOL_LAYERS
        global DECODERS
        global PICTURE_PATH
        filepath = self.ui.lineEdit1.text()
        if  filepath == "":
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("PT file is not selected.")
            ret = msgbox.exec()
        elif not os.path.exists(filepath):
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("The selected PT file does not exist.")
        else:
            res = self.ui.comboBox4.currentText() #get capture
            CTW, CTH = res.split('x') #input capture size
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Detection size is " + CTW + "x" + CTH)
            ret = msgbox.exec()
            CTW = int(CTW)
            CTH = int(CTH)
            c.input_size = CTW
            c.img_size = (c.input_size, c.input_size)  # HxW format
            c.crp_size = (c.input_size, c.input_size)  # HxW format
            c.img_dims = [3] + list(c.img_size)
            c.checkpoint = filepath
            L = c.pool_layers # number of pooled layers(default=3)
            #WConsole('Number of pool layers = ' + str(L))
            ENCODER, POOL_LAYERS, pool_dims = load_encoder_arch(c, L)
            ENCODER = ENCODER.to(c.device).eval()
            # NF decoder
            DECODERS = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
            DECODERS = [decoder.to(c.device) for decoder in DECODERS]
            load_weights(ENCODER, DECODERS, c.checkpoint)
            DirPath = QtWidgets.QFileDialog.getExistingDirectory(self)
            if DirPath:
                max_list = []
                FileList = glob.glob(DirPath + '/*.png')
                for FN in FileList: 
                    FN = FN.replace('\\', '/')
                    PICTURE_PATH = FN
                    ret = CHECK_MAX()
                    max_list.append(ret)
                max_max = max(max_list)
                max_min = min(max_list)
                WConsole("MAX VALUE OF MAX VALUE IS " + str(max_max))
                WConsole("MIN VALUE OF MAX VALUE IS " + str(max_min))
                WConsole("DIFFERENCE OF MAX AND MIN IS " + str(max_max - max_min))
                max_average = np.mean(max_list)
                WConsole("AVERAGE OF MAX VALUE IS " + str(max_average))
                max_deviation = statistics.pstdev(max_list)
                WConsole("STANDERD DEVIATION OF MAX VALUE IS " + str(max_deviation))
                max_median = statistics.median(max_list)
                WConsole("MEDIAN OF MAX VALUE IS " + str(max_median))
                tmp_threshold = max_median + max_deviation / 2
                tmp_threshold = int((tmp_threshold + 0.05) * 10) / 10
                WConsole("[THRESHOLD FOR DETECTION SHOULD BE AROUND " + str(tmp_threshold) + "]\r\n")

    def pushButton8_clicked(self):
        self.ui.lineEdit8.setText("")
        self.ui.lineEdit9.setText("")
        self.ui.lineEdit10.setText("")
        self.ui.lineEdit11.setText("")

    def pushButton9_clicked(self):
        global RUN_MODE
        global PIC_DIR
        global TAKE_PIC_MODE
        dirPath = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dirPath:
            RUN_MODE = 1
            TAKE_PIC_MODE = 1
            PIC_DIR = dirPath
            self.pushButton1_clicked()

    def pushButton10_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'png File (*.png)')
        if filepath:
            pic = cv2.imread(filepath)
            ret = DETECTION(pic)
            cv2.imshow('TEST ORIGINAL', pic)
            cv2.imshow('TEST DETECTED', ret)





    def closeEvent(self, event): #event.accept() event.ignore()
        if DETECTION_STARTED == 1:
            event.ignore()
        else:
            event.accept()





#=====メインウィンドウで取得したウィジットのイベント処理========================================
def onMouse(event, x, y, flags, param):  
        #global camWidth
        #global camHeight
        global sStartFlag
        global mX1
        global mY1
        global mX2
        global mY2
        global ssX
        global ssY
        global sXL
        global sYL
        #マウスが移動た時の処理
        #マウスボタンがクリックされた時の処理
        if event == cv2.EVENT_LBUTTONDOWN and win.ui.lineEdit8.text() == '':
            if sStartFlag == 0 and DETECTION_STARTED == 1:
                sFlag = 0
                sStartFlag = 1
                #マウス位置の取得
                mX1 = x
                mY1 = y
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, _ , _ = getRectanglePos(mX1, mY1, mX2, mY2, CAP_WIDTH, CAP_HIGHT)
        #マウスボタンがリリースされた時の処理
        elif event == cv2.EVENT_LBUTTONUP and win.ui.lineEdit8.text() == '':
            if sStartFlag == 1:
                sStartFlag = 0
                #マウス位置の取得
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, W , H = getRectanglePos(mX1, mY1, mX2, mY2, CAP_WIDTH, CAP_HIGHT)
                if ret == 1:
                    sStartFlag = 0
                    win.ui.lineEdit8.setText(str(ssX))
                    win.ui.lineEdit9.setText(str(ssY))
                    win.ui.lineEdit10.setText(str(W))
                    win.ui.lineEdit11.setText(str(H))

        #マウスボタンが移動した時の処理
        elif event == cv2.EVENT_MOUSEMOVE:
            if sStartFlag == 1:
                #マウス位置の取得
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, _ , _ = getRectanglePos(mX1, mY1, mX2, mY2, CAP_WIDTH, CAP_HIGHT)
            else:
                mX1 = x
                mY1 = y

if __name__ == '__main__':
    c = set_value()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow1()
    win.show()
    sys.exit(app.exec())
