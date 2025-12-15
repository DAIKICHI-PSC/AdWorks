import datetime
from itertools import count
import time
import numpy as np
import torch
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_models import *
from settings import set_value

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
####################################################
from torchvision.transforms import InterpolationMode
####################################################

import sys
from PySide6 import QtCore, QtGui, QtWidgets
from MAIN_TRAINER_GUI import Ui_MainWindow
import glob


log_theta = torch.nn.LogSigmoid()

class DataLoader(Dataset):
    def __init__(self, c):
        self.dataset_path = c.data_path
        self.cropsize = c.crp_size
        # load dataset
        self.x = self.load_dataset_folder()
        # set transforms
        if float(win.ui.lineEdit2.text()) == 0:
            self.transform_x = T.Compose([T.Resize(c.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),T.CenterCrop(c.crp_size),T.ToTensor()])
        else:
            self.transform_x = T.Compose([T.Resize(c.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),T.RandomRotation(float(win.ui.lineEdit2.text())),T.CenterCrop(c.crp_size),T.ToTensor()])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
    def __getitem__(self, idx):
        x = self.x[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        '''
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        '''
        #
        x = self.normalize(self.transform_x(x))
        #
        return x
    def __len__(self):
        return len(self.x)
    def load_dataset_folder(self):
        x = []
        # load images
        img_fpath_list = sorted([os.path.join(self.dataset_path, f)
                                    for f in os.listdir(self.dataset_path)
                                    if f.endswith('.png')])
        x.extend(img_fpath_list)
        return list(x)


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    #L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        app.processEvents()
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                image = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image = next(iterator)
            # encoder prediction
            image = image.to(c.device)  # single scale
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            #e_list = list()
            #c_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
                    app.processEvents()
            app.processEvents()
        app.processEvents()
        #
        mean_train_loss = train_loss / train_count
        WConsole('Epoch : {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c.pool_layers # number of pooled layers
    WConsole('Number of pool layers = ' + str(L) + "\r\n")
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()
    #print(encoder)
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
        app.processEvents()
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    train_dataset = DataLoader(c)
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    dirPath = win.ui.lineEdit1.text()
    dirName = dirPath.rsplit("/")
    fPath = ""
    total_elapsed_time = 0
    for epoch in range(c.meta_epochs):
        start = time.time()
        WConsole('Train meta epoch: {}'.format(epoch))
        train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, pool_layers, N)
        elapsed_time = int(((time.time() - start) / 60 * 10 + 0.5) / 10)
        total_elapsed_time += elapsed_time
        WConsole("Elapsed tine of this epoch : " + str(elapsed_time) + " MINUTS")
        WConsole("Total elapsed time : " + str(total_elapsed_time) + " MINUTS")
        fPath = ""
        for i in dirName:
            fPath += i + "_"
        ret = save_weights(encoder, decoders, fPath + run_date + "_" + str(total_elapsed_time))  #saves
        WConsole(ret + "\r\n")
        app.processEvents()
    WConsole("Learning prosess is finished")


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


def CHECK_DIGITS_2(string):
    string = string.replace(".", "")
    string = string.replace("+", "")
    string = string.replace("-", "")
    if string.isdigit() == True:
        return True
    else:
        return False


class MainWindow1(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow1, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.comboBox1.addItems(["128", "256", "512", "1024", "2048"]) #####コンボボックスにアイテムを追加
        self.ui.comboBox1.setCurrentIndex(2)
        self.ui.pushButton1.clicked.connect(self.pushButton1_clicked)
        self.ui.pushButton2.clicked.connect(self.pushButton2_clicked)
        self.ui.pushButton3.clicked.connect(self.pushButton3_clicked)

    def pushButton1_clicked(self):
        global c
        dirPath = self.ui.lineEdit1.text()
        if  dirPath == "":
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Picture directory is not selected.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(self.ui.lineEdit2.text()) is False:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to META EPOCHS.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(self.ui.lineEdit3.text()) is False:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to SUB EPOCHS.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS(self.ui.lineEdit4.text()) is False:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to LEARNING RATE.\r\nValue must be above 0.")
            ret = msgbox.exec()
        elif CHECK_DIGITS_2(self.ui.lineEdit5.text()) is False:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("Only digits are allowed to ROTATION ANGLE.")
            ret = msgbox.exec()
        elif len(glob.glob(dirPath + "/*.png")) == 0:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowTitle("cflow-ad GUI")
            msgbox.setText("No png file in the folder.")
            ret = msgbox.exec()
        else:
            self.ui.comboBox1.setEnabled(False)
            self.ui.lineEdit1.setEnabled(False)
            self.ui.lineEdit2.setEnabled(False)
            self.ui.lineEdit3.setEnabled(False)
            self.ui.lineEdit4.setEnabled(False)
            self.ui.lineEdit5.setEnabled(False)
            self.ui.pushButton1.setEnabled(False)
            self.ui.pushButton2.setEnabled(False)
            self.ui.pushButton3.setEnabled(False)
            c.data_path = dirPath
            c.input_size = int(self.ui.comboBox1.currentText())
            c.img_size = (c.input_size, c.input_size)  # HxW format
            c.crp_size = (c.input_size, c.input_size)  # HxW format
            c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            c.img_dims = [3] + list(c.img_size)
            c.meta_epochs = int(self.ui.lineEdit2.text())
            c.sub_epochs = int(self.ui.lineEdit3.text())
            c.lr = float(self.ui.lineEdit4.text())
            WConsole(c.msg)
            train(c)
            self.ui.comboBox1.setEnabled(True)
            self.ui.lineEdit1.setEnabled(True)
            self.ui.lineEdit2.setEnabled(True)
            self.ui.lineEdit3.setEnabled(True)
            self.ui.lineEdit4.setEnabled(True)
            self.ui.lineEdit5.setEnabled(True)
            self.ui.pushButton1.setEnabled(True)
            self.ui.pushButton2.setEnabled(True)
            self.ui.pushButton3.setEnabled(True)

    def pushButton2_clicked(self):
        dirPath = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dirPath:
            self.ui.lineEdit1.setText(dirPath)


    def pushButton3_clicked(self):
        msgbox = QtWidgets.QMessageBox(self)
        ret = msgbox.question(None, "cflow-ad GUI", "If you would like to renumber file names to serial number, press yes.\nDo not renumber the files in the folder you are currently editing or the folder you have edited.\nOtherwise the data is going to be corrupted!!!\n\nThis function renumbers any files in the folder.\nPlease only place files you want to renumber in the folder.", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No) #選択用メッセージボックスを表示
        if ret == QtWidgets.QMessageBox.Yes:
            tmpPath = QtWidgets.QFileDialog.getExistingDirectory(self)
            if tmpPath:
                cn, buttonState = QtWidgets.QInputDialog.getInt(self, "cflow-ad GUI", "Please input starting number.", 0, 0, 9999999, 1)
                if buttonState:
                    FileList = glob.glob(tmpPath + '/*.png')
                    FileList2 = []
                    lCount = len(FileList)
                    progC = 0
                    progP = 0
                    prog = QtWidgets.QProgressDialog('Renaming files.', None, 0, 100, None, QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)
                    prog.setWindowTitle("cflow-ad GUI")
                    prog.setFixedSize(prog.sizeHint())
                    prog.setValue(progP)
                    prog.show()
                    for FN in FileList:
                        FileList2.append(FN.replace('\\', '/'))
                    iNum = cn
                    rF = []
                    for FN in FileList2:
                        npFile = tmpPath + '/abcdefghijklmnopqrstuvqxyz' + str(iNum) +'.png'
                        os.rename(FN, npFile)
                        iNum += 1
                        progC += 1
                        progP = int(100 * progC / (lCount * 2))
                        prog.setValue(progP)
                        app.processEvents()
                        rF.append(npFile)
                    iNum = cn
                    for FN in rF:
                        npFile = tmpPath + '/' + str(iNum) +'.png'
                        os.rename(FN, npFile)
                        iNum += 1
                        progC += 1
                        progP = int(100 * progC / (lCount * 2))
                        prog.setValue(progP)
                        app.processEvents()
                    msgbox = QtWidgets.QMessageBox(self)
                    msgbox.setWindowTitle("cflow-ad GUI")
                    msgbox.setText('Renumbering done.')
                    ret = msgbox.exec_()
                #####

if __name__ == '__main__':
    c = set_value()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow1()
    win.show()
    sys.exit(app.exec())
