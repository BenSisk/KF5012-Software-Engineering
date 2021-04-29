from user import Ui_MainWindow

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QCoreApplication, QObject
import numpy as np
import time
import sys
import cv2
import os


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    videoPath = ""
    running = False
    frameCount = 0
    currentFrame = 0

    def run(self):
        count = 0
        cap = cv2.VideoCapture(self.videoPath)
        video = cap

        fps = video.get(cv2.CAP_PROP_FPS)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            if (self.running):
                time.sleep(1 / fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)
                ret, frame = cap.read()
                self.currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if ret:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 640, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
            if (self.currentFrame >= self.frameCount - 1):
                break
        self.running = False

    def stop(self):
        self.terminate()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    th = Thread()
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.videoScrub.setValue(int(self.th.currentFrame))
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.label.frameGeometry().width(),self.label.frameGeometry().height(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle("Team Mace Windu User Interface")
        self.resize(1000, 700)

        self.openFile.clicked.connect(self.openFileDef)
        self.runTest.clicked.connect(self.runTestDef)
        self.playPause.clicked.connect(self.playPauseDef)
        self.videoScrub.valueChanged.connect(self.videoScrubChanged)

        self.weightButton.clicked.connect(self.weightButtonDef)
        

        self.weightText.setText("yolov5/trained_models/models/pre_more.pt")
        self.confText.setText("0.75")
        self.imageSizeText.setText("640")

        th = Thread(self)

    def openFileDef(self):
        self.th.stop()
        filename = QFileDialog.getOpenFileName(self, 'Open File', '',"Image or Video (*.jpg *.png *.mp4)")[0]
        self.fileOpen(filename)
        
            
        
    def fileOpen(self, filename):
        if (filename != ""):
            self.sourceText.setText(filename)
            splitFilename = filename.split(".")
            type = splitFilename[len(splitFilename) - 1]

            if (type == "mp4"):
                #video
                self.th.changePixmap.connect(self.setImage)
                self.th.videoPath = filename
                self.th.start()
                self.th.running = True
                self.videoScrub.setValue(0)
                self.videoScrub.setEnabled(True)
                self.playPause.setEnabled(True)
                cap = cv2.VideoCapture(self.th.videoPath)
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videoScrub.setRange(0, frameCount)
                self.playPause.setText("Pause")
            elif (type == "jpg" or type == "png"):
                #image
                image_path = filename
                self.show_frame_in_display(image_path)
                self.videoScrub.setEnabled(False)
                self.playPause.setEnabled(False)
                self.videoScrub.setValue(0)

            else:
                QMessageBox.about(self, "Error", "Invalid File Type")
        
    def show_frame_in_display(self,image_path):
        changePixmap = pyqtSignal(QImage)
        frame = self.label
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(frame.frameGeometry().width(),frame.frameGeometry().height(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        frame.setPixmap(pixmap)

    def isFloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def runTestDef(self):
        valid = True
        weightPath = self.weightText.text()
        conf = self.confText.text()
        imgSize = self.imageSizeText.text()
        sourcePath = self.sourceText.text()
        invalids = ""
        if (not os.path.isfile(sourcePath)):
            invalids += "Source File\n"
            valid = False
        if (not os.path.isfile(weightPath)):
            invalids += "Weight File\n"
            valid = False
        if (not (self.isFloat(conf) and float(conf) > 0)):
            invalids += "Confidence Threshold\n"
            valid = False
        if(not str.isdigit(imgSize)):
            invalids += "Image Size\n"
            valid = False
        if (valid):
            source = sourcePath.split("/")
            destinationFile = "runs/detect/" + (source[len(source) - 1])
            command = 'python yolov5/detect.py --source "' + sourcePath + '" --weights "' + weightPath + '" --conf ' + conf + " --img-size " + imgSize + " --project runs --name detect --exist-ok"
            os.system(command)
            self.fileOpen(destinationFile)
        else:
            message = "Invalid Parameters:\n\n" + invalids
            QMessageBox.about(self, "Error", message)
        

    def weightButtonDef(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Weights File', '',"Weights File (*.pt)")
        if (filename[0] != ""):
            self.weightText.setText(filename[0])


    def playPauseDef(self):
        self.th.running = not self.th.running
        if (self.th.running):
            self.playPause.setText("Pause")
        else:
            self.playPause.setText("Play")

    def videoScrubChanged(self, value):
        self.th.currentFrame = value

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.show()
    app.exec()