# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import time
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys
from inference import predict_with_model


root = dict()


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, options):
        super().__init__()
        self.options = options

    def run(self):
        global root
        # Here we pass the update_progress (uncalled!)
        self.options['update_percent_func'] = self.update_progress
        predict_with_model(self.options)
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        self.finished.emit()

    def update_progress(self, percent):
        self.progress.emit(percent)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(560, 360)
        self.move(300, 300)
        self.setWindowTitle('MVSEP music separation model')
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        global root
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)

    def execute_long_task(self):
        global root

        if len(root['input_files']) == 0 and 0:
            QMessageBox.about(root['w'], "Error", "No input files specified!")
            return

        root['progress_bar'].show()
        root['button_start'].setDisabled(True)
        root['button_finish'].setDisabled(False)
        root['start_proc'] = True

        options = {
            'input_audio': root['input_files'],
            'output_folder': root['output_folder'],
            'cpu': False,
            'overlap_large': 0.7,
            'overlap_small': 0.8,
        }

        self.update_progress(0)
        self.thread = QThread()
        self.worker = Worker(options)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)

        self.thread.start()

    def stop_separation(self):
        global root
        self.thread.terminate()
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        root['progress_bar'].hide()

    def update_progress(self, progress):
        global root
        root['progress_bar'].setValue(progress)


def dialog_select_input_files():
    global root
    files, _ = QFileDialog.getOpenFileNames(
        None,
        "QFileDialog.getOpenFileNames()",
        "",
        "All Files (*);;Audio Files (*.wav, *.mp3, *.flac)",
    )
    if files:
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)
    return files


def dialog_select_output_folder():
    global root
    foldername = QFileDialog.getExistingDirectory(
        None,
        "Select Directory"
    )
    root['output_folder'] = foldername + '/'
    root['output_folder_line_edit'].setText(root['output_folder'])
    return foldername


def create_dialog():
    global root
    app = QApplication(sys.argv)

    w = MyWidget()

    root['input_files'] = []
    root['output_folder'] = os.path.dirname(os.path.abspath(__file__)) + '/results/'

    button_select_input_files = QPushButton(w)
    button_select_input_files.setText("Input audio files")
    button_select_input_files.clicked.connect(dialog_select_input_files)
    button_select_input_files.setFixedHeight(35)
    button_select_input_files.setFixedWidth(150)
    button_select_input_files.move(30, 20)

    input_files_list_text_area = QTextEdit(w)
    input_files_list_text_area.setReadOnly(True)
    input_files_list_text_area.setLineWrapMode(QTextEdit.NoWrap)
    font = input_files_list_text_area.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    input_files_list_text_area.move(30, 60)
    input_files_list_text_area.resize(500, 100)

    button_select_output_folder = QPushButton(w)
    button_select_output_folder.setText("Output folder")
    button_select_output_folder.setFixedHeight(35)
    button_select_output_folder.setFixedWidth(150)
    button_select_output_folder.clicked.connect(dialog_select_output_folder)
    button_select_output_folder.move(30, 180)

    output_folder_line_edit = QLineEdit(w)
    output_folder_line_edit.setReadOnly(True)
    font = output_folder_line_edit.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    output_folder_line_edit.move(30, 220)
    output_folder_line_edit.setFixedWidth(500)
    output_folder_line_edit.setText(root['output_folder'])

    progress_bar = QProgressBar(w)
    # progress_bar.move(30, 310)
    progress_bar.setValue(0)
    progress_bar.setGeometry(30, 310, 500, 35)
    progress_bar.setAlignment(QtCore.Qt.AlignCenter)
    progress_bar.hide()
    root['progress_bar'] = progress_bar

    button_start = QPushButton('Start separation', w)
    button_start.clicked.connect(w.execute_long_task)
    button_start.setFixedHeight(35)
    button_start.setFixedWidth(150)
    button_start.move(30, 270)

    button_finish = QPushButton('Stop separation', w)
    button_finish.clicked.connect(w.stop_separation)
    button_finish.setFixedHeight(35)
    button_finish.setFixedWidth(150)
    button_finish.move(200, 270)
    button_finish.setDisabled(True)

    mvsep_link = QLabel(w)
    mvsep_link.setOpenExternalLinks(True)
    font = mvsep_link.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    mvsep_link.move(415, 30)
    mvsep_link.setText('Powered by <a href="https://mvsep.com">MVSep.com</a>')

    root['w'] = w
    root['input_files_list_text_area'] = input_files_list_text_area
    root['output_folder_line_edit'] = output_folder_line_edit
    root['button_start'] = button_start
    root['button_finish'] = button_finish

    # w.showMaximized()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    create_dialog()
