'''Zach's log code modularized'''

import cv2
import numpy as np
import time
import datetime
import csv

LOGDATA_PATH = '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/logFiles/'
LOGVID_PATH = '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/CSI_Front_Camera/'

class Log:
    def __init__(self, save_logs):
        self.save_logs = save_logs  # boolean
    
    def save_video(self, frame_list, isColor=False, image_width=640, image_height=480, sample_rate=30.0, path=LOGVID_PATH, name='videoLog_', append_time=True):
        file_name = self._get_log_name('video', path, name, append_time)
        makeVid = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), sample_rate, (image_width, image_height), isColor)
        for i in range(0,len(frame_list)):
            makeVid.write(frame_list[i])
        makeVid.release()

    def save_data(self, data_fields, data_list, path=LOGDATA_PATH, name='dataLog_', append_time=True):
        file_name = self._get_log_name('data', path, name, append_time)
        with open(file_name, 'a') as csvfile:
            log = csv.writer(csvfile)
            log.writerow(data_fields)
            log.writerows(data_list)

    def _get_log_name(self, type, path, name, append_time):
        if type == 'video': file_type = '.avi'
        if type == 'data': file_type = '.csv'
        file_name = path + name
        if append_time:
            now = self._get_date_time()
            file_name += now
        file_name += file_type
        return file_name

    def _get_date_time(self):
        now = datetime.datetime.now()
        now = str(now)
        now = now.replace(':','-')
        now = now.replace(' ','_')
        now = now[:19]
        return now