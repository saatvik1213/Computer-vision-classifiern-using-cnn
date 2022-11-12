#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 06:14:22 2022

@author: saatvikchoudhary
"""
import cv2 as cv


class Camera:

    def __init__(self):
        self.camera = cv.VideoCapture(0)
        self.camera.set(3, 100)
        self.camera.set(4, 100)
#        self.camera.set(3,300)
 #       self.camera.set(4,300)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)


    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None