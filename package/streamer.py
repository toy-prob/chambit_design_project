import argparse
import logging
import time

import time
import cv2
import imutils
import platform
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Python은 기본적으로 Single Thread로 실행
# 즉 한 개의 쓰레드로 순차적으로 실행하다 보니, 무거운 OpenCV를 실행하면 성능의 저하를 가져옴
# Python에는 Thread를 사용하면 성능 향상을 볼 수 있으며, 처리된 데이터를 유실없이 처리하기 위해, Queue를 사용
from threading import Thread
from queue import Queue

class Streamer :
    # 초기 선언
    def __init__(self):
        
        # 현재 컴퓨터가 OpenCL를 지원하는지 확인하는 메소드.
        # OpenCL을 사용할 경우 CPU보다 향상된 성능을 가져오므로, cv2.ocl.setUseOpenCL(True)와 같이 활성화 한다.
        if cv2.ocl.haveOpenCL() :
            cv2.ocl.setUseOpenCL(True)
        print('[chambit] ', 'OpenCL : ', cv2.ocl.haveOpenCL())
            
        self.capture = None
        self.thread = None
        self.width = 640
        self.height = 360
        self.stat = False
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False

        self.Motionflag = False
        self.Skeletonflag = False
        self.Blinkflag = False

# ===========================================================================================================================
# Motion Detect 환경
        self.FRAMES_TO_PERSIST = 10
        self.MIN_SIZE_FOR_MOVEMENT = 2000
        self.MOVEMENT_DETECTED_PERSISTENCE = 100
        self.THRESH_HOLD = 60
        self.first_frame = None
        self.next_frame = None

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.delay_counter = 0
        self.movement_persistent_counter = self.MOVEMENT_DETECTED_PERSISTENCE
        self.k = 0
        self.block_movement = [[0 for _ in range(self.MOVEMENT_DETECTED_PERSISTENCE)] for _ in range(3)]
# ===========================================================================================================================
# ===========================================================================================================================
# Skeleton Detect 환경
        self.logger = logging.getLogger('TfPoseEstimator-WebCam')
        self.logger.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)

        self.fps_time = 0

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--camera', type=int, default=0)

        self.parser.add_argument('--resize', type=str, default='0x0',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        self.parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        self.parser.add_argument('--model', type=str, default='mobilenet_thin',
                            help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        self.parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')

        self.parser.add_argument('--tensorrt', type=str, default="False",
                            help='for tensorrt process.')

        self.args = self.parser.parse_args()

        self.logger.debug('initialization %s : %s' % (self.args.model, get_graph_path(self.args.model)))

        self.w, self.h = model_wh(self.args.resize)

        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(self.args.model), target_size=(w, h), trt_bool=self.str2bool(self.args.tensorrt))
        else:
            self.e = TfPoseEstimator(get_graph_path(self.args.model), target_size=(432, 368),
                                trt_bool=self.str2bool(self.args.tensorrt))

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")
 # ===========================================================================================================================

    # Camera On
    def run(self, src = 0, selection = 0) :

        if selection == 1:
            self.Motionflag = True
        if selection == 2:
            self.Skeletonflag = True
        if selection == 3:
            self.Motionflag = True
            self.Skeletonflag = True

        self.stop()
    
        if platform.system() == 'Windows' :        
            self.capture = cv2.VideoCapture( src , cv2.CAP_DSHOW )
        
        else :
            self.capture = cv2.VideoCapture( src )
            
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if self.thread is None :
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()
        
        self.started = True

        self.logger.debug('cam read+')
        ret_val, image = self.capture.read()
        self.logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # Camera Off
    def stop(self):
        
        self.started = False
        
        if self.capture is not None :
            
            self.capture.release()
            self.clear()
    
    # 영상 실시간 처리
    # 웹캠이 켜지고, 지속적인 프레임을 추출
    def update(self):
        while True:
            if self.started :
                (_grabbed, _frame) = self.capture.read()
                if self.Motionflag:
                    _grabbed, _frame = self.MotionDetection( _grabbed, _frame)
                if self.Skeletonflag:
                    _grabbed, _frame = self.SkeletonDetection(_grabbed, _frame)
                if _grabbed :
                    self.Q.put(_frame)

    # 영상 데이터 삭제
    # Queue 데이터에 쌓인 데이터를 모두 삭제하여, 공간을 확보       
    def clear(self):
        with self.Q.mutex:
            self.Q.queue.clear()
    
    # 영상 데이터 읽기
    def read(self):
        return self.Q.get()

    # 영상 데이터 부재시 빈 영상 데이터
    # frame의 영상 데이터가 없을 경우, 검은 화면으로 출력
    def blank(self):
        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)
    
    # 영상을 바이너리 코드로 전환
    # OpenCV의 영상을 jpeg 바이너리 변환하여 리턴
    def bytescode(self):
        if not self.capture.isOpened():
            frame = self.blank()

        else :
            frame = imutils.resize(self.read(), width=int(self.width) )
        
            if self.stat :  
                cv2.rectangle( frame, (0,0), (120,30), (0,0,0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText  ( frame, fps, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
            
            
        return cv2.imencode('.jpg', frame )[1].tobytes()
    
    # 영상 FPS 처리
    # 화면에 FPS를 출력
    def fps(self):
        
        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time
        
        if self.sec > 0 :
            fps = round(1/(self.sec),1)
            
        else :
            fps = 1
            
        return fps
    
    # 클래스 종료
    def __exit__(self) :
        print( '* streamer class exit')
        self.capture.release()

    def MotionDetection(self, _grabbed, _frame):

        grabbed = _grabbed
        frame = _frame

        flag = 0

        if self.movement_persistent_counter == 0:
            self.movement_persistent_counter = self.MOVEMENT_DETECTED_PERSISTENCE
            self.k = self.k + 1

        self.k = self.k % 3

        # Set transient motion detected as false
        transient_movement_flag = False

        # If there's an error in capturing
        if not grabbed:
            print("CAPTURE ERROR")

        # Resize and save a greyscale version of the image
        frame = imutils.resize(frame, width=750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if self.first_frame is None: self.first_frame = gray

        self.delay_counter += 1

        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if self.delay_counter > self.FRAMES_TO_PERSIST:
            self.delay_counter = 0
            self.first_frame = self.next_frame

        # Set the next frame to compare (the current frame)
        self.next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(self.first_frame, self.next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > self.MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True

                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The moment something moves momentarily, reset the persistent
        # movement timer.

        # As long as there was a recent transient movement, say a movement
        # was detected
        if transient_movement_flag == True:
            self.block_movement[self.k][self.movement_persistent_counter - 1] = 1
            text = "Movement Detected " + str(self.movement_persistent_counter)
            self.movement_persistent_counter -= 1
        else:
            self.block_movement[self.k][self.movement_persistent_counter - 1] = 0
            text = "No Movement Detected"
            self.movement_persistent_counter -= 1

        # Print the text on the screen, and display the raw and processed video
        # feeds
        cv2.putText(frame, str(text), (10, 35), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # For if you want to show the individual video frames
        #    cv2.imshow("frame", frame)
        #    cv2.imshow("delta", frame_delta)

        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

        # Splice the two video frames together to make one long horizontal one
        cv2.imshow("frame", np.hstack((frame_delta, frame)))

        # Interrupt trigger by pressing q to quit the open CV program
        # ch = cv2.waitKey(1)
        # if ch & 0xFF == ord('q'):
        #    break

        for j in range(3):
            if sum(self.block_movement[j]) > self.THRESH_HOLD:
                flag = flag + 1
            if flag == 3:
                print('\n\nfrequent moving!\n\n')

        return grabbed, frame

    def SkeletonDetection(self, _grabbed, _frame):
        ret_val, image = _grabbed, _frame
        self.logger.debug('image process+')
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.args.resize_out_ratio)
    
        self.logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
        self.logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        self.fps_time = time.time()
        #if cv2.waitKey(1) == 27:
        #    break
        #logger.debug('finished+')
        return ret_val, image