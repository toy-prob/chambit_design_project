import time
import cv2
import imutils
import platform
import numpy as np

# Python은 기본적으로 Single Thread로 실행
# 즉 한 개의 쓰레드로 순차적으로 실행하다 보니, 무거운 OpenCV를 실행하면 성능의 저하를 가져옴
# Python에는 Thread를 사용하면 성능 향상을 볼 수 있으며, 처리된 데이터를 유실없이 처리하기 위해, Queue를 사용
from threading import Thread
from queue import Queue

class Streamer :
    # 초기 선언
    def __init__(self ):
        
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
    
    # Camera On
    def run(self, src = 0 ) :
        
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
                (grabbed, frame) = self.capture.read()
                if grabbed : 
                    self.Q.put(frame)

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