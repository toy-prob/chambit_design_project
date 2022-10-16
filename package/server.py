from flask import Flask
from flask import request
from flask import Response
from flask import stream_with_context

from package.streamer import Streamer

app = Flask( __name__ )
streamer = Streamer()

# 영상을 처리하는 streamer 클래스의 영상 바이너리 코드를 실시간으로 처리하는 함수
def stream_gen( src, mod ):   
    try :
        streamer.run( src, mod )

        while True :
            
            # streamer.bytecode()를 통해 영상을 jpeg로 인코딩하여, yield로 호출한 url로 실시간으로 바이너리를 전송
            frame = streamer.bytescode()

            # yield 키워드를 사용하면 제너레이터를 반환
            # 제너레이터는 여러 개의 데이터를 미리 만들어 놓지 않고 필요할 때마다
            # 즉석해서 하나씩 만들어낼 수 있는 객체를 의미
            # 파일 시스템으로 갔다오지 않고 클라이언트로 바로 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
    except GeneratorExit :
        streamer.stop()

# Index
@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1><a href="/">Test Capstone</a></h1>
            <ul>
                <li><a href="/stream?src=0/">Moniter</a></li>
                <li><a href="/setting">Setting</a></li>
            </ul>
        </body>
    </html>
    '''

# Setting
@app.route('/setting')
def setting():
    return '''
    <html>
        <body>
            <h1><a href="/">Test Capstone</a></h1>
            <ul>
                <li><a href="/setting/MotionDetection">Setting 1</a></li>
                <li><a href="/setting/SkeletonDetection">Setting 2</a></li>
                <li><a href="/setting/Detection">Setting 3</a></li>
            </ul>
        </body>
    </html>
    '''

@app.route('/setting/MotionDetection')
def MotionDetection():
    src = request.args.get('src', default=0, type=int)

    try:
        # 웹브라우저에 streaming으로 전달하는 코드, 웹 header의 mineType을 multipart/x-mixed-replace로 선언
        return Response(
            stream_with_context(stream_gen( src, 1) ),
            mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        print('[chambit] ', 'stream error : ', str(e))

@app.route('/setting/SkeletonDetection')
def SkeletonDetection():
    src = request.args.get('src', default=0, type=int)

    try:
        # 웹브라우저에 streaming으로 전달하는 코드, 웹 header의 mineType을 multipart/x-mixed-replace로 선언
        return Response(
            stream_with_context(stream_gen( src, 2) ),
            mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        print('[chambit] ', 'stream error : ', str(e))

@app.route('/setting/Detection')
def Detection():
    src = request.args.get('src', default=0, type=int)

    try:
        # 웹브라우저에 streaming으로 전달하는 코드, 웹 header의 mineType을 multipart/x-mixed-replace로 선언
        return Response(
            stream_with_context(stream_gen( src, 3) ),
            mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        print('[chambit] ', 'stream error : ', str(e))


# flask를 통한 서버의 url 호출
@app.route('/stream')
def stream():
    src = request.args.get( 'src', default = 0, type = int )
    
    try :
        # 웹브라우저에 streaming으로 전달하는 코드, 웹 header의 mineType을 multipart/x-mixed-replace로 선언
        return Response(
                                stream_with_context( stream_gen( src, 0 ) ),
                                mimetype='multipart/x-mixed-replace; boundary=frame' )
        
    except Exception as e :
        print('[chambit] ', 'stream error : ',str(e))