# $ pip install --upgrade pip
# $ pip install cython
# $ pip install "numpy<17"
# $ pip install imutils
# $ pip install flask
# $ pip install opencv-python
# $ pip install opencv-contrib-python

from package.server import app

if __name__ == '__main__' :
    print('CV on')
    app.run(host='0.0.0.0', port=5000, debug=True)