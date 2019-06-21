import os
import cv2
import create_detections
from PIL import Image
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/Users/Matthew/front_end/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))
    return render_template('home.html')
    
@app.route('/show/<filename>')
def uploaded_file(filename):
    rs,sb = create_detections.detect('model_Octonauts.pb', filename)
    im = Image.open(sb)
    width, height = im.size
    im.close()
    if (os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], sb)) != True):
        os.rename(sb, UPLOAD_FOLDER + '/' + sb)
    
    sb = 'http://127.0.0.1:5000/uploads/' + sb
    results = open(rs, "r")
    data =  results.read().replace('\n', ',')
    return render_template('results.html', filename=sb, myList=data, myHeight=height,myWidth=width)

@app.route('/examples')
def go_to_examples():
    filename1 = 'http://127.0.0.1:5000/uploads/' + '107_Octonauts.png'
    filename2 = 'http://127.0.0.1:5000/uploads/' + '5_Octonauts.png'
    filename3 = 'http://127.0.0.1:5000/uploads/' + '20_Octonauts.png'

    return render_template('examples.html', pic1 = filename1, pic2 = filename2, pic3 = filename3)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results1')
def results1():
    filename = 'http://127.0.0.1:5000/uploads/' + '107_Octonauts.png'
    results = open("107_Octonauts_preds.txt", "r")
    data =  results.read().replace('\n', ',')
    height = 2607
    width = 3096
    return render_template('results.html', filename=filename, myList=data, myHeight=height,myWidth=width)

@app.route('/results2')
def results2():
    filename = 'http://127.0.0.1:5000/uploads/' + '5_Octonauts.png'
    results = open("5_Octonauts_preds.txt", "r")
    data =  results.read().replace('\n', ',')
    height = 2931
    width = 3093
    return render_template('results.html', filename=filename, myList=data, myHeight=height,myWidth=width)

@app.route('/results3')
def results3():
    filename = 'http://127.0.0.1:5000/uploads/' + '20_Octonauts.png'
    results = open("20_Octonauts_preds.txt", "r")
    data =  results.read().replace('\n', ',')
    height = 2931
    width = 3093

    return render_template('results.html', filename=filename, myList=data, myHeight=height,myWidth=width)



if __name__ == '__main__':
    app.debug = True
    app.run()