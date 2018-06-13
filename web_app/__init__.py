import os

import cv2
import pandas as pd

from sklearn.externals import joblib
from flask import Flask, render_template, redirect, url_for, request, flash


def create_app(test_config=None):
    SECRET_KEY = '?\xbf,\xb4\x8d\xa3"<\x9c\xb0@\x0f5\xab,w\xee\x8d$0\x13\x8b83'

    app = Flask(__name__, instance_relative_config=True)
    app.config['SECRET_KEY'] = SECRET_KEY

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    surf = cv2.xfeatures2d_SURF.create(5000, 4, 3, True, False)
    classifier = joblib.load('{}/model/classifier.svc'.format(app.instance_path))

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1] in ['png', 'jpg', 'jpeg', 'gif']

    @app.route('/')
    def home():
        return(render_template('base.html'))

    @app.route('/result/<id>')
    def result(id):
        img = cv2.imread(os.path.join(app.instance_path, id))
        _, descriptor = surf.detectAndCompute(img, None)
        descriptor = pd.DataFrame(descriptor)
        descriptor.insert(0, 'id', 'imagen')
        descriptor.to_csv('{}/temp/features.csv'.format(app.instance_path), sep=';', index=False, header=False)
        os.system('java -jar {}/model/openXBOW.jar -i {}/temp/features.csv -o {}/temp/bow.csv -b {}/model/codebook.txt'.format(*[app.instance_path]*4))
        descriptor = pd.read_csv('{}/temp/bow.csv'.format(app.instance_path), sep=';', header=None)
        descriptor = classifier.predict(descriptor)
        print(descriptor)
        flash('label: {}'.format(*descriptor), 'alert alert-success')
        return(redirect(url_for('home')))

    @app.route('/upload', methods=['POST'])
    def upload():
        if 'image' not in request.files:
            flash('No file', category='alert alert-danger')
            return redirect(url_for('home'))
        
        file = request.files['image']

        if file.filename == '':
            flash('No filename', category='alert alert-danger')
            return redirect(url_for('home'))
        
        if not allowed_file(file.filename):
            flash('Invalid file format', category='alert alert-danger')
            return redirect(url_for('home'))

        f = os.path.join(app.instance_path, 'temp', file.filename)
        file.save(f)
        flash('Upload successful', category='alert alert-success')
        return(redirect(url_for('result', id=file.filename)))    

    return app