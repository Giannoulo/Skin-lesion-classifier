import os
from flask import Flask, flash, render_template, redirect, request, url_for, send_file
import efficientnetB0_model
import tensorflow as tf
# from flask_bootstrap import Bootstrap

# App config
app = Flask(__name__)
app.config['SECRET_KEY'] = "supertopsecretprivatekey"
app.config['UPLOAD_FOLDER'] = '/tmp/'
# Bootstrap(app)

# Get the tf graph
global graph
graph = tf.get_default_graph()

# Load the efficientnetB0 model
eff_B0_model = efficientnetB0_model.load_EfficientnetB0(app.root_path)

# Check for suitable file extentions
ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])


def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            passed = False
            try:
                filename = image_file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False
            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)

# Serve images route
@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


# Predict image class route
@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results = efficientnetB0_model.predict_class(filepath, eff_B0_model, graph)
    return render_template('predict.html', results=results)

# Serve Error
@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


if __name__ == "__main__":
    app.run('127.0.0.1')
