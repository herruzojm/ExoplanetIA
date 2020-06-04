import io
import json
import os
import base64
import pandas as pd
import numpy as np

from torchvision import models
from flask import Flask, flash, jsonify, request, redirect, render_template, session, url_for
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_babel import Babel, _
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import fft

from modelo_perceptron import *
from modelo_lstm import *
from config import *
from forms import *

app = Flask(__name__)
babel = Babel(app)
bootstrap = Bootstrap(app)
app.config.from_object(Config)
app.secret_key = app.config['MODEL_FOLDER']

# Cargamos el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

modelLSTM = ModeloLSTM()
modelLSTM.load_state_dict(torch.load(app.config['MODEL_FOLDER'] + app.config['MODEL_NAMES'][0], map_location = device))
modelLSTM.eval()

modelPerceptron = Perceptron()
modelPerceptron.load_state_dict(torch.load(app.config['MODEL_FOLDER'] + app.config['MODEL_NAMES'][1], map_location = device))
modelPerceptron.eval()

# Creamos el directorio para subir los ficheros
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    form = UploadFileForm()
    return render_template('index.html', form = form)

@app.route('/data/')
def data():
    return render_template('data.html')

@app.route('/kepler/')
def kepler():
    return render_template('kepler.html')
	
@app.route('/help/')
def help():
    return render_template('help.html')
	
@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/language/<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(url_for('index'))

@app.route('/', methods=['POST'])
def predict():  
    if request.method == 'POST':
        form = UploadFileForm(request.form)
        model = form.select.data        

        if 'file' not in request.files:
            flash('_(Archivo no encontrado en la petición)', 'danger')                        
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash(_('Archivo no seleccionado'), 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df, predictions = inference(filepath, model)
            labels = [_('Estrellas'), _('Exoplanetas Detectados')]
            hovercolors = [ "rgb(199, 5, 5)", "rgb(0, 200, 234)" ]
            colors = [ "rgb(150, 4, 4)", "rgb(0, 154, 174)" ]
            exoplanets_indexes = predictions.nonzero().tolist()
            values = [len(predictions) - len(exoplanets_indexes), len(exoplanets_indexes)]
            results = {
                'no_exoplanets': values[0],  
                'exoplanets': values[1],
                'total_stars': len(predictions),
                'indexes': exoplanets_indexes
            }
            images = plot_flux(df, exoplanets_indexes)
            return render_template('results.html', results = results, images = images, 
                                   values = values, labels = labels, colors = colors, hovercolors = hovercolors)
        else:
            flash(_('Solo se permiten archivos con extensión csv'), 'danger')
            return redirect(request.url)        

def z_score_normalizing(df):
    return (df - df.mean()) / df.std()  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def plot_flux(df, indexes):
    images = {}
    indexes = [i for element in indexes for i in element]
    flux_points = np.arange(len(df.iloc[0])) * (36.0/60.0)
    frequency_points = np.arange(len(df.iloc[0])//2) * (1/(36.0/60.0))

    for i in indexes:        
        flux = df.iloc[i]
        
        image_flux = create_image(flux, i, flux_points, 'Time, hours')
        
        fourier = np.abs(fft(flux))
        fourier = fourier[:len(fourier)//2]
        image_frequency = create_image(fourier, i, frequency_points, 'Frequency')

        images[i] = [image_flux, image_frequency]
    return images

def create_image(points, i, time_points, x_text):        
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('Star {}'.format(i))
    axis.set_ylabel('Flux')
    axis.set_xlabel(x_text)
    axis.grid()
    axis.plot(time_points, points)
        
    # Convertimos el grafico a png
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return pngImageB64String

def inference(filepath, selected_model):
    df = pd.read_csv(filepath, low_memory=False)
    if df.shape[1] == 3198:
        df.drop(df.columns[0], axis = 1, inplace = True)
    df = df.apply(z_score_normalizing, axis = 1)
    df_tensor = torch.tensor(df.values).float()
    if selected_model == 'LSTM':
        model = modelLSTM
        tensor = df_tensor.view(len(df_tensor), 1, -1)
    else:
        model = modelPerceptron
        tensor = df_tensor
    predictions = torch.argmax(model(tensor), 1)
    return df, predictions

@babel.localeselector
def get_locale():
    try:
        language = session['language']
    except KeyError:
        language = None
    if language is not None:
        return language
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys())

@app.context_processor
def inject_languages_var():
    return dict(
                AVAILABLE_LANGUAGES = app.config['LANGUAGES'],
                CURRENT_LANGUAGE = session.get('language',request.accept_languages.best_match(app.config['LANGUAGES'].keys())))    
