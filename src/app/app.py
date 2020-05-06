import io
import json
import os
import base64
import pandas as pd

from torchvision import models
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from flask import Flask, flash, jsonify, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from modelo_perceptron import *
from modelo_lstm import * 
from utils import *

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.secret_key = 'key'
MODEL_NAME = './models/perceptron_adam_cross_mini.pth'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['csv'])

# Cargamos el modelo
model = Perceptron()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model.load_state_dict(torch.load(MODEL_NAME, map_location = device))
model.eval()


class UploadFileForm(FlaskForm):
    file = FileField('Selecciona un fichero csv con los datos')
    submit = SubmitField('Cargar')
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def plot_flux(df, indexes):
    images = []
    for i in indexes:
        i = i[0]
        flux = df.iloc[i,:]
        
        # 80 days / 3197 columns = 36 minutes
        time = np.arange(len(flux)) * (36.0/60.0) # plotting each hour
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title('Flux of star {}'.format(i))
        axis.set_ylabel('Flux, e-/s')
        axis.set_xlabel('Time, hours')        
        axis.grid()
        axis.plot(time, flux)
           
        # Convertimos el grafico a png
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
    
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
        images.append(pngImageB64String)
    return images

def inference(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df_tensor = torch.tensor(df.values).float()
    predictions = torch.argmax(model(df_tensor), 1)
    exoplanets = predictions.nonzero()
    return df, predictions

@app.route('/')
def index():
    form = UploadFileForm()
    return render_template('index.html', form = form)

@app.route('/', methods=['POST'])
def predict():  
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Archivo no encontrado en la petición', 'danger')                        
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('Archivo no seleccionado', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            df, predictions = inference(filepath)
            labels = ["Estrellas", "Exoplanetas Detectados"]            
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
            flash('Solo se permiten archivos con extensión csv', 'danger')
            return redirect(request.url)        

        
