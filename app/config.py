import os

class Config():
	LANGUAGES = {
		'es': 'Español',
		'en': 'English'
	}
	MODEL_FOLDER = './models/'
	MODELS = ['LSTM', 'Perceptron']
	MODEL_NAMES = ['lstm_sgd_cross_smote_diferencia.pth', 'perceptron_smote_sgd_cross_solo_filtro.pth']
	UPLOAD_FOLDER = './uploads'
	ALLOWED_EXTENSIONS = set(['csv'])
	SECRET_KEY = os.urandom(32)
	