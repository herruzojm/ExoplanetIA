import os

class Config():
	LANGUAGES = {
		'es': 'Español',
		'en': 'English'
	}
	MODEL_NAME = './models/lstm_sgd_cross_smote_diferencia.pth'
	UPLOAD_FOLDER = './uploads'
	ALLOWED_EXTENSIONS = set(['csv'])
	SECRET_KEY = os.urandom(32)
	