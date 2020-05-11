import os

class Config():
	LANGUAGES = {
		'es': 'Español',
		'en': 'English'
	}
	UPLOAD_FOLDER = 'uploads'
	MODEL_NAME = './models/perceptron_adam_cross_mini.pth'
	UPLOAD_FOLDER = './uploads'
	ALLOWED_EXTENSIONS = set(['csv'])
	SECRET_KEY = os.urandom(32)
	