import os

class Config():
	LANGUAGES = {
		'en': 'English',
		'es': 'Espa�ol'
	}
	MODEL_NAME = './models/perceptron_adam_cross_mini.pth'
	UPLOAD_FOLDER = './uploads'
	ALLOWED_EXTENSIONS = set(['csv'])
	SECRET_KEY = os.urandom(32)
	