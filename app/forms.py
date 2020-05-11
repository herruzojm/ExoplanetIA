from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from flask_babel import lazy_gettext as _l

class UploadFileForm(FlaskForm):
    file = FileField(_l('Selecciona un fichero csv con los datos'))
    submit = SubmitField(_l('Cargar'))