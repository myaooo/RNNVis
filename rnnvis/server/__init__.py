"""
Application of RNNVis
"""
from flask import Flask
from flask_cors import CORS
from rnnvis.server.model_manager import ModelManager
from rnnvis.utils.io_utils import get_path
from rnnvis.procedures import init_tf_environ
path = get_path('frontend/dist/static', absolute=True)
init_tf_environ(1)
print("Static folder: {:s}".format(path))
app = Flask(__name__)
app.config['FRONT_END_ROOT'] = get_path('frontend/dist', absolute=True)
app.config['STATIC_FOLDER'] = get_path('frontend/dist/static', absolute=True)
_manager = ModelManager()
CORS(app)
from rnnvis.server.routes import *
