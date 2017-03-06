"""
Application of RNNVis
"""
from flask import Flask
from flask_cors import CORS
from rnnvis.server.model_manager import ModelManager
app = Flask(__name__)
_manager = ModelManager()
CORS(app)
from rnnvis.server.routes import *
