"""
Application of RNNVis
"""
from flask import Flask
from rnnvis.server.model_manager import ModelManager
app = Flask(__name__)
_manager = ModelManager()

from rnnvis.server.routes import *
