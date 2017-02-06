"""
Application of RNNVis
"""
from flask import Flask
from py.server.model_manager import ModelManager
app = Flask(__name__)
_manager = ModelManager()

from py.server.routes import *
