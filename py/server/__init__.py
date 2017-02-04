"""
Application of RNNVis
"""
from flask import Flask
from py.server.model_manager import ModelManager
app = Flask(__name__)
_manager = ModelManager()

from py.server.routes import *


@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()