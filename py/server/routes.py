from py.server import app
from py.server import _manager


@app.route("/")
def hello():
    return "Hello World!"

@app.route('/models/<model_name>')
def models(model_name):
    _manager.get_model(model_name)
