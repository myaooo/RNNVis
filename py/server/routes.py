from py.server import app
from py.server import _manager


@app.route('/models/<model_name>')
def models(model_name):
    _manager.load_model(model_name)
