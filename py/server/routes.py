from py.server import app
from py.server import _manager


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/models/available')
def available_models():
    return _manager.available_models


@app.route('/models/generate/<str:name>')
def model_generate(name):
    result = _manager.model_generate(name)
    if result is None:
        return 'Cannot generate using the given url!', 404
    return result


