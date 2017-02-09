import yaml
from flask import jsonify, send_file, request
from py.server import app
from py.server import _manager


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/models/available')
def available_models():
    return jsonify(_manager.available_models)


@app.route('/models/generate')
def model_generate():
    name = request.args.get('name', '')
    seeds = request.args.get('seeds', '')
    branch = request.args.get('branch', 1)
    accum_cond = request.args.get('accum_cond', 1.0)
    min_cond = request.args.get('min_cond', 0.0)
    min_prob = request.args.get('min_prob', 0.0)
    step = request.args.get('step', 20)
    neg_words = request.args.get('neg_words', '')
    seeds = seeds.split(sep='+')
    neg_words = neg_words.split(sep='+')
    result = _manager.model_generate(name, seeds, branch, accum_cond,
                                     min_cond, min_prob, step, neg_words)
    if result is None:
        return 'Cannot generate using the given url!', 404
    return result


@app.route('/models/evaluate/<string:name>&<string:sequence>')
def model_evaluate(name, sequence):
    sequence = sequence.split(sep='+')
    result = _manager.model_evaluate(name, sequence)
    if result is None:
        return 'Cannot find model with name {:s}'.format(name), 404
    return result


@app.route('/models/config/<string:name>')
def model_config(name):
    result = _manager.get_config_filename(name)
    if result is None:
        return 'Cannot find model with name {:s}'.format(name), 404
    return jsonify(yaml.load(open(result)))

