import yaml
from flask import jsonify, send_file, request
from rnnvis.server import app
from rnnvis.server import _manager


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/models/available')
def available_models():
    return jsonify({'models': _manager.available_models})


@app.route('/models/generate')
def model_generate():
    model = request.args.get('model', '')
    seeds = request.args.get('seeds', '')
    branch = request.args.get('branch', 1)
    accum_cond = request.args.get('accum_cond', 1.0)
    min_cond = request.args.get('min_cond', 0.0)
    min_prob = request.args.get('min_prob', 0.0)
    step = request.args.get('step', 20)
    neg_words = request.args.get('neg_words', '')
    seeds = seeds.split(sep='+')
    neg_words = neg_words.split(sep='+')
    result = _manager.model_generate(model, seeds, branch, accum_cond,
                                     min_cond, min_prob, step, neg_words)
    if result is None:
        return 'Cannot generate using the given url!', 404
    return result


# @app.route('/models/evaluate/<string:name>&<string:sequence>')
# def model_evaluate(name, sequence):
#     sequence = sequence.split(sep='+')
#     result = _manager.model_evaluate(name, sequence)
#     if result is None:
#         return 'Cannot find model with name {:s}'.format(name), 404
#     return result


@app.route('/models/config/<string:model>')
def model_config(model):
    result = _manager.get_config_filename(model)
    if result is None:
        return 'Cannot find model with name {:s}'.format(model), 404
    return jsonify(yaml.load(open(result)))


@app.route('/state_signature')
def state_signature():
    model = request.args.get('model', '')
    state_name = request.args.get('state', '')
    layer = int(request.args.get('layer', -1))
    sample_size = int(request.args.get('size', 1000))
    try:
        state_signature_ = _manager.model_state_signature(model, state_name, layer, sample_size)
        if state_signature_ is None:
            return 'Cannot find model with name {:s}'.format(model), 404
        return jsonify(state_signature_)
    except LookupError:
        return 'No state records for model {:s} available!'.format(model), 404
    except:
        return 'page not found', 404


@app.route('/strength')
def word_state_strength():
    model = request.args.get('model', '')
    state_name = request.args.get('state', '')
    layer = int(request.args.get('layer', -1))
    top_k = int(request.args.get('top_k', 100))
    try:
        strength = _manager.model_strength(model, state_name, layer, top_k)
        if strength is None:
            return 'Cannot find model with name {:s}'.format(model), 404
        return strength
    except ValueError:
        return 'too large top_k', 404
    except:
        raise
        # return 'page not found', 404


@app.route('/projection')
def state_projection():
    model = request.args.get('model', '')
    state_name = request.args.get('state', '')
    layer = int(request.args.get('layer', -1))
    method = request.args.get('method', 'tsne')
    try:
        projection = _manager.model_state_projection(model, state_name, layer, method)
        if projection is None:
            return 'Cannot find model with name {:s}'.format(model), 404
        return projection
    except:
        raise
        # return 'page not found', 404
