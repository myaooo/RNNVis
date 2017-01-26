"""
Utility functions for reading recorded summaries from TF event files
"""

from tensorflow.python.summary import event_file_inspector as inspector
from tensorflow.python.framework.tensor_util import MakeNdarray as convert2ndarray
from collections import namedtuple

# TensorNode = namedtuple('TensorNode', ['name', 'tensor'])
LSTMStateNode = namedtuple('LSTMStateNode', ['c', 'h'])
EvalNode = namedtuple('EvalNode', ['input', 'input_embedding', 'states', 'output', 'output_project'])


def load_event_file(event_path):
    g = inspector.generator_from_event_file(event_path)
    events = []
    next(g)  # skip the firt one
    for e in g:
        values = e.summary.value
        node_dict = {}
        for v in values:
            node_dict[v.node_name] = convert2ndarray(v.tensor)
        states = []
        states_c = []
        states_h = []
        for name, tensor in node_dict.items():
            if name.startswith("state_layer"):
                name_list = name.split(sep="_")
                if len(name_list) == 3:
                    states.append(tensor)
                elif len(name_list) == 4:
                    if name_list[3] == 'c':
                        states_c.append(tensor)
                    else:
                        states_h.append(tensor)
        if len(states):
            assert (not len(states_c)) and (not len(states_h))
        else:
            assert len(states_c) == len(states_h)
            states = []
            for c, h in zip(states_c, states_h):
                states.append(LSTMStateNode(c, h))
        node = EvalNode(node_dict['input'], node_dict['input_embedding'], states, node_dict['output'], None)
        events.append(node)
    return events


if __name__ == '__main__':

    nodes = load_event_file('../../model/LSTM-PTB/evaluate/events.out.tfevents.1484915486.191host015')

