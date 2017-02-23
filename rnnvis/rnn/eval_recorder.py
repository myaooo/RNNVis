"""
Recorders for Evaluator
"""

from collections import defaultdict

from rnnvis.db.db_helper import insert_evaluation, push_evaluation_records


class Recorder(object):

    def start(self, inputs, targets):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")

    def record(self, message):
        """
        Do some preparations on the message, and then do the recording
        :param message: a dictionary, containing the results of a run of the loo[
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")

    def flush(self):
        """
        Used for flushing the records to the disk / db
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")

    def close(self):
        """
        Called after all the evaluations is done.
        :return:
        """
        raise NotImplementedError("This is the Recorder base class")


class StateRecorder(Recorder):

    def __init__(self, data_name, model_name, flush_every=100):
        self.data_name = data_name
        self.model_name = model_name
        self.eval_doc_id = []
        self.buffer = defaultdict(list)
        self.batch_size = 1
        self.inputs = None
        self.flush_every = flush_every
        self.step = 0

    def start(self, inputs, targets):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :return: None
        """
        self.batch_size = inputs.shape[0]
        self.inputs = inputs.full_data
        for i in range(self.inputs.shape[0]):
            self.eval_doc_id.append(insert_evaluation(self.data_name, self.model_name, self.inputs[i].tolist()))

    def record(self, record_message):
        """
        Record one step of information, note that there is a batch of them
        :param record_message: a dict, with keys as summary_names,
            and each value as corresponding record info [batch_size, ....]
        :return:
        """
        records = [{name: value[i] for name, value in record_message.items()} for i in range(self.batch_size)]
        for i, record in enumerate(records):
            record['word_id'] = int(self.inputs[i, self.step])
        self.buffer['records'] += records
        self.buffer['eval_ids'] += self.eval_doc_id
        self.step += 1
        if len(self.buffer['eval_ids']) >= self.flush_every:
            self.flush()

    def flush(self):
        push_evaluation_records(self.buffer.pop('eval_ids'), self.buffer.pop('records'))

    def close(self):
        pass
