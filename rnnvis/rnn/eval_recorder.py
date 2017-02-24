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
        self.input_data = None
        self.input_length = None
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
        self.input_data = inputs.full_data
        self.input_length = self.input_data.shape[1]
        # for i in range(self.inputs.shape[0]):
        sentence_num = self.input_data.shape[0]
        sentence_lengths = [count_length(self.input_data[i]) for i in range(sentence_num)]
        self.eval_doc_id = insert_evaluation(self.data_name, self.model_name,
                                             [self.input_data[i, :sentence_lengths[i]].tolist()
                                              for i in range(sentence_num)], replace=True)

    def record(self, record_message):
        """
        Record one step of information, note that there is a batch of them
        :param record_message: a dict, with keys as summary_names,
            and each value as corresponding record info [batch_size, ....]
        :return:
        """
        records = [{name: value[i] for name, value in record_message.items()} for i in range(self.batch_size)]
        start_x = self.step // self.input_length * self.batch_size
        start_y = self.step % self.input_length

        good_records = []
        eval_ids = []
        for i, record in enumerate(records):
            word_id = int(self.input_data[start_x + i, start_y])
            record['word_id'] = word_id
            if word_id >= 0:
                good_records.append(record)
                eval_ids.append(self.eval_doc_id[start_x + i])
        self.buffer['records'] += good_records
        self.buffer['eval_ids'] += eval_ids
        self.step += 1
        if len(self.buffer['eval_ids']) >= self.flush_every:
            self.flush()

    def flush(self):
        push_evaluation_records(self.buffer.pop('eval_ids'), self.buffer.pop('records'))

    def close(self):
        pass


def count_length(inputs, marker=-1):
    for i, data in enumerate(inputs):
        if data == marker:
            return i
    return len(inputs)
