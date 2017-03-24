"""
Recorders for Evaluator
"""

from collections import defaultdict

from rnnvis.db.db_helper import insert_evaluation, push_evaluation_records


class Recorder(object):

    def start(self, inputs, targets, pos_tagger=None):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :param pos_tagger: Part-of-Speech Tagger of type lambda list(int): list(str)
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

    def __init__(self, data_name, model_name, set_name=None, flush_every=100):
        self.data_name = data_name
        self.model_name = model_name
        self.set_name = set_name
        self.eval_doc_id = []
        self.buffer = defaultdict(list)
        self.batch_size = 1
        self.input_data = None
        self.input_length = None
        self.flush_every = flush_every
        self.pos_tagger = None
        self.pos_tags = None
        self.step = 0

    def start(self, inputs, targets, pos_tagger=None):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :param pos_tagger: Part-of-Speech Tagger of type lambda list(int): list(str) convert word_ids to pos tags
        :return: None
        """
        self.pos_tagger = pos_tagger
        self.batch_size = inputs.shape[0]
        self.input_data = inputs.full_data
        self.input_length = self.input_data.shape[1]
        # for i in range(self.inputs.shape[0]):
        sentence_num = self.input_data.shape[0]
        sentence_lengths = [count_length(self.input_data[i]) for i in range(sentence_num)]
        sentences = [self.input_data[i, :sentence_lengths[i]].tolist() for i in range(sentence_num)]
        self.write_evaluation(sentences)
        if self.pos_tagger is not None:
            self.pos_tags = [self.pos_tagger(sentence) for sentence in sentences]

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
                if self.pos_tags is not None:
                    record['pos'] = self.pos_tags[start_x + i][start_y]
                good_records.append(record)
                eval_ids.append(self.eval_doc_id[start_x + i])
        self.buffer['records'] += good_records
        self.buffer['eval_ids'] += eval_ids
        self.step += 1
        if len(self.buffer['eval_ids']) >= self.flush_every:
            self.flush()

    def flush(self):
        push_evaluation_records(self.buffer.pop('eval_ids'), self.buffer.pop('records'))

    def write_evaluation(self, sentences):
        self.eval_doc_id = insert_evaluation(self.data_name, self.model_name, self.set_name, sentences, replace=True)

    def close(self):
        pass


class BufferRecorder(StateRecorder):
    """
    A recorder that writes in a memory buffer, instead of DB.
    """
    def __init__(self, data_name, model_name, max_buffer=5000):
        """
        :param data_name:
        :param model_name:
        :param max_buffer: the max number of records that can be stored in the recorder,
            since this recorder store all the data in the memory, it's better to set this value small.
        """
        super(BufferRecorder, self).__init__(data_name, model_name, 'buffer', 100)
        self.max_buffer = max_buffer
        self.eval_docs = None

    def write_evaluation(self, sentences):
        total_size = sum([len(sentence) for sentence in sentences])
        if total_size > self.max_buffer:
            raise ValueError("total size of the evaluating sequence is larger than permitted!")
        self.eval_docs = [{'data': sentence, 'records': []} for sentence in sentences]
        self.eval_doc_id = list(range(len(sentences)))

    def flush(self):
        def _flush(eval_ids, records):
            for i, id_ in enumerate(eval_ids):
                self.eval_docs[id_]['records'].append(records[i])
        _flush(self.buffer.pop('eval_ids'), self.buffer.pop('records'))

    def sentences(self):
        for eval_doc in self.eval_docs:
            yield eval_doc['data']

    def records(self):
        for eval_doc in self.eval_docs:
            yield eval_doc['records']

    def evals(self):
        for eval_doc in self.eval_docs:
            yield eval_doc['data'], eval_doc['records']


def count_length(inputs, marker=-1):
    for i, data in enumerate(inputs):
        if data == marker:
            return i
    return len(inputs)
