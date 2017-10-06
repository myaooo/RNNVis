import {
  SentenceRecord,
  CoClusterProcessor,
  StateProcessor,
} from '../service/preprocess';

import {
  getTextEvaluation,
  getStateStatistics,
  getCoCluster,
  getPOSStatistics,
} from '../service/dataService';

import { memorizePromise } from '../service/utils';

const cell2states = {
  GRU: ['state'],
  BasicLSTM: ['state_c', 'state_h'],
  BasicRNN: ['state'],
};

function getCoClusterProcessor(modelName, stateName, layer, params = {
  topK: 500,
  mode: 'raw',
  nCluster: 10,
}) {
  return getCoCluster(modelName, stateName, layer, params)
    .then(data => new CoClusterProcessor(data));
}

function getSentenceRecord(modelName, sentence) {
  return getTextEvaluation(modelName, sentence)
    .then(data => new SentenceRecord(data));
}

function getStateProcessor(modelName, stateName, layer) {
  return getStateStatistics(modelName, stateName, layer)
    .then(data => new StateProcessor(data));
}

export default class RNNModel {
  constructor(name, config, layout) {
    this.name = name;
    this.config = config;
    this.layout = layout;
    this.isRunnable = false;
    this.isLoading = false;
    this.hasRenderPOS = false;
    this.coClusterCache = {};
    this.layerNum = config.model.cells.length;
    this.cellType = config.model.cell_type;
    this.availableStates = cell2states[config.model.cell_type];
    if (config.model.app === 'seq2seq') {
      this.availableStates = ['state_h_en', 'state_h_de'];
    }
    this.getCoClusterProcessor = memorizePromise((params) =>
      getCoClusterProcessor(this.name, this.selectedState, this.selectedLayer, params));
    this.getStateProcessor = memorizePromise(() =>
      getStateProcessor(this.name, this.selectedState, this.selectedLayer));
    this.getSentenceRecord = memorizePromise((sentence) => getSentenceRecord(this.name, sentence));
    this.getPOSStatistics = memorizePromise((topK) => getPOSStatistics(this.name, topK));

    this.selectedState = this.availableStates[0];
    this.selectedLayer = this.layerNum - 1;
    this.selectedUnits = [];
    this.selectedWords = [];
  }

  layerSize(layer = -1) {
    return this.config.model.cells[layer].num_units;
  }
}
