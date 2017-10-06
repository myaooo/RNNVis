// import {
//   SentenceRecord,
//   CoClusterProcessor,
//   StateProcessor,
// } from '../service/preprocess';

// import {
//   getTextEvaluation,
//   getStateStatistics,
//   getCoCluster,
//   getPOSStatistics,
// } from '../service/dataService';

import {
  isString,
  // memorizePromise,
} from '../service/utils';

const cell2states = {
  GRU: ['state'],
  BasicLSTM: ['state_c', 'state_h'],
  BasicRNN: ['state'],
};

// function getCoClusterProcessor(modelName, stateName, layer, params = {
//   topK: 500,
//   mode: 'raw',
//   nCluster: 10,
// }) {
//   return getCoCluster(modelName, stateName, layer, params)
//     .then(data => new CoClusterProcessor(data));
// }

// function getSentenceRecord(modelName, sentence) {
//   return getTextEvaluation(modelName, sentence)
//     .then(data => new SentenceRecord(data));
// }

// function getStateProcessor(modelName, stateName, layer) {
//   return getStateStatistics(modelName, stateName, layer)
//     .then(data => new StateProcessor(data));
// }

function removeIfExists(arr, item) {
  const idx = arr.findIndex(v => item === v);
  if (idx === -1) {
    arr.push(item);
    return arr;
  }
  return arr.slice(0, idx).concat(arr.slice(idx + 1));
}

export default class RNNModel {
  constructor(name, config, style) {
    this.name = name;
    this.config = config;
    this.style = style;
    this.isRunnable = false;
    this.isLoading = false;
    this.hasRenderPOS = false;
    this.coCluster = null;
    this.stateStats = null;
    // this.coClusterCache = {};
    this.layerNum = config.model.cells.length;
    this.layerSizes = config.model.cells.map(cell => cell.num_units);
    this.cellType = config.model.cell_type;
    this.stateList = cell2states[config.model.cell_type];
    this.availableStates = new Set(this.stateList);

    if (config.model.app === 'seq2seq') {
      this.availableStates = ['state_h_en', 'state_h_de'];
    }

    this.selectedState = null;
    this.selectedLayer = null;
    this.nCluster = 0;
    this.selectedUnits = [];
    this.selectedWords = [];
    this.topK = 500;
    this.connectionMode = 'raw';

    this.needRendering = false;
  }

  isLegalState(stateName = this.selectedState) {
    return isString(stateName) && this.availableStates.has(stateName);
  }

  isLegalLayer(layer = this.selectedLayer) {
    return Number.isInteger(layer) && layer > -2 && layer < this.layerNum;
  }

  selectState(stateName) {
    if (this.isLegalState(stateName)) {
      this.selectedState = stateName;
    }
  }

  selectLayer(layer) {
    if (this.isLegalLayer(layer)) {
      this.selectedLayer = layer === -1 ? (this.layerNum - 1) : layer;
    }
  }

  selectUnit(unit) {
    if (Number.isInteger(unit) && unit >= 0 && unit < this.layerSize()) {
      this.selectedUnits = removeIfExists(this.selectedUnits, unit);
    }
  }

  selectWord(word) {
    if (word) {
      this.selectedWords = removeIfExists(this.selectedWords, word);
    }
  }

  layerSize(layer = this.selectedLayer) {
    const theLayer = layer === -1 ? (this.layerNum - 1) : layer;
    return this.layerSizes[theLayer];
  }

  static defaultStyle() {
    return {
      // clusterNum: 5,
      strokeControlStrength: 5,
      linkFilterThreshold: [0, 1],
      stateClip: 1,
      mode: 'width',
      renderPOS: false,
    };
  }

}
