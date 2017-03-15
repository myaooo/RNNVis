import Vue from 'vue';
import dataService from './services/dataService';

const state = {
  selectedModel: null,
  selectedState: null,
  modelConfigs: {},
  coClusters: {},
  availableModels: null,
  // sentenceRecords: [],
};

const bus = new Vue({
  data: {
    state: state,
    cell2states: {
      'GRU': ['state'],
      'BasicLSTM': ['state_c', 'state_h'],
      'BasicRNN': ['state'],
    },
  },
  computed: {
  },
  methods: {

    loadModelConfig(modelName = this.state.selectedModel) { // return a Promise
      if (!Object.prototype.hasOwnProperty.call(state.modelConfigs, modelName)) {
        return dataService.getModelConfig(modelName, response => {
          if (response.status === 200) {
            state.modelConfigs[modelName] = response.data;
            // state.selectedModel = modelName;
          }
        });
      }
      return Promise.resolve(this.state.modelConfigs[modelName]);
    },

    loadAvailableModels() {
      // console.log(this.availableModels);
      if (this.state.availableModels === null) {
        return dataService.getModels(response => {
          if (response.status === 200) {
            const data = response.data;
            this.state.availableModels = data.models;
          } else throw response;
        });
      }
      return Promise.resolve('Already Loaded');
    },
    loadCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 300, mode: 'raw' }) {
      const coCluster = new CoClusterProcessor(modelName, stateName, nCluster, params);
      const coClusterName = CoClusterProcessor.identifier(coCluster);
      if (this.state.coClusters.hasOwnProperty(coClusterName))
        return Promise.resolve('Cocluster data already loaded');
      return this.loadAvailableModels()
        .then(() => {
          if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)) {
            return coCluster.load();
          }
          throw `No model named ${modelName}`;
        })
        .then(() => {
          this.state.coClusters[coClusterName] = coCluster;
          return 'Succeed';
        });
    },
    getCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 300, mode: 'raw' }) {
      const coCluster = new CoClusterProcessor(modelName, stateName, nCluster, params);
      const coClusterName = CoClusterProcessor.identifier(coCluster);
      if (this.state.coClusters.hasOwnProperty(coClusterName))
        return this.state.coClusters[coClusterName];
      console.log('First call loadCoCluster(...) to load remote Data!');
      return undefined;
    },
    getModelConfig(modelName = this.selectedModel) {
      if (this.state.availableModels)
        return this.state.availableModels[modelName];
      return undefined;
    },
    availableStates(modelName = this.state.selectedModel) { // helper function that returns available states of the current selected Model`
      // modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)) {
        const config = this.state.modelConfigs[modelName];
        return this.cell2states[config.model.cell_type];
      }
      return undefined;
    },
    layerNum(modelName = this.selectedModel) {
      // modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)) {
        const config = this.state.modelConfigs[modelName];
        return config.model.cells.length;
      }
      return undefined;
    },
    layerSize(modelName = this.state.selectedModel, layer = -1) {
      // modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)) {
        if (layer === -1) {
          layer = this.layerNum(modelName) - 1;
        }
        const config = this.state.modelConfigs[modelName];
        return config.model.cells[layer].num_units;
      }
      return undefined;
    }
  },
});

// event definitions goes here
const SELECT_MODEL = 'SELECT_MODEL';

// register event listener
bus.$on(SELECT_MODEL, (modelName) => {
  bus.state.selectedModel = modelName;
});

class CoClusterProcessor {
  constructor(modelName, stateName, nCluster = 10, params = { top_k: 300, mode: 'raw' }) {
    this.rawData;
    this.modelName = modelName;
    this.stateName = stateName;
    this.nCluster = nCluster;
    this.params = params;
    // this.s
  }
  get correlation() {
    return this.hasData ? this.rawData.data : undefined;
  }
  get labels() {
    if (this.hasData) {
      this._labels = this.rowLabels.filter((v, i, arr) => { return arr.indexOf(v) === i; });
      this._labels.sort();
      return this._labels;
    }
    return this._labels;
  }
  get rowLabels() {
    return this.hasData ? this.rawData.row : undefined;
  }
  get colLabels() {
    return this.hasData ? this.rawData.col : undefined;
  }
  get ids() {
    return this.hasData ? this.rawData.ids : undefined;
  }
  get words() {
    return this.hasData ? this.rawData.words : undefined;
  }
  load() {
    return dataService.getCoCluster(this.modelName, this.stateName, this.nCluster, this.params, (response) => {
      if (response.status === 200) {
        this.rawData = response.data;
      } else {
        throw response;
      }
    });
  }
  get hasData() {
    return Boolean(this.rawData);
  }
  get rowClusters() {
    if (this.hasData) {
      // delete this.rowClusters;
      const range = Array.from({ length: this.rowLabels.length }, (v, i) => i);
      return this._rowClusters = this.labels.map((label) => {
        return range.filter((i) => this.rowLabels[i] == label);
      });
    }
    return this._rowClusters;
  }
  get colClusters() {
    if (this.hasData) {
      // delete this.colClusters;
      const range = Array.from({ length: this.colLabels.length }, (v, i) => i);
      return this._colClusters = this.labels.map((label) => {
        return range.filter((i) => this.colLabels[i] == label);
      });
    }
    return this._colClusters;
  }
  static identifier(processor) {
    return `${processor.modelName}_${processor.stateName}_${processor.nCluster}`;
  }

}

class SentenceRecord{
  constructor(inputs) {
    this.inputs = inputs;
    this.tokens;
    this.records;
  }
  evaluate(modelName) {
    return dataService.getTextEvaluation(modelName, this.inputs, (response => {
      if(response.status === 200){
        const data = response.data;
        this.tokens = data.tokens;
        this.records = data.records;
      }
    }));
  }
  get states() {
    if (this.records) {
      this._states = Object.keys(this.records[0][0]);
    }
  }
  get layerNum() {
    return this.records[0][0][this.states[0]].length;
  }
  getRecords(stateName, layer = -1){
    if (this.records) {
      layer = layer === -1 ? this.layerNum - 1 : layer;
      return this.records.forEach((u) => {
        return u.forEach((v) => {
          return v[stateName][layer];
        });
      });
    }
    return undefined;
  }
}

// bus.$on('test', (message) => {
//   console.log('1:' + message);
// });

// bus.$on('test', (message) => {
//   console.log('2:' + message);
// });

// bus.$emit('test', 'haha');

export default bus;

export {
  bus,
  SELECT_MODEL,
  CoClusterProcessor,
  SentenceRecord,
}
