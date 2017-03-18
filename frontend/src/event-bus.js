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
    loadCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 30, mode: 'positive' }) {
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
    getCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 30, mode: 'positive' }) {
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
    this._rowClusters;
    this._colClusters;
    this._aggregation_info;
    this.modelName = modelName;
    this.stateName = stateName;
    this.nCluster = nCluster;
    this.params = params;
  }
  get correlation() {
    return this.hasData ? this.rawData.data : undefined;
  }
  get labels() {
    if (this.hasData) {
      this._labels = [...new Set(this.rowLabels)];
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

  strength_filter(strength, mode=this.params.mode) {
    let strength_item = 0;
    switch(mode) {
      case 'positive':
        strength_item = strength > 0 ? strength : 0;
        break;
      case 'negative':
        strength_item = strength < 0 ? Math.abs(strength) : 0;
        break;
      case 'abs':
        strength_item = Math.abs(strength);
        break;
      case 'raw':
        strength_item = strength;
        break;
    }
    return strength_item;
  }

  Create2DArray(rowNum, colNum) {
    return Array.from({ length: rowNum }, (v, i) => {
      return Array.from({ length: colNum }, (v, i) => 0);
    });
  }
  aggregation_info() {
    if (this.hasData) {
      let rowClusters = this.rowClusters;
      let colClusters = this.colClusters;
      let row_cluster_2_col_cluster = this.Create2DArray(this.nCluster, this.nCluster);
      let row_single_2_col_cluster = this.Create2DArray(this.rawData.row.length, this.nCluster);
      let row_cluster_2_col_single = this.Create2DArray(this.nCluster, this.rawData.col.length);
      let row_single_2_col_single = this.Create2DArray(this.rawData.row.length, this.rawData.col.length);
      let cluster = [];
      // calculate the correlation between clusters
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_cluster_2_col_cluster[this.rawData.row[r_index]][this.rawData.col[c_index]] += strength;
          row_cluster_2_col_single[this.rawData.row[r_index]][c_index] += strength;
          row_single_2_col_cluster[r_index][this.rawData.col[c_index]] += strength;
          row_single_2_col_single[r_index][c_index] += strength;
        });
      });

      return {row_cluster_2_col_cluster: row_cluster_2_col_cluster,
              row_single_2_col_cluster: row_single_2_col_cluster,
              row_cluster_2_col_single: row_cluster_2_col_single,
              row_single_2_col_single: row_single_2_col_single};
    }
    return null;
  }

  get rowClusters() {
    if (this.hasData) {
      this._rowClusters = [];
      this.rawData.row.forEach((r, i) => {
        if (this._rowClusters[r] === undefined) {
          this._rowClusters[r] = [];
        }
        this._rowClusters[r].push(i);
      });
      return this._rowClusters;
    }
    return this._rowClusters;
  }
  get colClusters() {
    if (this.hasData) {
      this._colClusters = [];
      this.rawData.col.forEach((c, i) => {
        if (this._colClusters[c] === undefined) {
          this._colClusters[c] = [];
        }
        this._colClusters[c].push(i);
      });
      return this._colClusters;
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
