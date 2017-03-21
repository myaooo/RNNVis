import Vue from 'vue';
import dataService from './services/dataService';
import { CoClusterProcessor, SentenceRecord, StateStatistics } from './preprocess'

// event definitions goes here
const SELECT_MODEL = 'SELECT_MODEL';
const SELECT_STATE = 'SELECT_STATE';
const CHANGE_LAYOUT = 'CHANGE_LAYOUT';
const EVALUATE_SENTENCE = 'EVALUATE_SENTENCE';
const SELECT_UNIT = 'SELECT_UNIT';
const SELECT_WORD = 'SELECT_WORD';
const DESELECT_UNIT = 'DESELECT_UNIT';
const DESELECT_WORD = 'DESELECT_WORD';
const SELECT_LAYER = 'SELECT_LAYER';

const state = {
  selectedModel: null,
  selectedState: null,
  selectedLayer: null,
  selectedModel2: null,
  selectedState2: null,
  selectedLayer2: null,
  layout: null,
  layout2: null,
  modelConfigs: {},
  coClusters: {},
  availableModels: null,
  sentenceRecords: {},
  statistics: {},
  modelsSet: null,
  selectedUnits: [],
  selectedWords: [],
  selectedUnits2: [],
  selectedWords2: [],
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

    loadModelConfig(modelName) { // return a Promise
      if (!modelName)
        return Promise.reject(modelName);
      if (!Object.prototype.hasOwnProperty.call(state.modelConfigs, modelName)) {
        return dataService.getModelConfig(modelName, response => {
          if (response.status === 200) {
            state.modelConfigs[modelName] = response.data;
            // state.selectedModel = modelName;
          }
        });
      }
      return Promise.resolve(modelName);
    },

    loadAvailableModels() {
      // console.log(this.availableModels);
      if (this.state.availableModels === null) {
        return dataService.getModels(response => {
          if (response.status === 200) {
            const data = response.data;
            this.state.availableModels = data.models;
            this.state.modelsSet = new Set(this.state.availableModels);
            // console.log(this.state.modelsSet);
          } else throw response;
        });
      }
      return Promise.resolve('Already Loaded');
    },

    loadCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 300, mode: 'raw', layer: -1 }) {
      const coCluster = new CoClusterProcessor(modelName, stateName, nCluster, params);
      const coClusterName = CoClusterProcessor.identifier(coCluster);
      if (this.state.coClusters.hasOwnProperty(coClusterName))
        return Promise.resolve('Cocluster data already loaded');
      return this.loadAvailableModels()
        .then(() => {
          if (this.state.modelsSet.has(modelName)) {
            return coCluster.load();
          }
          throw `No model named ${modelName}`;
        })
        .then(() => {
          this.state.coClusters[coClusterName] = coCluster;
          return 'Succeed';
        });
    },
    getCoCluster(modelName = this.state.selectedModel, stateName = this.state.selectedState, nCluster = 10, params = { top_k: 300, mode: 'raw', layer: -1 }) {
      const coCluster = new CoClusterProcessor(modelName, stateName, nCluster, params);
      const coClusterName = CoClusterProcessor.identifier(coCluster);
      if (this.state.coClusters.hasOwnProperty(coClusterName))
        return this.state.coClusters[coClusterName];
      console.log('First call loadCoCluster(...) to load remote Data!');
      return undefined;
    },
    // getModelConfig(modelName = state.selectedModel) {
    //   if (this.state.availableModels)
    //     return this.state.availableModels[modelName];
    //   return undefined;
    // },
    modelCellType(modelName = state.selectedModel) {
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)) {
        const config = this.state.modelConfigs[modelName];
        return config.model.cell_type;
      }
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
    },
    evalSentence(sentence, modelName = state.selectedModel) {
      if (!state.sentenceRecords.hasOwnProperty(modelName)) {
        state.sentenceRecords[modelName] = [];
      }
      const record = new SentenceRecord(sentence, modelName);
      state.sentenceRecords[modelName].push(record);
      return record;
    },
    loadStatistics(modelName = state.selectedModel, stateName = state.selectedState, layer = -1, top_k = 300) {
      if (!state.statistics.hasOwnProperty(modelName)) {
        state.statistics[modelName] = {};
      }
      if(!state.statistics[modelName].hasOwnProperty(stateName)) {
        state.statistics[modelName][stateName] = [];
      }
      if (layer === -1) {
        layer = this.layerNum(modelName) - 1;
      }
      if (state.statistics[modelName][stateName][layer]){
        return Promise.resolve('Already Loaded');
      }
      const stat = new StateStatistics(modelName, stateName, layer, top_k);
      state.statistics[modelName][stateName][layer] = stat;
      return stat.load();
    },
    getStatistics(modelName = state.selectedModel, stateName = state.selectedState, layer = -1, top_k = 300) {
      if (state.statistics.hasOwnProperty(modelName)) {
        if(state.statistics[modelName].hasOwnProperty(stateName)){
          if(state.statistics[modelName][stateName][layer])
            return state.statistics[modelName][stateName][layer];
        }
      }
      console.log(`bus > unable to get statistics for ${modelName}, ${stateName}, ${layer}`);
      return undefined;
    }
  },
  created() {
    // register event listener
    this.$on(SELECT_MODEL, (modelName, compare) => {
      if (compare)
        this.state.selectedModel2 = modelName;
      else
        this.state.selectedModel = modelName;
    });

    this.$on(SELECT_STATE, (stateName, compare) => {
      if (compare)
        this.state.selectedState2 = stateName;
      else
        this.state.selectedState = stateName;
    });

    this.$on(SELECT_LAYER, (layer, compare) => {
      if (compare)
        this.state.selectedLayer2 = layer;
      else
        this.state.selectedLayer = layer;
    });

// bus.$on(CLUSTER_NUM, (clusterNum) => {
//   bus.state.clusterNum = clusterNum;
// });

    this.$on(CHANGE_LAYOUT, (newLayout, compare) => {
      if(compare)
        this.state.layout2 = newLayout;
      else
        this.state.layout = newLayout;
      console.log(`bus > clusterNum: ${newLayout.clusterNum}`);
    });

    this.$on(EVALUATE_SENTENCE, (sentence, compare) => {
      console.log(`bus > evaluating model ${compare ? state.selectedModel2 : state.selectedModel} on sentence "${sentence}"`);
    });

    this.$on(SELECT_UNIT, (unitDim, compare) => {
      if (compare) {
        const units = this.state.selectedUnits2.slice();
        units.push(unitDim);
        if (units.length > 2)
          units.splice(0, 1);
        this.state.selectedUnits2 = units;
      } else {
        const units = this.state.selectedUnits.slice();
        units.push(unitDim);
        if (units.length > 2)
          units.splice(0, 1);
        this.state.selectedUnits = units;
      }
      console.log(`bus > selected unit ${unitDim}`);

    });

    this.$on(SELECT_WORD, (word, compare) => {
      if (compare) {
        const words = this.state.selectedWords2.slice();
        words.push(word);
        if (words.length > 3)
          words.splice(0, 1);
        this.state.selectedWords2 = words;
      } else {
        const words = this.state.selectedWords.slice();
        words.push(word);
        if (words.length > 3)
          words.splice(0, 1);
        this.state.selectedWords = words;
      }
      console.log(`bus > selected word: ${word}`);
    });

    this.$on(DESELECT_UNIT, (unit, compare) => {
      if (compare) {
        const idx = this.state.selectedUnits2.indexOf(unit);
        this.state.selectedUnits2.splice(idx, 1);
      } else {
        const idx = this.state.selectedUnits.indexOf(unit);
        this.state.selectedUnits.splice(idx, 1);
      }
      console.log(`bus > deselected unit: ${unit}`);
    });

    this.$on(DESELECT_WORD, (word, compare) => {
      if (compare) {
        const idx = this.state.selectedWords2.indexOf(word);
        this.state.selectedWords2.splice(idx, 1);
      } else {
        const idx = this.state.selectedWords.indexOf(word);
        this.state.selectedWords.splice(idx, 1);
      }
      console.log(`bus > deselected word: ${word}`);
    });

  }
});

export default bus;

export {
  bus,
  SELECT_MODEL,
  SELECT_STATE,
  CHANGE_LAYOUT,
  EVALUATE_SENTENCE,
  CoClusterProcessor,
  SentenceRecord,
  SELECT_UNIT,
  SELECT_WORD,
  SELECT_LAYER,
  DESELECT_UNIT,
  DESELECT_WORD,
}
