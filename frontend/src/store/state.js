// import dataService from '../service/dataService';
// import { isString } from '../service/utils';
// import RNNModel from './RNNModel';
import * as d3 from 'd3';

export function targetModel(state, compareOrModelName) {
  if (compareOrModelName in state.loadedModels) return state.loadedModels[compareOrModelName];
  return compareOrModelName ? state.selectedModel2 : state.selectedModel;
}

const state = {
  availableModels: {},
  modelList: [],
  loadedModels: {},
  selectedModel: null,
  selectedModel2: null,
  modelConfigs: {},
  coClusters: {},
  sentenceRecords: {},
  statistics: {},
  // modelsSet: null,
  selectedNode: null,
  selectedNode2: null,
  compare: false,
  isLoading: false,
  color: d3.scaleOrdinal(d3.schemeCategory10),
  // color: d3.scaleOrdinal(d3.schemeCategory10),
  // defaultLayout() {
  //   return {
  //     clusterNum: 5,
  //     strokeControlStrength: 50,
  //    //  strokeControlStrength: 8,
  //     linkFilterThreshold: [0, 1],
  //    //  linkFilterThreshold: [0.2, 1],
  //     stateClip: 1,
  //     mode: 'width',
  //   };
  // },
  // getAvailableModels() {
  //   // console.log(this.availableModels);
  //   if (this.availableModels === null) {
  //     console.log('Start loading model data');
  //     return dataService.getModels()
  //       .then(data => {
  //         this.availableModels = data.models;
  //         // this.modelsSet = new Set(this.availableModels);
  //         return Promise.resolve(this.availableModels);
  //       });
  //   }
  //   return Promise.resolve(this.availableModels);
  // },
  // getModelByName(modelName) { // return a Promise
  //   if (!isString(modelName)) return Promise.reject(modelName);
  //   if (!(modelName in this.loadedModels)) {
  //     return dataService.getModelConfig(modelName)
  //       .then(data => {
  //         const model = new RNNModel(modelName, data, this.defaultLayout());
  //         this.loadedModels[modelName] = model;
  //         return Promise.resolve(model);
  //       });
  //   }
  //   return Promise.resolve(this.loadedModels[modelName]);
  // },
  // getCoCluster(params, model = this.selectedModel) {
  //   if (model instanceof RNNModel) return model.getCoClusterProcessor(params);
  //   return this.getModelByName(model)
  //     .then(loadedModel => loadedModel.getCoClusterProcessor(params));
  // },
  // evalSentence(sentence, model = this.selectedModel) {
  //   if (model instanceof RNNModel) return model.getSentenceRecord(sentence);
  //   return this.getModelByName(model).then(loadedModel => loadedModel.getSentenceRecord(sentence));
  // },
  // getStatistics(topK = 500, model = this.selectedModel) {
  //   if (model instanceof RNNModel) return model.getStateProcessor(topK);
  //   return this.getModelByName(model).then(loadedModel => loadedModel.getStateProcessor(topK));
  // },
  // // load
  // getPosStatistics(topK = 300, model = this.selectedModel) {
  //   if (model instanceof RNNModel) return model.getPOSStatistics(topK);
  //   return this.getModelByName(model).then(loadedModel => loadedModel.getPOSStatistics(topK));
  // },
};

export default state;
