import dataService from '../service/dataService';
import {
  CoClusterProcessor,
  StateProcessor,
  SentenceRecord,
} from '../service/preprocess';

import {
  GET_MODEL_LIST,
  GET_MODEL,
  GET_COCLUSTER_DATA,
  GET_SENTENCE_EVALUATION,
  GET_STATE_STATISTICS,
  GET_POS_STATISTICS,
  GET_WORD_STATISTICS,
  SELECT_MODEL,
  SELECT_UNIT,
  SELECT_WORD,
  // CHANGE_LAYOUT,
  LOADING,
  LOADED,
} from './types';

import { targetModel } from './state';

import RNNModel from './RNNModel';

function fetchDataWrapper(commit, mutationType, fetch, ...args) {
  // This is a wrapper function that fetches the data using `fetch(...args)`
  // and then `commit` a mutation with type `mutationType`
  // using the fetched data as the playload of the mutation.
  // This function will return a Promise of the fetched data if resolved.
  commit(LOADING);
  return fetch(...args).then(data => {
    commit(mutationType, data);
    commit(LOADED);
    return Promise.resolve(data);
  });
}

function memorizeFetch(fetch) {
  const cache = {};
  return (...args) => {
    const key = args.length + Array.prototype.join.call(args, ',');
    if (!(key in cache)) {
      return fetch(...args).then(value => {
        cache[key] = value;
        return Promise.resolve(value);
      });
    }
    return Promise.resolve(cache[key]);
  };
}

const getCoClusterProcessor = (modelName, stateName, layer, nCluster, params) => {
  // eslint-disable-next-line
  console.log('Actions > start loading co-cluster data');
  return dataService.getCoCluster(modelName, stateName, layer, nCluster, params)
    .then(data => ({
      data: new CoClusterProcessor({
        data,
        params,
      }),
      modelName,
      stateName,
      layer,
      nCluster,
    }));
};

const getStateProcessor = memorizeFetch((modelName, stateName, layer, topK) =>
  dataService.getStateStatistics(modelName, stateName, layer, topK)
    .then(data => ({
      data: new StateProcessor(data),
      modelName,
    })));

const getSentenceRecord = memorizeFetch((modelName, sentence) =>
  dataService.getTextEvaluation(modelName, sentence)
    .then(data => ({
      data: new SentenceRecord(data),
      modelName,
    })));

const getPOSStatistics = (modelName, topK) =>
  dataService.getPOSStatistics(modelName, topK)
    .then(data => ({
      data,
      modelName,
    }));

const getWordStatistics = (modelName, stateName, layer, word) =>
  dataService.getWordStatistics(modelName, stateName, layer, word)
    .then(data => ({
      data,
      modelName,
    }));

const actions = {
  [GET_MODEL_LIST]({ commit }) {
    return dataService.getModels()
      .then(data => {
        const modelList = data.models;
        commit(GET_MODEL_LIST, { modelList });
      });
  },

  [SELECT_MODEL]({ dispatch, commit, state }, playload) {
    // mark the selected model, get model data if the model is not loaded
    const { modelName } = playload;
    if (modelName in state.loadedModels) {
      commit(SELECT_MODEL, playload);
      return Promise.resolve(state.selectedModel);
    }
    return dispatch(GET_MODEL, playload).then(() => {
      commit(SELECT_MODEL, playload);
      return Promise.resolve(state.selectedModel);
    });
  },

  [GET_MODEL]({ commit }, { modelName }) {
    return dataService.getModelConfig(modelName)
      .then(data => {
        const model = new RNNModel(modelName, data, RNNModel.defaultStyle());
        commit(GET_MODEL, { modelName, model });
      });
  },

  [GET_COCLUSTER_DATA]({ commit, state }, playload) {
    // legality check
    const { modelName, stateName, layer, nCluster, params } = playload;
    // the modelName has to be valid
    if (!(modelName in state.loadedModels)) return Promise.resolve();
    const model = state.loadedModels[modelName];
    // the stateName, layerName, nCluster has to be valid
    if (!model.isLegalState(stateName) || !model.isLegalLayer(layer) || nCluster < 1) return Promise.resolve();
    // check if the playload has already match the current status of the model
    if (model.selectedState === stateName && model.selectedLayer === layer && model.nCluster === nCluster) {
      return Promise.resolve();
    }
    return fetchDataWrapper(commit, GET_COCLUSTER_DATA, getCoClusterProcessor,
      modelName, stateName, layer, nCluster, params);
  },

  [GET_SENTENCE_EVALUATION]({ commit, state }, playload) {
    const { sentence, compare } = playload;
    const model = targetModel(state, compare);
    return fetchDataWrapper(commit, GET_SENTENCE_EVALUATION, getSentenceRecord, model.name, sentence);
  },

  [GET_STATE_STATISTICS]({ commit, state }, { modelName }) {
    // const { topK, compare } = playload;
    // const model = targetModel(state, compare);
    const model = state.loadedModels[modelName];
    if (!model) return Promise.reject(`No model with name ${modelName} loaded!`);
    return fetchDataWrapper(commit, GET_STATE_STATISTICS, getStateProcessor,
      model.name, model.selectedState, model.selectedLayer, model.topK);
  },

  [GET_POS_STATISTICS]({ commit, state }, { modelName }) {
    // const { topK, compare } = playload;
    const model = state.loadedModels[modelName];
    if (!model) return Promise.reject(`No model with name ${modelName} loaded!`);
    // const model = targetModel(state, compare);
    return fetchDataWrapper(commit, GET_POS_STATISTICS, getPOSStatistics,
      model.name, model.topK);
  },

  [GET_WORD_STATISTICS]({ commit, state }, { modelName, word }) {
    const model = state.loadedModels[modelName];
    if (!model) return Promise.reject(`No model with name ${modelName} loaded!`);
    return fetchDataWrapper(commit, GET_WORD_STATISTICS, getWordStatistics,
      model.name, model.selectedState, model.selectedLayer, word);
  },

  [SELECT_UNIT]({ commit, state }, playload) {
    const { compare, unit } = playload;
    const modelName = targetModel(state, compare).name;
    commit(SELECT_UNIT, { unit, modelName });
  },

  [SELECT_WORD]({ commit, state }, playload) {
    const { compare, word } = playload;
    const modelName = targetModel(state, compare).name;
    commit(SELECT_WORD, { word, modelName });
  },
};

export default actions;
