import {
  LOADING,
  LOADED,
  GET_MODEL_LIST,
  GET_MODEL,
  GET_COCLUSTER_DATA,
  GET_SENTENCE_EVALUATION,
  GET_STATE_STATISTICS,
  GET_POS_STATISTICS,
  GET_WORD_STATISTICS,
  SELECT_MODEL,
  UPDATE_STYLE,
  // SELECT_LAYER,
  // SELECT_STATE,
  SELECT_UNIT,
  SELECT_WORD,
  RENDER_GRAPH,
  GRAPH_RENDERED,
  COMPARE,
  // SELECT_SENTENCE_NODE,
} from './types';

// import { targetModel } from './state';

import {
  isString,
} from '../service/utils';

const mutations = {

  [LOADING](state) {
    state.isLoading = true;
  },

  [LOADED](state) {
    state.isLoading = false;
  },

  [GET_MODEL_LIST](state, {
    modelList,
  }) {
    if (modelList) {
      state.modelList = modelList;
      state.availableModels = new Set(modelList);
    }
  },

  [GET_MODEL](state, { modelName, model }) {
    state.loadedModels[modelName] = model;
  },

  [GET_COCLUSTER_DATA](state, playload) {
    const { data, modelName, stateName, layer, nCluster } = playload;
    // console.log('CoCluster data loaded.');
    if (data && modelName) {
      const model = state.loadedModels[modelName];
      model.coCluster = data;
      model.selectedState = stateName;
      model.selectedLayer = layer;
      model.nCluster = nCluster;
      // the co cluster data need to be re-rendered.
    }
  },

  [GET_SENTENCE_EVALUATION](state, playload) {
    const { data, modelName } = playload;
    if (data && modelName) {
      state.loadedModels[modelName].sentences.push(data);
    }
  },

  [GET_STATE_STATISTICS](state, playload) {
    const { data, modelName } = playload;
    if (data && modelName) {
      state.loadedModels[modelName].stateStats = data;
    }
  },

  [GET_POS_STATISTICS](state, playload) {
    const { data, modelName } = playload;
    if (data && modelName) {
      state.loadedModels[modelName].POSStats = data;
    }
  },

  [GET_WORD_STATISTICS](state, playload) {
    const { data, modelName } = playload;
    if (data && modelName) {
      // console.log(data);
      state.loadedModels[modelName].wordStats = data;
    }
  },

  [UPDATE_STYLE](state, playload) {
    const { modelName, style } = playload;
    if (style && modelName) {
      // get a deep copy
      state.loadedModels[modelName].style = JSON.parse(JSON.stringify(style));
    }
  },

  [SELECT_MODEL](state, {
    modelName,
    compare,
  }) {
    if (isString(modelName)) {
      if (compare) {
        state.selectedModel2 = state.loadedModels[modelName];
      } else {
        state.selectedModel = state.loadedModels[modelName];
      }
    }
  },

  // [SELECT_STATE](state, {
  //   stateName,
  //   compare,
  // }) {
  //   if (isString(stateName)) {
  //     const model = targetModel(state, compare);
  //     if (stateName in model.availableStates) {
  //       model.selectedState = stateName;
  //     }
  //   }
  // },

  // [SELECT_LAYER](state, {
  //   layer,
  //   compare,
  // }) {
  //   const model = targetModel(state, compare);
  //   model.selectLayer(layer);
  // },

  [SELECT_UNIT](state, {
    unit,
    modelName,
  }) {
    state.loadedModels[modelName].selectUnit(unit);
  },

  [SELECT_WORD](state, {
    word,
    modelName,
  }) {
    state.loadedModels[modelName].selectWord(word);
    // console.log(word);
    // console.log(`word ${word} selected!`);
  },

  [RENDER_GRAPH](state, { modelName }) {
    state.loadedModels[modelName].needRendering = true;
  },

  [GRAPH_RENDERED](state, { modelName }) {
    state.loadedModels[modelName].needRendering = false;
  },

  [COMPARE](state) {
    state.compare = !state.compare;
  },

};

export default mutations;

