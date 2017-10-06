import Vue from 'vue';
import Vuex from 'vuex';

import state from './state';
import getters from './getters';
import mutations from './mutations';
import actions from './actions';

Vue.use(Vuex);

const store = new Vuex.Store({
  state,
  getters,
  mutations,
  actions,
});

export default store;

export {
  GET_MODEL_LIST,
  // export const SET_MODEL_LIST = 'SET_MODEL_LIST';
  GET_MODEL,
  GET_COCLUSTER_DATA,
  GET_SENTENCE_EVALUATION,
  GET_STATE_STATISTICS,
  GET_POS_STATISTICS,
  // export const SET_MODEL = 'SET_MODEL';
  SELECT_MODEL,
  SELECT_STATE,
  SELECT_UNIT,
  SELECT_WORD,
  SELECT_LAYER,
  SELECT_SENTENCE_NODE,
  UPDATE_STYLE,
  EVALUATE_SENTENCE,
  // DESELECT_UNIT,
  // DESELECT_WORD,
  CLOSE_SENTENCE,
  RENDER_GRAPH,
  GRAPH_RENDERED,
  COMPARE,
} from './types';
