import Vue from 'vue';
import VueResource from 'vue-resource';

Vue.use(VueResource);

// Test version
// const mainPath = 'http://143.89.191.20';
const mainPath = 'http://localhost:5000';

const $http = Vue.http;

const cache = {};

function getUrlData(url) {
  if (url in cache) {
    return Promise.resolve(cache[url].data);
  }
  return $http.get(url).then(response => {
    if (response.status === 200) {
      cache[url] = {
        status: 200,
        data: response.data,
      };
      return cache[url].data;
    }
    throw response.status;
  }, errResponse => {
    throw errResponse;
  });
}

function getProjectionData(model, state, parameters = {}) {
  //  empty api for future implementation
  let url = `${mainPath}/projection?model=${model}&state=${state}`;
  Object.keys(parameters).forEach((p) => {
    url += `&${p}=${parameters[p]}`;
  });
  return getUrlData(url);
}

function getStrengthData(model, state, parameters = {}) {
  // additional parameters: layer: -1, top_k: 100
  let url = `${mainPath}/strength?model=${model}&state=${state}`;
  Object.keys(parameters).forEach((p) => {
    url += `&${p}=${parameters[p]}`;
  });
  return getUrlData(url);
}


function getStateSignature(model, state, parameters = {}) {
  // additional parameters: layer: -1, size: 1000
  let url = `${mainPath}/state_signature?model=${model}&state=${state}`;
  Object.keys(parameters).forEach((p) => {
    url += `&${p}=${parameters[p]}`;
  });
  return getUrlData(url);
}

function getModels() {
  const url = `${mainPath}/models/available`;
  return getUrlData(url);
}

function getModelConfig(model) {
  const url = `${mainPath}/models/config/${model}`;
  return getUrlData(url);
}

function getTextEvaluation(model, text) {
  const url = `${mainPath}/models/evaluate`;
  // Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.post(url, {
    model,
    text,
  }).then(response => {
    if (response.status === 200) {
      return response.data;
    }
    throw response.status;
  }, errResponse => {
    throw errResponse;
  });
}

function getCoCluster(model, state, layer, nCluster = 10, parameters = {}) {
  // topK: 500
  // mode: 'raw'
  // seed: 0
  const acceptableParameters = new Set(['topK', 'mode', 'seed']);
  let url = `${mainPath}/co_clusters?model=${model}&state=${state}&layer=${layer}&nCluster=${nCluster}`;
  Object.keys(parameters).forEach((p) => {
    if (p in acceptableParameters) {
      url += `&${p}=${parameters[p]}`;
    }
  });
  return getUrlData(url);
}

function getVocab(model, topK = 100) {
  const url = `${mainPath}/vocab?model=${model}&top_k=${topK}`;
  return getUrlData(url);
}

// Get statistics of all states in a layer.
// The statistics are relating to words, e.g. reaction distribution
function getStateStatistics(model, state, layer, topK) {
  // k: k words with highest strength, and k words with lowest negative strength
  // console.log(`If no statistics data available, try visit url:
  // ${mainPath}/models/record_default?model=${model}&set=test for generating state records!`);
  const url = `${mainPath}/state_statistics?model=${model}&state=${state}&layer=${layer}&top_k=${topK}`;
  return getUrlData(url);
}

// Get statistics of a word regarding all states in a layer.
function getWordStatistics(model, state, layer, word) {
  const url = `${mainPath}/word_statistics?model=${model}&state=${state}&layer=${layer}&word=${word}`;
  return getUrlData(url);
}

function getPOSStatistics(model, topK) {
  const url = `${mainPath}/pos_statistics?model=${model}&top_k=${topK}`;
  return getUrlData(url);
}

export default {
  getProjectionData,
  getStrengthData,
  // getTextData,
  getModels,
  getModelConfig,
  getTextEvaluation,
  getCoCluster,
  getVocab,
  getStateSignature,
  getStateStatistics,
  getWordStatistics,
  getPOSStatistics,
};

