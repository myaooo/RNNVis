import Vue from 'vue'
import VueResource from 'vue-resource';
Vue.use(VueResource);

// Test version
const devMainUrl = 'http://localhost:5000';

const $http = Vue.http;

let getProjectionData = function (model, state, parameters = {}, callback) {
  //  empty api for future implementation
  let url = `${devMainUrl}/projection?model=${model}&state=${state}`;
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getStrengthData = function (model, state, parameters = {}, callback) {
  // additional parameters: layer: -1, top_k: 100
  let url = `${devMainUrl}/strength?model=${model}&state=${state}`
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getStateSignature = function (model, state, parameters = {}, callback) {
  // additional parameters: layer: -1, size: 1000
  let url = `${devMainUrl}/state_signature?model=${model}&state=${state}`
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

// let getTextData = function (model, field) {
//   return [
//     [['i', 0.2], ['love', 0.4], ['you', 0.5], ['omg', 0.2], ['<eos>', 0.1]],
//     [['i', 0.4], ['like', 0.2], ['you', 0.3], ['<eos>', 0.1], ['omg', 0.2]],
//   ];
// }

let getModels = function (callback) {
  const url = `${devMainUrl}/models/available`;
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getModelConfig = function (model, callback) {
  const url = `${devMainUrl}/models/config/${model}`;
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getTextEvaluation = function (model, state, layer, text, callback) {
  // layer: -1
  layer = layer || -1;
  let url = `${devMainUrl}/models/evaluate`;
  // Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.post(url, { model: model, state: state, layer: layer, text: text }).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getCoCluster = function (model, state, n_cluster, params = {}, callback) {
  // layer: -1
  // layer = layer || -1;
  let url = `${devMainUrl}/co_clusters?model=${model}&state=${state}&n_cluster=${n_cluster}`;
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getVocab = function (model, top_k = 100, callback) {
  const url = `${devMainUrl}/vocab?model=${model}&top_k=${top_k}`;
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

// Get statistics of all states in a layer. The statistics are relating to words, e.g. reaction distribution
let getStateStatistics = function (model, state, layer, top_k, callback) {
  // k: k words with highest strength, and k words with lowest negative strength
  const url = `${devMainUrl}/state_statistics?model=${model}&state=${state}&layer=${layer}&top_k={top_k}`;
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

// Get statistics of a word regarding all states in a layer.
let getWordStatistics = function (model, state, layer, word, callback) {
  const url = `${devMainUrl}/word_statistics?model=${model}&state=${state}&layer=${layer}&word=${word}`;
  return $http.get(url).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

export default {
  getProjectionData,
  getStrengthData,
  getStateSignature,
  // getTextData,
  getModels,
  getModelConfig,
  getTextEvaluation,
  getCoCluster,
  getVocab,
  getStateStatistics,
  getWordStatistics,
}
