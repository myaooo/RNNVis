import Vue from 'vue'
import VueResource from 'vue-resource';
Vue.use(VueResource);

// Test version
// const devMainUrl = 'http://143.89.191.20';
const devMainUrl =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:5000'
    : location.origin;

const $http = Vue.http;

const cache = {}

const getUrlData = function(url, callback) {
  if (url in cache) {
      return Promise.resolve(callback(cache[url]));
  } else {
    return $http.get(url).then(response => {
      if (response.status === 200)
        cache[url] = {status: 200, data: response.data};
      callback(cache[url]);
    }, errResponse => {
      console.log(errResponse);
      throw errResponse;
    });
  }
}

let getProjectionData = function (model, state, parameters = {}, callback) {
  //  empty api for future implementation
  let url = `${devMainUrl}/projection?model=${model}&state=${state}`;
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return getUrlData(url, callback);
}

let getStrengthData = function (model, state, parameters = {}, callback) {
  // additional parameters: layer: -1, top_k: 100
  let url = `${devMainUrl}/strength?model=${model}&state=${state}`
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return getUrlData(url, callback);
}


let getStateSignature = function (model, state, parameters = {}, callback) {
  // additional parameters: layer: -1, size: 1000
  let url = `${devMainUrl}/state_signature?model=${model}&state=${state}`
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return getUrlData(url, callback);
}

let getModels = function (callback) {
  const url = `${devMainUrl}/models/available`;
  return getUrlData(url, callback);
}

let getModelConfig = function (model, callback) {
  const url = `${devMainUrl}/models/config/${model}`;
  return getUrlData(url, callback);
}

let getTextEvaluation = function (model, text, callback) {

  let url = `${devMainUrl}/models/evaluate`;
  // Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return $http.post(url, { model: model, text: text }).then(response => {
    callback(response);
  }, errResponse => {
    console.log(errResponse);
    throw errResponse;
  });
}

let getCoCluster = function (model, state, n_cluster, parameters = {}, callback) {
  // layer: -1
  // top_k: 100
  // mode: 'positive'
  // seed: 0
  let url = `${devMainUrl}/co_clusters?model=${model}&state=${state}&n_cluster=${n_cluster}`;
  Object.keys(parameters).forEach((p) => { url += `&${p}=${parameters[p]}`; });
  return getUrlData(url, callback);
}

let getVocab = function (model, top_k = 100, callback) {
  const url = `${devMainUrl}/vocab?model=${model}&top_k=${top_k}`;
  return getUrlData(url, callback);
}

// Get statistics of all states in a layer. The statistics are relating to words, e.g. reaction distribution
let getStateStatistics = function (model, state, layer, top_k, callback) {
  // k: k words with highest strength, and k words with lowest negative strength
  console.log(`If no statistics data available, try visit url ${devMainUrl}/models/record_default?model=${model}&set=test for generating state records!`);
  const url = `${devMainUrl}/state_statistics?model=${model}&state=${state}&layer=${layer}&top_k=${top_k}`;
  return getUrlData(url, callback);
}

// Get statistics of a word regarding all states in a layer.
let getWordStatistics = function (model, state, layer, word, callback) {
  const url = `${devMainUrl}/word_statistics?model=${model}&state=${state}&layer=${layer}&word=${word}`;
  return getUrlData(url, callback);
}

let getPosStatistics = function (model, top_k, callback) {
  const url = `${devMainUrl}/pos_statistics?model=${model}&top_k=${top_k}`;
  return getUrlData(url, callback);
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
  getPosStatistics,
}
