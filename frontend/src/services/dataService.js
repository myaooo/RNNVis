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

let getTextData = function (model, field) {
  return [
    [['i', 0.2], ['love', 0.4], ['you', 0.5], ['omg', 0.2], ['<eos>', 0.1]],
    [['i', 0.4], ['like', 0.2], ['you', 0.3], ['<eos>', 0.1], ['omg', 0.2]],
  ];
}

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

let getCoclusterData = function (model, state, parameters, callback) {
  let url = `${devMainUrl}/co_clusters?model=${model}&state=${state}`;
  Object.keys(parameters).forEach( (d) => { url += `&${d}=${parameters[d]}`})
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
  getTextData,
  getCoclusterData,
  getModels,
  getModelConfig
}
