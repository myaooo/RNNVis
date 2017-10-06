import Vue from 'vue';
import * as d3 from 'd3';
import dataService from '../service/dataService';
import {
  // has,
  isString,
} from '../service/utils';
import RNNModel from './RNNModel';

// event definitions goes here
export const SELECT_MODEL = 'SELECT_MODEL';
export const SELECT_STATE = 'SELECT_STATE';
export const CHANGE_LAYOUT = 'CHANGE_LAYOUT';
export const EVALUATE_SENTENCE = 'EVALUATE_SENTENCE';
export const SELECT_UNIT = 'SELECT_UNIT';
export const SELECT_WORD = 'SELECT_WORD';
export const DESELECT_UNIT = 'DESELECT_UNIT';
export const DESELECT_WORD = 'DESELECT_WORD';
export const SELECT_LAYER = 'SELECT_LAYER';
export const CLOSE_SENTENCE = 'CLOSE_SENTENCE';
export const SELECT_SENTENCE_NODE = 'SELECT_SENTENCE_NODE';
// const SELECT_COLOR = 'SELECT_COLOR';

export const state = {
  availableModels: null,
  loadedModels: {},
  selectedModel: null,
  selectedState2: null,
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
  defaultLayout() {
    return {
      clusterNum: 5,
      strokeControlStrength: 50,
     //  strokeControlStrength: 8,
      linkFilterThreshold: [0, 1],
     //  linkFilterThreshold: [0.2, 1],
      stateClip: 1,
      mode: 'width',
    };
  },
  getAvailableModels() {
    // console.log(this.availableModels);
    if (this.availableModels === null) {
      console.log('Start loading model data');
      return dataService.getModels()
        .then(data => {
          this.availableModels = data.models;
          // this.modelsSet = new Set(this.availableModels);
          return Promise.resolve(this.availableModels);
        });
    }
    return Promise.resolve(this.availableModels);
  },
  getModelByName(modelName) { // return a Promise
    if (!isString(modelName)) return Promise.reject(modelName);
    if (!(modelName in this.loadedModels)) {
      return dataService.getModelConfig(modelName)
        .then(data => {
          const model = new RNNModel(modelName, data, this.defaultLayout());
          this.loadedModels[modelName] = model;
          return Promise.resolve(model);
        });
    }
    return Promise.resolve(this.loadedModels[modelName]);
  },
  getCoCluster(params, model = this.selectedModel) {
    if (model instanceof RNNModel) return model.getCoClusterProcessor(params);
    return this.getModelByName(model)
      .then(loadedModel => loadedModel.getCoClusterProcessor(params));
  },
  evalSentence(sentence, model = this.selectedModel) {
    if (model instanceof RNNModel) return model.getSentenceRecord(sentence);
    return this.getModelByName(model).then(loadedModel => loadedModel.getSentenceRecord(sentence));
  },
  getStatistics(topK = 500, model = this.selectedModel) {
    if (model instanceof RNNModel) return model.getStateProcessor(topK);
    return this.getModelByName(model).then(loadedModel => loadedModel.getStateProcessor(topK));
  },
  // load
  getPosStatistics(topK = 300, model = this.selectedModel) {
    if (model instanceof RNNModel) return model.getPOSStatistics(topK);
    return this.getModelByName(model).then(loadedModel => loadedModel.getPOSStatistics(topK));
  },
};

export const bus = new Vue({
  data: {
    state,
  },
  created() {
    // register event listener
    this.$on(SELECT_MODEL, (modelName, compare) => {
      if (compare) {
        this.state.selectedModel2 = modelName;
        this.state.compare = Boolean(modelName);
      } else {
        this.state.selectedModel = modelName;
      }
      console.log(`bus > current models : ${state.selectedModel}, ${state.selectedModel2}`);
    });

    this.$on(SELECT_STATE, (stateName, compare) => {
      if (compare) {
        this.state.selectedState2 = stateName;
      } else {
        this.state.selectedState = stateName;
      }
      console.log(`bus > current states : ${state.selectedState}, ${state.selectedState2}`);
    });

    this.$on(SELECT_LAYER, (layer, compare) => {
      if (compare) {
        this.state.selectedLayer2 = layer;
      } else {
        this.state.selectedLayer = layer;
      }
    });

    this.$on(CHANGE_LAYOUT, (newLayout, compare) => {
      if (compare) {
        this.state.layout2 = Object.assign({}, newLayout);
      } else {
        this.state.layout = Object.assign({}, newLayout);
      }
      console.log(`bus > ${compare ? 'compare' : ''} clusterNum: ${newLayout.clusterNum}`);
    });

    this.$on(EVALUATE_SENTENCE, (sentence, compare) => {
      console.log(`bus > evaluating model ${compare ? state.selectedModel2 : state.selectedModel} on sentence "${sentence}"`);
    });

    this.$on(SELECT_UNIT, (unitDim, compare) => {
      if (compare) {
        const units = this.state.selectedUnits2.slice();
        units.push(unitDim);
        if (units.length > 2) {
          units.splice(0, 1);
        }
        this.state.selectedUnits2 = units;
        this.state.selectedWords2 = [];
        this.state.selectedNode2 = null;
      } else {
        const units = this.state.selectedUnits.slice();
        units.push(unitDim);
        if (units.length > 2) {
          units.splice(0, 1);
        }
        this.state.selectedUnits = units;
        if (this.state.compare) {
          this.state.selectedWords = [];
          this.state.selectedNode = null;
        }
      }
      console.log(`bus > selected unit ${unitDim}`);
    });

    // this.$on(SELECT_WORD, (word, compare) => {
    //   const maxSelected = 3;
    //   let words;
    //   if (compare) {
    //     words = this.state.selectedWords2.slice();
    //     this.state.selectedUnits2 = [];
    //     this.state.selectedNode2 = null;
    //   } else {
    //     words = this.state.selectedWords.slice();
    //     if (this.state.compare) {
    //       this.state.selectedUnits = [];
    //       this.state.selectedNode = null;
    //     }
    //   }
    //   words.push(word);
    //   if (words.length > maxSelected) {
    //     deactivateText(words[0]);
    //     words.splice(0, 1);
    //   }
    //   afterChangeWords(words);
    //   if (compare) this.state.selectedWords2 = words;
    //   else this.state.selectedWords = words;
    //   console.log(`bus > selected word: ${word.text}`);
    // });

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

    // this.$on(DESELECT_WORD, (word, compare) => {
    //   let words;
    //   if (compare) words = this.state.selectedWords2.slice();
    //   else words = this.state.selectedWords.slice();
    //   const idx = words.findIndex((d) => d.text === word.text);
    //   console.log(`bus > deleted idx: ${idx}`);
    //   deactivateText(words[idx]);
    //   words.splice(idx, 1);
    //   afterChangeWords(words);
    //   if (compare) this.state.selectedWords2 = words;
    //   else this.state.selectedWords = words;
    //   console.log(`bus > deselected word: ${word.text}`);
    // });

    this.$on(CLOSE_SENTENCE, (sentence) => {
      console.log(`bus > close sentence: ${sentence}`);
    });
    this.$on(SELECT_SENTENCE_NODE, (node, compare) => {
      if (compare) {
        this.state.selectedNode2 = node;
      } else {
        this.state.selectedNode = node;
      }
      console.log(`bus > sentence node selected: ${node.word}`);
    });
  },
});

// function afterChangeWords(words) {
//   words.forEach((word, i) => {
//     word.color = state.color(i);
//     activateText(word);
//     focusText(word);
//   });
// }

// function deactivateText(data) {
//   d3.select(data.el)
//     .style('fill-opacity', data.opacity)
//     .style('font-weight', data.weight)
//     .style('stroke', 'none')
//     .style('fill', data.baseColor);
//   if (data.bound) {
//     data.bound.remove();
//     data.bound = null;
//   }
// }

// function activateText(data) {
//   d3.select(data.el).style('fill-opacity', 1)
//     .style('font-weight', data.weight + 300)
//     .style('stroke', 'none')
//     .style('fill', data.color);
//   if (data.bound) {
//     data.bound.remove();
//     data.bound = null;
//   }
// }

// function focusText(data) {
//   const box = data.el.getBBox();
//   data.bound = d3.select(data.el.parentNode).insert('path')
//     .attr('d', 'M ' + (((data.x - box.width) / 2) - 1.5) + ' ' + (data.y + 2) +
//       ' H ' + (((data.x + box.width) / 2) + 1.5))
//     .style('stroke', data.color);
// }

export default {
  bus,
  state,
  SELECT_MODEL,
  SELECT_STATE,
  CHANGE_LAYOUT,
  EVALUATE_SENTENCE,
  SELECT_UNIT,
  SELECT_WORD,
  SELECT_LAYER,
  DESELECT_UNIT,
  DESELECT_WORD,
  CLOSE_SENTENCE,
  SELECT_SENTENCE_NODE,
};

