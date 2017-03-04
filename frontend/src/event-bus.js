import Vue from 'vue';
import dataService from './services/dataService';

export const SELECT_MODEL = 'SELECT_MODEL';

const state = {
  selectedModel: null,
  modelConfigs: {},
  availableModels: [],
}

export const bus = new Vue({
  data: {
    state: state,
  },
  computed: {
  },
  methods: {
    selectModel(modelName) {
      let state = this.state;
      const originSelectedModel = state.selectedModel;
      if (state.modelConfigs.hasOwnProperty(modelName)) {
        state.selectedModel = modelName;
      }
      else {
        dataService.getModelConfig(modelName, response => {
          if (response.status === 200) {
            state.modelConfigs[modelName] = response.data;
            state.selectedModel = modelName;
          }
        });
      }
      if (originSelectedModel !== state.selectedModel) {
        bus.$emit(SELECT_MODEL, state.selectedModel);
      }
    },

    loadModelConfig(modelName) { // return a Promise
      if (! state.modelConfigs.hasOwnProperty(modelName)) {
        return dataService.getModelConfig(modelName, response => {
          if (response.status === 200) {
            state.modelConfigs[modelName] = response.data;
            state.selectedModel = modelName;
          }
        });
      }
      return Promise.resolve(this.state.modelConfigs[modelName]);
    },

    loadAvailableModels() {
      // console.log(this.availableModels);
      if(this.state.availableModels.length === 0){
        return dataService.getModels(response => {
          if (response.status === 200) {
            const data = response.data;
            this.state.availableModels = data.models;
          }
          // console.log(this.availableModels);
        });
      }
      return Promise.resolve("Already Loaded");
    }
  }
});

// bus.$on(SELECT_MODEL, (modelName) => {

// })

// bus.$on('test', (message) => {
//   console.log('1:' + message);
// });

// bus.$on('test', (message) => {
//   console.log('2:' + message);
// });

// bus.$emit('test', 'haha');

export default bus;
