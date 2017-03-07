import Vue from 'vue';
import dataService from './services/dataService';

const SELECT_MODEL = 'SELECT_MODEL';

const state = {
  selectedModel: null,
  selectedState: null,
  modelConfigs: {},
  availableModels: null,
};

const bus = new Vue({
  data: {
    state: state,
    cell2states: {
      'GRU': ['state'],
      'BasicLSTM': ['state_c', 'state_h'],
      'BasicRNN': ['state'],
    },
  },
  computed: {
  },
  methods: {
    // selectModel(modelName) {
    //   // const state = this.state;
    //   const originSelectedModel = state.selectedModel;
    //   if (Object.prototype.hasOwnProperty.call(state.modelConfigs, modelName)) {
    //     state.selectedModel = modelName;
    //   } else {
    //     dataService.getModelConfig(modelName, response => {
    //       if (response.status === 200) {
    //         state.modelConfigs[modelName] = response.data;
    //         state.selectedModel = modelName;
    //       }
    //     });
    //   }
    //   if (originSelectedModel !== state.selectedModel) {
    //     bus.$emit(SELECT_MODEL, state.selectedModel);
    //   }
    // },

    loadModelConfig(modelName) { // return a Promise
      if (!Object.prototype.hasOwnProperty.call(state.modelConfigs, modelName)) {
        return dataService.getModelConfig(modelName, response => {
          if (response.status === 200) {
            state.modelConfigs[modelName] = response.data;
            // state.selectedModel = modelName;
          }
        });
      }
      return Promise.resolve(this.state.modelConfigs[modelName]);
    },

    loadAvailableModels() {
      // console.log(this.availableModels);
      if (this.state.availableModels === null) {
        return dataService.getModels(response => {
          if (response.status === 200) {
            const data = response.data;
            this.state.availableModels = data.models;
          }
          // console.log(this.availableModels);
        });
      }
      return Promise.resolve('Already Loaded');
    },
    availableStates(modelName) { // helper function that returns available states of the current selected Model`
      modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)){
        const config = this.state.modelConfigs[modelName];
        return this.cell2states[config.model.cell_type];
      }
      return undefined;
    },
    layerNum(modelName) {
      modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)){
        const config = this.state.modelConfigs[modelName];
        return config.model.cells.length;
      }
      return undefined;
    },
    layerSize(modelName, layer){
      modelName = modelName || this.state.selectedModel;
      if (Object.prototype.hasOwnProperty.call(this.state.modelConfigs, modelName)){
        if (!layer || layer === -1){
          layer = this.layerNum(modelName) - 1;
        }
        const config = this.state.modelConfigs[modelName];
        return config.model.cells[layer].num_units;
      }
      return undefined;
    }
  },
});

bus.$on(SELECT_MODEL, (modelName) => {
  bus.state.selectedModel = modelName;
});

// bus.$on('test', (message) => {
//   console.log('1:' + message);
// });

// bus.$on('test', (message) => {
//   console.log('2:' + message);
// });

// bus.$emit('test', 'haha');

export default bus;

export {
  bus,
  SELECT_MODEL,
}
