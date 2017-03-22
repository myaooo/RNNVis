<template>
  <div>
    <h4 class="normal">Models</h4>
    <hr>
    <div class="content model-view">
      <model-config> </model-config>
      <hr v-if="compare">
      <model-config v-if="compare" :compare="true" :toggle="toggleCompare"> </model-config>
      <hr>
      <el-button @click="toggleCompare" size="small">Compare Model</el-button>
    </div>
  </div>
</template>
<style scoped>
  .model-view {
    text-align: left;
    /*padding: 5px;*/
  }

  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
<script>
  import { bus, SELECT_MODEL, SELECT_STATE } from '../event-bus.js';
  import ModelConfig from './ModelConfig';

  export default{
    name: 'ModelView',
    components: { ModelConfig },
    data() {
      return {
        compare: false,
        shared: bus.state,
      };
    },
    mounted() {
      this.getModels();
    },
    computed: {
      availableModels: function() { // model list
        return this.shared.availableModels;
      },
    },
    methods: {
      getModels() {
        bus.loadAvailableModels();
      },
      modelColor(i) {
        return i === 1 ? 'primary' : 'success';
      },
      toggleCompare() {
        this.compare = !this.compare;
        if (!this.compare) {
          this.selectedModel2 = null;
          this.selectedState2 = null;
          this.states2 = [];
          bus.$emit(SELECT_MODEL, null, true);
          bus.$emit(SELECT_STATE, null, true);

        }
      },
      stateName(state) {
        switch(state) {
          case 'state_c': return 'c_state';
          case 'state_h': return 'h_state';
          case 'state': return 'h_state';
          default: return 'Unknown';
        }
      }
    },
    watch: {
      // selectedModel: function(selectedModel){
      //   bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
      //     .then(() => {
      //       const states = bus.availableStates(selectedModel);
      //       if (states){
      //         this.states = states;
      //         this.selectedState = null; // reset
      //         const config = bus.state.modelConfigs[selectedModel];
      //         this.config = {
      //           Cell: config.model.cell_type,
      //           LayerNum: config.model.cells.length,
      //           LayerSize: config.model.cells[0].num_units,
      //         };
      //       // this.configTree = json2tree(config).children;
      //         bus.$emit(SELECT_MODEL, this.selectedModel, this.selectedModel2);
      //       }
      //     });
      // },
      // selectedModels: function(selectedModels){
      //   if (!selectedModel) return;
      //   bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
      //     .then(() => {
      //       const states = bus.availableStates(selectedModel);
      //       if (states){
      //         this.states2 = states;
      //         this.selectedState2 = null; // reset
      //       // const config = bus.state.modelConfigs[selectedModel];
      //       // this.configTree = json2tree(config).children;
      //         bus.$emit(SELECT_MODEL, this.selectedModel, this.selectedModel2);
      //       }
      //     });
      // },
      // selectedState: function (newState) {
      //   if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
      //     bus.$emit(SELECT_STATE, this.selectedState, this.selectedState2);
      //   }
      // },
      // selectedState2: function (newState) {
      //   if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
      //     bus.$emit(SELECT_STATE, this.selectedState, this.selectedState2);
      //   }
      // },
    }
  }

  function json2tree(json){
    let data = {label: '', children: []};
    if (isObject(json)) {
      Object.keys(json).map( key => {
        let child = json2tree(json[key]);
        child.label = key + child.label; // assign key back
        if (Array.isArray(json)){
          child.label = child.children[0].label;
        }
        data.children.push(child);
      });
    }
    else{
      data.label = ": " + json;
      delete data.children;
    }
    return data;
  }

  function isObject(thing){
    return (typeof thing === "object") && (thing !== null);
  }
</script>
