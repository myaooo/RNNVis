<template>
  <div>
    <h4 class="normal">Models</h4>
    <hr>
    <div class="content model-view">
      <model-config :compare="false"> </model-config>
      <hr v-if="compare">
      <model-config v-if="compare" :compare="true" :toggle="toggleCompare"> </model-config>
      <hr>
      <el-tooltip placement="top" :open-delay="500">
      <div slot="content">click to add another model for comparison</div>
      <el-button @click="toggleCompare" size="small">Compare Model</el-button>
      </el-tooltip>
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
      this.getAvailableModels();
    },
    computed: {
      // availableModels: function() { // model list
      //   return this.shared.availableModels;
      // },
    },
    methods: {
      getAvailableModels() {
        console.log("getting available models...");
        bus.loadAvailableModels();
        if(this.shared.availableModels) return;
        setTimeout(() => {
          this.getAvailableModels();
        }, 2000);
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
