<template>
  <div class="model-diagram">
    <svg :id="diagramId"></svg>
  </div>
</template>
<style>
  .model-view {
    text-align: left;
    padding: 5px;
  }

  .model-detail {
    padding: 10px;
  }

  el-form-item .model-view {
    margin: 5px;
  }

  /*.config-tree {
    margin-top: 10px;
  }*/
</style>
<script>
  import { bus, SELECT_MODEL } from '../event-bus.js'

  export default{
    name: 'ModelDiagram',
    data() {
      return {
        diagramId: 'model-diagram-svg',
      };
    },
    mounted() {
      this.getModels();
    },
    computed: {
      models: function() { // model list
        return bus.state.availableModels;
      }
    },
    methods: {
      getModels() {
        bus.loadAvailableModels();
      },
    },
    watch: {
      selected: function(selected){
        const selectedModel = this.models[selected];
        bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
          .then( value => {
            const config = bus.state.modelConfigs[selectedModel];
            this.configTree = json2tree(config).children;
            bus.$emit(SELECT_MODEL, selectedModel);
          });
      }
    }
  }

  class CellDiagram{
    constructor(svg, unrollNum){
      this.svgEl = svg;
      this.unrollNum = unrollNum;
    }
  }

  const RNN = {

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
