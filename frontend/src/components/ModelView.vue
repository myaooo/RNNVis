<template>
  <div>
    <h3>Models</h3>
    <div class="grid-content bg-purple-light">
      <el-select v-model="selected" placeholder="Select">
        <el-option v-for="(model, idx) in models" :value="idx" :label="model"></el-option>
      </el-select>
      <div class="model-config">
        <span>Configs</span>
        <el-tree :data="configTree" :props="configProps" class="config-tree"></el-tree>
      </div>
    </div>
  </div>
</template>
<style>
  .model-config {
    text-align: left;
    padding: 14px;
  }

  .config-tree {
    margin-top: 10px;
  }
</style>
<script>
  import dataService from '../services/dataService.js'
  import { bus, SELECT_MODEL } from '../event-bus.js'

  // let activeColorScheme = ["88, 126, 182", "201, 90, 95"];
  export default{
    name: 'ModelView',
    data() {
      return {
        selected: null,
        shared: bus.state,
        // models: bus.availableModels,
        configTree: null,
        // config: null,
        configProps: {
          children: 'children',
          label: 'label'
        },
        configs: {}
      };
    },
    mounted() {
      // bus.loadAvailableModels();
      this.getModels();
    },
    computed: {
      models: function() {
        return this.shared.availableModels;
      }
    },
    methods: {
      getModels() {
        bus.loadAvailableModels();
      }
    },
    watch: {
      selected: function(selected){
        const selectedModel = this.models[selected];
        bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
          .then( value => {
            const config = this.shared.modelConfigs[selectedModel];
            this.configTree = json2tree(config).children;
          });
      }
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
