<template>
  <div class="model-view">
    <h3>Models</h3>
    <div class="grid-content bg-purple-light model-view">
      <el-form label-position="top" :model="params">
        <el-form-item label="Model">
          <el-select v-model="selected" placeholder="Select">
            <el-option v-for="(model, idx) in models" :value="idx" :label="model"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="Config"></el-form-item>
        <el-tree :data="configTree" :props="configProps" class="config-tree" :indent="10"></el-tree>
      </el-form>
    </div>
  </div>
</template>
<style>
  .model-view {
    text-align: left;
    padding: 14px;
  }

  .model-config {
    padding: 14px;
  }

  el-form-item .model-view {
    margin: 10px;
  }

  /*.config-tree {
    margin-top: 10px;
  }*/
</style>
<script>
  import { bus, SELECT_MODEL } from '../event-bus.js'

  export default{
    name: 'ModelView',
    data() {
      return {
        params: {
          selected: null,
        },
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
      }
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
