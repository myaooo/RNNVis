<template>
  <div class="grid-content">
    <el-form label-position="left" label-width="80px">
      <el-form-item :label="compare ? 'Model 2' : 'Model'">
        <el-select v-model="selectedModel" placeholder="Select A Model" size="small">
          <el-option v-for="(model, idx) in availableModels" :value="model" :label="model"></el-option>
        </el-select>
        <el-button v-if="compare" @click="toggle" icon="delete" size="small"></el-button>
      </el-form-item>
      <div v-if="config">
        <!--<el-tag>cell: {{config.Cell}}</el-tag>
        <el-tag>layer: {{config.LayerNum}}</el-tag>-->
        <el-tag v-for="key in Object.keys(config)" :type="compare ? 'gray' : ''">{{key}}: {{config[key]}}</el-tag>
      </div>
      <el-form-item label="Hidden State" v-if="states.length">
        <el-radio-group v-model="selectedState" size="small">
          <el-radio-button v-for="state in states" :label="state">{{stateName(state)}}</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="Cluster Num" v-if="selectedState">
        <el-slider v-model="layout.clusterNum" :min="2" :max="20" @change="layoutChange"></el-slider>
      </el-form-item>
    </el-form>
  </div>
</template>
<style scoped>
  .el-form-item {
    margin-bottom: 4px;
  }

  .el-tag {
    margin-right: 5px;
  }
</style>
<script>
  import { bus, SELECT_MODEL, SELECT_STATE, CHANGE_LAYOUT } from '../event-bus';

  export default {
    name: 'ModelConfig',
    data() {
      return {
        shared: bus.state,
        states: [],
        selectedState: null,
        selectedModel: null,
        config: null,
        layout: {clusterNum: 10},
      };
    },
    props: {
      toggle: {
        type: Function,
        default: () => { return; },
      },
      compare: {
        type: Boolean,
        default: false,
      },
    },
    computed: {
      availableModels: function() { // model list
        return this.shared.availableModels;
      },
    },
    watch: {
      selectedModel: function(selectedModel){
        if (!selectedModel) return;
        bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
          .then(() => {
            const states = bus.availableStates(selectedModel);
            if (states){
              this.states = states;
              this.selectedState = null; // reset
              const config = bus.state.modelConfigs[selectedModel];
              this.config = {
                Cell: config.model.cell_type,
                LayerNum: config.model.cells.length,
                LayerSize: config.model.cells[0].num_units,
              };
              bus.$emit(SELECT_MODEL, this.selectedModel, this.compare);
            }
          });
      },
      selectedState: function (newState) {
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
          bus.$emit(SELECT_STATE, this.selectedState, this.compare);
        }
      },
    },
    methods: {
      stateName(state) {
        switch(state) {
          case 'state_c': return 'c_state';
          case 'state_h': return 'h_state';
          case 'state': return 'h_state';
          default: return 'Unknown';
        }
      },
      layoutChange() {
        console.log("Layout changed")
        bus.$emit(CHANGE_LAYOUT, this.layout, this.compare);
      }
    }
  };
</script>
