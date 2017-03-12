<template>
  <div class="project">
    <div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>
    <project-graph :svg-id="paneId(model, selectedState)" :width="width" :height="height" :graph-data="graphData" :ready="ready" :config="config">
    </project-graph>
    <div class="config">
      <!--<el-form :model="config" :inline="true">-->
      <el-col :span="4">
        <span>Cluster No.</span>
      </el-col>
      <el-col :span="6">
        <el-slider v-model="config.clusterNum" :min="1" :max="20" @change="configWatcher"></el-slider>
      </el-col>
      <el-col :span="4">
        <span>Opacity</span>
      </el-col>
      <el-col :span="6">
        <el-slider v-model="config.opacity" :max="1" :step="0.05" @change="configWatcher" :width="100"></el-slider>
      </el-col>
      <!--</el-form>-->
    </div>

  </div>
</template>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService';
  import { bus, SELECT_MODEL } from 'event-bus';
  import ProjectGraph from 'components/ProjectGraph';

  export default {
    name: 'ProjectView',
    props: {
      width: {
        type: Number,
        default: 800,
      },
      height: {
        type: Number,
        default: 600,
      },
    },
    data() {
      const drawConfig = Object.assign({}, ProjectGraph.defaultConfig); // get a copy
      return {
        model: '',
        selectedState: '',
        states: '',
        ready: false,
        graphData: { states: null, strength: null },
        config: drawConfig,
        cache: {},
        shared: bus.state,
      };
    },
    components: { ProjectGraph },
    computed: {

    },
    watch: {
      selectedState: function (newState, oldState) {
        bus.state.selectedState = newState;
        if (newState === oldState) {
          return;
        }
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
          this.reset();
        }
      },
    },

    methods: {

      paneId(model, state) {
        return `${model}-${state}-svg`;
      },

      reset() { // reset the whole graph
        this.ready = false; // reset ready signal
        delete this.graphData.signature; // reset signature data
        this.config = Object.assign({}, ProjectGraph.defaultConfig); // reset controlings
        if(this.selectedState === 'state'){ //hack
          this.config.strength_thred = 0.9;
        }
        if(this.selectedState === 'state_h'){
          this.config.strength_thred = 0.5;
        }
        if(this.model === 'IMDB'){
          this.config.strength_thred = 0.0;
          this.config.strengthFn = (v => { return (v * 30) ** 2; });
          this.config.farDistance = 120;
        }
        const cacheTag = this.paneId(this.model, this.selectedState);
        if (Object.prototype.hasOwnProperty.call(this.cache, cacheTag)) { // already has the cache data
          Object.assign(this.graphData, this.cache[cacheTag]);
          setTimeout(() => { this.ready = true; }, 100);
          return;
        }
        // request for data
        const p1 = dataService.getProjectionData(this.model, this.selectedState, {}, response => {
          this.graphData.states = response.data;
          console.log('states data loaded');
        });
        const p2 = dataService.getStrengthData(this.model, this.selectedState, { top_k: 200 }, response => {
          const strengthData = response.data;
          this.graphData.strength = [strengthData[163]].concat(strengthData.slice(20, 40)); //.concat(strengthData.slice(170, 190));
          console.log('strength data loaded');
        });
        const pAll = Promise.all([p1, p2]).then(values => {
          this.cache[cacheTag] = { states: this.graphData.states, strength: this.graphData.strength }; // cache fetched data;
          setTimeout(() => { this.ready = true; }, 100);
          // continue to download signature data which might be used for clustering
          dataService.getStateSignature(this.model, this.selectedState, {}, response => {
            this.graphData.signature = response.data;
            this.cache[cacheTag].signature = this.graphData.signature;
            console.log('signature data loaded');
          });
        }, errResponse => {
          console.log("Failed to build force graph!");
        });

      },

      configWatcher() {
        if (this.ready === false) return;
        if (this.config.clusterNum > 1 && !Object.prototype.hasOwnProperty.call(this.graphData, 'signature')) {
          dataService.getStateSignature(this.model, this.selectedState, {}, response => {
            this.graphData.signature = response.data;
            console.log('signature data loaded');
            bus.$emit("REFRESH_PROJECT_GRAPH", this.ready);
          });
          return;
        }
        bus.$emit("REFRESH_PROJECT_GRAPH", this.ready);
      },
    },

    mounted() {
      bus.$on(SELECT_MODEL, (modelName) => {
        console.log(`selected model: ${modelName}`);
        this.model = modelName;
        bus.loadModelConfig(modelName).then((_) => {
          const states = bus.availableStates(modelName);
          if (states){
            this.states = states;
            // console.log(this.states);
            setTimeout(() => { // wait 100ms to update incase the dom is not ready
              if (this.states[0] === this.selectedState) { // ugly hacker incase states are same to manually reset the data.
                this.reset();
              }
              this.selectedState = null;
            }, 100);
          }
          else {
            console.log("Unknown cell type!");
          }
        })
      });
    },

  };

</script>
<style scope>
  .config span {
    line-height: 34px;
  }
  .header {
    text-align: left;
  }
</style>
