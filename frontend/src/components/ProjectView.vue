<template>
  <div class="project">
    <el-tabs v-model="selectedState">
      <el-tab-pane v-for="state in states" :label="state" :name="state">
        <project-graph :svg-id="paneId(model, state)" :width="width" :height="height" :graph-data="graphData" :ready="ready" :config="config">
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
      </el-tab-pane>

    </el-tabs>
  </div>
</template>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService';
  import { bus, SELECT_MODEL } from 'event-bus';
  import ProjectGraph from 'components/ProjectGraph';

  const cell2states = {
    'GRU': ['state'],
    'BasicLSTM': ['state_c', 'state_h'],
    'BasicRNN': ['state'],
  };

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
      };
    },
    components: { ProjectGraph },
    computed: {

    },
    watch: {
      selectedState: function (newState) {
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
          this.reset();
        }
      },
      // config: function (newConfig)
    },

    methods: {
      paneId(model, state) {
        return `${model}-${state}-svg`;
      },

      reset() {
        this.ready = false;
        // const data = {states: null, strength: null};
        const p1 = dataService.getProjectionData(this.model, this.selectedState, {}, response => {
          this.graphData.states = response.data;
          console.log('states data loaded');
        });
        const p2 = dataService.getStrengthData(this.model, this.selectedState, { top_k: 200 }, response => {
          const strengthData = response.data;
          this.graphData.strength = [strengthData[163]].concat(strengthData.slice(15, 20)).concat(strengthData.slice(170, 190));
          console.log('strength data loaded');
        });
        const pAll = Promise.all([p1, p2]).then(values => {
          setTimeout(() => { this.ready = true; }, 100)
          // this.ready = true;
        }, errResponse => {
          console.log("Failed to build force graph!");
        });
      },

      configWatcher() {
        if (this.config.clusterNum > 1 && !Object.prototype.hasOwnProperty.call(this.graphData, 'signature')) {
          dataService.getStateSignature(this.model, this.selectedState, {}, response => {
            this.graphData.signature = response.data;
            console.log('signature data loaded');
            bus.$emit("REFRESH_PROJECT_GRAPH");
          });
          return;
        }
        bus.$emit("REFRESH_PROJECT_GRAPH");
      }
    },

    mounted() {
      bus.$on(SELECT_MODEL, (modelName) => {
        console.log(`selected model: ${modelName}`);
        this.model = modelName;
        bus.loadModelConfig(modelName).then((_) => {
          const config = bus.state.modelConfigs[modelName];
          const cell_type = config.model.cell_type;
          if (cell2states.hasOwnProperty(cell_type)) {
            this.states = cell2states[cell_type];
            // console.log(this.states);
            setTimeout(() => {
              this.selectedState = this.states[0];
            }, 100);
          }
        })
      });
    },

  };

</script>
<style>
.config span {
  line-height: 34px;
}
</style>
