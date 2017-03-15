<style>
.hidden-cluster {
  stroke: gray;
  fill: lightgray;
}
</style>
<template>
  <div>
    <div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>
    <svg :id='svgId' :width='width' :height='height'> </svg>
  </div>
</template>

<script>
  import * as d3 from 'd3'
  import { bus, SELECT_MODEL } from '../event-bus'

  const layoutParams = {
    lineInterval: 12,
    clusterHeight: 18,
    packNum: 3,
    // clusterWidth: 2,
  };
  layoutParams.clusterWidth = layoutParams.clusterHeight / (layoutParams.packNum);

  export default {
    name: 'ClustaerView',
    data() {
      return {
        params: layoutParams,
        svgId: 'cluster-svg',
        clusterData: null,
        clusterNum: 15,
        painter: null,
        selectedModel: null,
        selectedState: null,
        states: [],
      }
    },
    props: {
      width: {
        type: Number,
        default: 800,
      },
      height: {
        type: Number,
        default: 800,
      },
    },
    watch: {
      selectedState: function (newState, oldState) {
        this.reload(this.selectedModel, newState, this.clusterNum);
      },
    },
    methods: {
      init() {
        this.painter = new Painter(`#${this.svgId}`);
      },
      reload(model, state, clusterNum) {
        bus.loadCoCluster(model, state, clusterNum)
          .then(() => {
            this.clusterData = bus.getCoCluster(model, state, clusterNum);
            this.painter.draw(this.clusterData);
          });
      },
    },
    mounted() {
      this.init();
      // register events
      bus.$on(SELECT_MODEL, (model) => {
        this.selectedModel = model;
        bus.loadModelConfig(model).then(() => {
          this.states = bus.availableStates(model);
        });
      })
    }
  }

  class Painter {
    constructor(selector, params = layoutParams) {
      this.svg = d3.select(selector);
      this.params = params;
      this.hg = this.svg.append('g');
      this.wg = this.svg.append('g');
    }
    draw(coCluster) {

      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = ~~(clusterHeight / packNum);
      const clusterWidth = this.params.clusterWidth;
      const lineInterval = this.params.lineInterval;
      const hiddenClusters = this.hg.selectAll('g rect')
        .data(coCluster.colClusters)

      const hGroups = hiddenClusters.enter()
        .append('g')
        .attr('transform', (clst, i) => {
          return 'translate(' + [-clst.length * clusterWidth / 2, i * (clusterHeight + lineInterval)] +')';
        });

      hGroups.append('rect')
        .attr('width', (clst) => clst.length * clusterWidth)
        .attr('height', clusterHeight)
        // .attr('x')
        .classed('hidden-cluster', true);

      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      // const unitWidth = clusterWidth - 1
      units.enter()
        .append('rect')
        .attr('width', clusterWidth-1)
        .attr('height', clusterHeight-1)
        .attr('transform', (u, j, d) => {
          return 'translate(' + [(d.length - ~~(j/packNum)) * clusterWidth, -j%packNum * unitHeight] + ')';
        })

      hiddenClusters.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();

      this.hg.attr('transform', 'translate(' + [200, 0] + ')')
    }
  }

</script>
