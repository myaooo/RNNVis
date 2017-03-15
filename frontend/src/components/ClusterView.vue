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
    clusterInterval: 12,
    packNum: 3,
    unitWidth: 4,
    unitHeight: 4,
    unitMargin: 1,
    // clusterHeight: 16,
    // clusterWidth: 2,
  };
  layoutParams.clusterHeight = layoutParams.unitHeight*layoutParams.packNum + layoutParams.unitMargin * (layoutParams.packNum + 1);
  // layoutParams.clusterWidth = layoutParams.clusterHeight / (layoutParams.packNum);

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

      this.hg.attr('transform', 'translate(' + [200, 0] + ')');
      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;
      console.log(coCluster.colClusters);

      const hiddenClusters = this.hg.selectAll('g rect')
        .data(coCluster.colClusters, (clst, i) => Array.isArray(clst) ? (String(clst.length) + String(i)) : this.id); // matching function

      // hiddenClusters.exit().remove();

      const hGroups = hiddenClusters.enter()
        .append('g')
        .attr('id', (clst, i) => (String(clst.length) + String(i)));
      hGroups.transition()
        .duration(400)
        .attr('transform', (clst, i) => {
          const width = Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
          return 'translate(' + [-width / 2, i * (clusterHeight + clusterInterval)] +')';
        });

      hGroups.append('rect')
        .classed('hidden-cluster', true)
        .transition()
        .duration(400)
        .attr('width', (clst) => {
          return Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
        })
        .attr('height', clusterHeight);
        // .attr('x')

      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      // const unitWidth = clusterWidth - 1
      units.enter()
        .append('rect')
        .transition()
        .duration(400)
        .attr('width', unitWidth)
        .attr('height', unitHeight)
        .attr('transform', (u, j) => {
          return 'translate(' + [(~~(j/packNum)) * (unitMargin + unitWidth) + 1, j%packNum * (unitHeight + unitMargin) + 1] + ')';
        })
        .attr('fill', 'steelblue')
        .attr('fill-opacity', 0.5);

      units.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();

      hiddenClusters.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();


    }
  }

</script>
