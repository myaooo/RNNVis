<style>
.hidden-cluster {
  stroke: gray;
  fill: lightgray;
}
#middle_line {
  stroke: lightskyblue;
  stroke-width: 2;
  stroke-dasharray: 3, 3;
  fill: none;
  opacity: 0.5;
}
.little-triangle {
  fill: #1f77b4;
}
</style>
<template>
  <div>
    <!--<div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>-->
    <svg :id='svgId' :width='width' :height='height'> </svg>
  </div>
</template>

<script>
  import * as d3 from 'd3'
  import { bus, SELECT_MODEL, SELECT_STATE, CHANGE_LAYOUT } from '../event-bus'
  import { WordCloud } from '../layout/cloud.js';


  const layoutParams = {
    clusterInterval: 20,
    packNum: 3,
    unitWidth: 4,
    unitHeight: 4,
    unitMargin: 1,
    wordCloudArcDegree: 110,
    wordCloudNormalRadius: 20,
    littleTriangleWidth: 5,
    littleTriangleHeight: 5,
    // clusterHeight: 16,
    // clusterWidth: 2,
  };
  layoutParams.clusterHeight = layoutParams.unitHeight*layoutParams.packNum + layoutParams.unitMargin * (layoutParams.packNum + 1);
  // layoutParams.clusterWidth = layoutParams.clusterHeight / (layoutParams.packNum);

  export default {
    name: 'ClusterView',
    data() {
      return {
        params: layoutParams,
        svgId: 'cluster-svg',
        clusterData: null,
        clusterNum: 5,
        painter: null,
        shared: bus.state,
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
    computed: {
      selectedState: function() {
        console.log(`cluster > state changed to ${this.shared.selectedState}`);
        return this.shared.selectedState;
      },
      selectedModel: function() {
        return this.shared.selectedModel;
      }
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

      bus.$on(CHANGE_LAYOUT, (layout, compare) => {
        if (compare)
          return;
        console.log("cluster > Changing Layout...");
        // this.clusterNum = layout.clusterNum;
      });
    }
  }

  class Painter {
    constructor(selector, params = layoutParams) {
      this.svg = d3.select(selector);
      this.params = params;
      this.hg = this.svg.append('g');
      this.wg = this.svg.append('g');

      this.client_width = this.svg.node().getBoundingClientRect().width;
      this.client_height = this.svg.node().getBoundingClientRect().height;
      this.middle_line_x = this.client_width / 3;
      this.triangle_height = 5;
      this.triangle_width = 5;

      this.state_elements = [];
      this.loc = null;
    }

    calculate_state_info(coCluster) {
      let state_loc = [];
      let state_cluster_loc = [];
      let little_triangle_loc = [];

      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;

      const stateClusters = coCluster.colClusters;
      const agg_info = coCluster.aggregation_info();
      const nCluster = coCluster.labels.length;
      const words = coCluster.words;

      stateClusters.forEach((clst, i) => {
        let width = Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
        let height = clusterHeight;
        let top_left = [-width / 2, i * (clusterHeight + clusterInterval)];

        state_cluster_loc[i] = {top_left: top_left, width: width, height: height};

        clst.forEach((c, j) => {
          let s_width = unitWidth;
          let s_height = unitHeight;
          let s_top_left = [(~~(j/packNum)) * (unitMargin + unitWidth) + 1 + top_left[0],
            j%packNum * (unitHeight + unitMargin) + 1 + top_left[1]];
          state_loc[c] = {top_left: s_top_left, width: s_width, height: s_height};
        });
      });
      return {state_cluster_info: state_cluster_loc, state_info: state_loc};
    }

    calculate_word_info(coCluster, high_light=[]) {
      let self = this;
      const wordCloudNormalRadius = this.params.wordCloudNormalRadius;
      const wordCloudArcDegree = this.params.wordCloudArcDegree;
      const clusterInterval = this.params.clusterInterval;
      const clusterHeight = this.params.clusterHeight;
      const wordClusters = coCluster.rowClusters;
      const words = coCluster.words;
      const nWord = words.length;
      const nCluster = coCluster.labels.length;
      const agg_info = coCluster.aggregation_info();

      let chordLength = nCluster * (clusterHeight + clusterInterval);
      const wordCloudArcRadius = chordLength / 2 / Math.sin(wordCloudArcDegree / 2 * Math.PI / 180);
      // console.log(`word cloud arc radius is word`)
      const wordCloudArcLength = wordCloudArcRadius * wordCloudArcDegree * Math.PI / 180;

      let word_cluster_info = [];
      let wd_radius = wordClusters.map((d, i) => {
        return high_light.length ? d.length + wordCloudNormalRadius : wordCloudNormalRadius;
      });

      let padding_degree = self.ArcLength2Angle(wordCloudArcLength - 2 * self.sum(wd_radius), wordCloudArcRadius) / nCluster;
      // console.log(padding_degree);
      let offset = -wordCloudArcDegree / 2;
      wordClusters.forEach((wdst, i) => {
        let tmp_radius_angle = self.ArcLength2Angle(wd_radius[i], wordCloudArcRadius);
        let angle_loc = tmp_radius_angle + offset;
        offset += 2 * tmp_radius_angle + padding_degree;
        let words_data = wdst.map((d) => {
          return {text: words[d], size: agg_info.row_single_2_col_cluster[d][i]};
        });
        let pos_x = wordCloudArcRadius * Math.cos(angle_loc / 180 * Math.PI);
        let pos_y = wordCloudArcRadius * Math.sin(angle_loc / 180 * Math.PI);
        // let link_pos_x = (wordCloudArcRadius - wd_radius[i]) * Math.cos(angle_loc / 180 * Math.PI);
        // let link_pos_y = (wordCloudArcRadius - wd_radius[i]) * Math.sin(angle_loc / 180 * Math.PI);
        let link_pos_x = pos_x - wd_radius[i];
        let link_pos_y = pos_y;
        word_cluster_info[i] = {position: [pos_x, pos_y], link_point_position: [link_pos_x, link_pos_y],
          word_cloud_radius: wd_radius[i], words_data: words_data};
        // word_cluster_info[i] = {arc_radius: wordCloudArcRadius, arc_angle_loc: angle_loc, word_cloud_radius: wd_radius[i], words_data: words_data};
      });
      return word_cluster_info;
    }

    calculate_link_info(state_info, word_info, coCluster, dx, dy) {
      const wordCloudArcDegree = this.params.wordCloudArcDegree;
      const clusterInterval = this.params.clusterInterval;
      const clusterHeight = this.params.clusterHeight;
      const nCluster = coCluster.labels.length;
      const agg_info = coCluster.aggregation_info();

      let chordLength = nCluster * (clusterHeight + clusterInterval);
      const wordCloudArcRadius = chordLength / 2 / Math.sin(wordCloudArcDegree / 2 * Math.PI / 180);
      let links = [];

      state_info.forEach((s, i) => {
        word_info.forEach((w, j) => {
          if (links[i] === undefined) {
            links[i] = [];
          }
          links[i][j] = {source: {x: s.top_left[0] + s.width,
            y: s.top_left[1] + s.height / 2},
            target: {x: w.link_point_position[0] + dx, y: w.link_point_position[1] + dy},
          };
        });
      });
      return links;
    }

    ArcLength2Angle(length, radius) {
      return length / radius * 180 / Math.PI;
    }

    draw_state_2(g, coCluster) {
      let self = this;
      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;

      g.append('line')
        .attr('id', 'middle_line')
        .attr('x1', 0)
        .attr('y1', -1000)
        .attr('x2', 0)
        .attr('y2', 1000)

      const hiddenClusters = g.selectAll('g rect')
        .data(coCluster.colClusters, (clst, i) => Array.isArray(clst) ? (String(clst.length) + String(i)) : this.id); // matching function

      // enter() the given data
      // add a group for holding all units in a cluster
      const hGroups = hiddenClusters.enter()
        .append('g')
        .attr('id', (clst, i) => (String(clst.length) + String(i)));

      hGroups
        .transition()
        .duration(400)
        .attr('transform', (clst, i) => {
          const width = Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
          return 'translate(' + [-width / 2, i * (clusterHeight + clusterInterval)] +')';
        });

      // add a background rect for this cluster
      hGroups.append('rect')
        .classed('hidden-cluster', true)
        .transition()
        .duration(400)
        .attr('width', (clst) => {
          return Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
        })
        .attr('height', clusterHeight);

      hGroups.append('path')
        .classed('little-triangle', true)
        .attr('d', 'M 0, 0 L ' + -self.triangle_width/2 + ', ' + self.triangle_height + ' L ' +  self.triangle_width/2 + ', ' + self.triangle_height + ' L 0, 0')
        .transition()
        .duration(400)
        .attr('transform', (k) => {
          let width_rect = Math.ceil(k.length / packNum) * (unitWidth + unitMargin) + unitMargin;
          return 'translate(' + [width_rect / 2, clusterHeight] + ')';
        });

      // add another group and specify data for units
      // see https://github.com/d3/d3-selection/blob/master/README.md#joining-data
      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      // enter units data
      // add a rect for each unit
      // add entering animations
      units.enter()
        .append('rect')
        .transition()
        .duration(400)
        .attr('width', unitWidth)
        .attr('height', unitHeight)
        .attr('transform', (u, j) => {
          return 'translate(' + [(~~(j/packNum)) * (unitMargin + unitWidth) + 1, j%packNum * (unitHeight + unitMargin) + 1] + ')';
        })
        .attr('fill', '#ff7f0e')
        .attr('fill-opacity', 0.5);

      // add exiting animation for units
      // units.exit()
      //   .transition()
      //   .duration(4000)
      //   .style('fill-opacity', 1e-6)
      //   .attr('width', 1)
      //   .attr('height', 1)
      //   .remove();

      hiddenClusters.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();
    }

    draw_state(g, coCluster, state_info) {
      let self = this;
      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;
      const littleTriangleWidth = this.params.littleTriangleWidth;
      const littleTriangleHeight = this.params.littleTriangleHeight;

      g.append('line')
        .attr('id', 'middle_line')
        .attr('x1', 0)
        .attr('y1', -1000)
        .attr('x2', 0)
        .attr('y2', 1000)

      const hiddenClusters = g.selectAll('g rect')
        .data(coCluster.colClusters, (clst, i) => Array.isArray(clst) ? (String(clst.length) + String(i)) : this.id); // matching function

      // hiddenClusters.exit().remove();

      const hGroups = hiddenClusters.enter()
        .append('g')
        .attr('id', (clst, i) => (String(clst.length) + String(i)));

      hGroups.append('rect')
        .classed('hidden-cluster', true)
        .transition()
        .duration(400)
        .attr('width', (clst, i) => state_info.state_cluster_info[i].width)
        .attr('height', (clst, i) => state_info.state_cluster_info[i].height)
        .attr('x', (clst, i) => state_info.state_cluster_info[i].top_left[0])
        .attr('y', (clst, i) => state_info.state_cluster_info[i].top_left[1])

      hGroups.append('path')
        .classed('little-triangle', true)
        .attr('d', 'M 0, 0 L ' + -littleTriangleWidth/2 + ', ' +
          littleTriangleHeight + ' L ' +  littleTriangleWidth/2 +
          ', ' + littleTriangleHeight + ' L 0, 0')
        .transition()
        .duration(400)
        .attr('transform', (k, i) => {
          return 'translate(' + [state_info.state_cluster_info[i].top_left[0] + state_info.state_cluster_info[i].width / 2,
            state_info.state_cluster_info[i].top_left[1] + state_info.state_cluster_info[i].height] + ')';
        });

      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      units.enter()
        .append('rect')
        .transition()
        .duration(400)
        .attr('width', (i) => state_info.state_info[i].width)
        .attr('height', (i) => state_info.state_info[i].height)
        .attr('x', (i) => state_info.state_info[i].top_left[0])
        .attr('y', (i) => state_info.state_info[i].top_left[1])
        // .attr('transform', (u, j) => {
        //   return 'translate(' + [(~~(j/packNum)) * (unitMargin + unitWidth) + 1, j%packNum * (unitHeight + unitMargin) + 1] + ')';
        // })
        .attr('fill', '#ff7f0e')
        .attr('fill-opacity', 0.5);

      // units.exit()
      //   .transition()
      //   .duration(4000)
      //   .style('fill-opacity', 1e-6)
      //   .attr('width', 1)
      //   .attr('height', 1)
      //   .remove();

      // add
      hiddenClusters.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();
    }

    sum(arr) {
      return arr.reduce(function (acc, val) {
        return acc + val;
      }, 0);
    }

    draw_word_2(g, coCluster) {
      let self = this;
      const rowClusters = coCluster.rowClusters;
      const agg_info = coCluster.aggregation_info();
      const nCluster = coCluster.labels.length;
      const words = coCluster.words;
      let word_clouds = [];

      console.log(agg_info);
      rowClusters.forEach((row_clst, i) => {
        console.log('rotate(' + i / nCluster * 90 + ')translate(' + [200, 0] + ')rotate(' + -i / nCluster * 90 + ')');
        let g_word = g.append('g')
          // .data(coCluster.colClusters)
          .attr('transform', 'rotate(' + (i / nCluster * 90 - 45) + ')translate(' + [500, 0] + ')rotate(' + -(i / nCluster * 90 - 45) + ')');
        console.log(row_clst);
        let words_data = row_clst.map((d) => {
          console.log({text: words[d], size: agg_info.row_single_2_col_cluster[d][i]/5});
          return {text: words[d], size: agg_info.row_single_2_col_cluster[d][i]/5};
        });
        console.log(words_data);
        // console.log(`words data is ${words_data}`);
        let myWordCloud = new WordCloud(g_word, words_data.length);
        myWordCloud.update(words_data);
      });

      // g.selectAll('circle')
      //   .data(rowClusters)
      //   .enter().append('circle')
      //   .attr('cx', 0)
      //   .attr('cy', (d, i) => {return 50 * i; })
      //   .attr('r', (d) => {return d.length})
      //   .attr('fill', 'pink')

    }

    draw_word_1(g, coCluster, word_info) {
      let self = this;
      const rowClusters = coCluster.rowClusters;
      const agg_info = coCluster.aggregation_info();
      const nCluster = coCluster.labels.length;
      const words = coCluster.words;

      rowClusters.forEach((clst, i) => {
        let tmp_g = g.append('g')
          .attr('transform', 'rotate(' + word_info[i].arc_angle_loc + ')translate(' +
            [word_info[i].arc_radius, 0] + ')rotate(' + -word_info[i].arc_angle_loc + ')');

        let myWordCloud = new WordCloud(tmp_g, word_info[i].word_cloud_radius)
        myWordCloud.update(word_info[i].words_data);
      });
    }

    draw_word(g, coCluster, word_info) {
      let self = this;
      const rowClusters = coCluster.rowClusters;
      const agg_info = coCluster.aggregation_info();
      const nCluster = coCluster.labels.length;
      const words = coCluster.words;

      rowClusters.forEach((clst, i) => {
        let myWordCloud = new WordCloud(g, word_info[i].word_cloud_radius)
          .translate(...word_info[i].position)
        myWordCloud.update(word_info[i].words_data);
      });
    }

    draw_link(g, link_info) {
      function link(d) {
        return "M" + d.source.x + "," + d.source.y
            + "C" + (d.source.x + d.target.x) / 2 + "," + d.source.y
            + " " + (d.source.x + d.target.x) / 2 + "," + d.target.y
            + " " + d.target.x + "," + d.target.y;
      }
      link_info.forEach((l) => {
        g.append('path')
          .attr('d', link(l))
          .attr('fill', 'none')
          .attr('stroke-width', '1')
          .attr('stroke', 'blue')
          .attr('opacity', 0.1)
      });
    }

    draw(coCluster) {
      let self = this;
      const clusterInterval = this.params.clusterInterval;
      const clusterHeight = this.params.clusterHeight;
      const nCluster = coCluster.labels.length;
      let chordLength = nCluster * (clusterHeight + clusterInterval);
      let dx = 200, dy = chordLength / 2;

      this.hg.attr('transform', 'translate(' + [this.middle_line_x, 100] + ')');
      this.wg.attr('transform', 'translate(' + [this.middle_line_x + dx, 100 + dy] + ')');
      // this.wg.attr('transform', 'translate(' + [this.middle_line_x + 100, 100 + chordLength / 2 - 50] + ')');
      const coClusterAggregation = coCluster.aggregation_info();
      let state_info = this.calculate_state_info(coCluster);
      // console.log(state_info.state_cluster_info)
      let word_info = this.calculate_word_info(coCluster);
      let link_info = this.calculate_link_info(state_info.state_cluster_info, word_info, coCluster, dx, dy);

      this.draw_state(this.hg, coCluster, state_info);
      this.draw_word(this.wg, coCluster, word_info);
      this.draw_link(this.hg, link_info);

    }
  }

</script>
