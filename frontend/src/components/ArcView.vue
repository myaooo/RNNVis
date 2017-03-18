<template>
  <div class="arc">
    <div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>
    <svg :id="svgId" :width="width" :height="height"> </svg>
    <div class="arc_config">
      <el-col :span="4">
        <span>Cluster No.</span>
      </el-col>
      <el-col :span="6">
        <el-slider v-model="clusterNum" :min="2" :max="20"></el-slider>
      </el-col>
    </div>
    <!--<el-tabs v-model="selectedState">
      <el-tab-pane v-for="state in states" :label="state" :name="state">


      </el-tab-pane>
    </el-tabs>-->
  </div>
</template>

<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import {bus, SELECT_MODEL} from 'event-bus.js';
  import { WordCloud } from '../layout/cloud.js';



  const cell2states = {
    'GRU': ['state'],
    'BasicLSTM': ['state_c', 'state_h'],
    'BasicRNN': ['state'],
  };

  export default {
    name: 'ArcView',
    data(){
      return{
         states: '',
         fdGraph: null,
         selectedState: '',
         model: '',
         wordNum: 200,
         clusterNum: 5,
         cluster_data: {},
         state_data: null,
         word_data: null,
         cluster_mode: 'positive',
      };
    },
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
    watch: {
      width: function (newWidth) {
        this.fdGraph.update_scale();
      },
      height: function (newHeight) {
        this.fdGraph.update_scale();
      },
      selectedState: function(newState) {
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h'){
          if (this.word_data === null || this.state_data === null) {
            let ps = this.loadStateWordData();
            var p = this.loadClusterData();
            ps.push(p);
            Promise.all(ps).then( values => {
              this.reset();
            });
          }
        }
      },
      wordNum: function (newWordNum) {
        console.log('wordNum has changed');
      },

      clusterNum: function (newClusterNum) {
        if (newClusterNum > 1) {
          this.loadClusterData().then( values => { this.reset() });
        }
      }
    },
    computed: {
      svgId: function() {
        return this.paneId(this.model, this.selectedState);
      }
    },
    methods: {
      paneId(model, state) {
        return `arc-${model}-${state}-svg`;
      },
      loadStateWordData() {
        let p1 = dataService.getProjectionData(this.model, this.selectedState, {}, response => {
          this.state_data = response.data;
          console.log('state data loaded');
        });
        let p2 = dataService.getStrengthData(this.model, this.selectedState, {top_k: this.wordNum}, response => {
          this.word_data = response.data;
          console.log('word data loaded');
        });
        return [p1, p2];
      },

      loadClusterData() {
        return bus.loadCoCluster(this.model, this.selectedState, this.clusterNum,
          { top_k: this.wordNum, mode: this.cluster_mode, })
          .then(() => {
            this.cluster_data = bus.getCoCluster(this.model, this.selectedState, this.clusterNum,
              { top_k: this.wordNum, mode: this.cluster_mode, });
            console.log(`co_cluster_data ${this.clusterNum} loaded`);
          });
      },
      reset() {
        if (this.fdGraph) {
          this.fdGraph.destroy();
        }
        const svg = document.getElementById(this.paneId(this.model, this.selectedState));
        this.fdGraph = new ForceDirectedGraph(svg);
        this.fdGraph.process_data(this.state_data, this.word_data, this.cluster_data, this.cluster_mode);
        this.fdGraph.insert_element();
        this.fdGraph.start_simulation();
      },
    },
    mounted() {
      bus.$on(SELECT_MODEL, (modelName) => {
        console.log(`Selected model in ArcView: ${modelName}`);
        this.model = modelName;
        bus.loadModelConfig(modelName).then( () => {
          this.states = bus.availableStates(modelName);
        });
      });
    },
};

class ForceDirectedGraph{
  constructor(svg, strengthfn) {
    let self = this;
    this.svg = d3.select(`#${svg.id}`);
    this.width = svg.clientWidth;
    this.height = svg.clientHeight;
    this.arcNodes = null;
    this.innerNodes = null;
    this.links = null;
    this.word_clouds = [];
    this.simulation = null;
    this.strengthfn = strengthfn || (v => {return v; });
    this.graph = null;
    this.rScale = 0.1;
    this.radius = 1.5;
    this.defaultAlpha = 0.3;
    this.scale = {
      x: null,
      y: null,
    };
    this.color = d3.scaleOrdinal(d3.schemeCategory20);
    this.strength_threshold = 0.5;
    this.normal_opacity_line = 0.1;
    this.high_opacity_line = 0.1;
    this.low_opacity_line = 0.01;
    this.normal_opacity_node = 0.5;
    this.high_opacity_node = 1;
    this.low_opacity_node = 0.01;
    this.strength_range = [0.1, 5];
    this.arc_radius = 0.9 * Math.min(self.width/2, self.height/2);
    this.cluster_data = [];
    this.g_links = null;

    // this.arc_gap_angle = Math.PI / 8;
  }

  process_data(arc_data, inner_data, cluster_data, cluster_mode) {
    let self = this;
    let label2arc = [];
    let label2inner = [];
    let circle_radius = 0.9 * Math.min(self.width/2, self.height/2);
    this.cluster_data = cluster_data;

    label2arc = cluster_data.colClusters.map((cluster, i) => {
      return { data: cluster.map((col) => {
        return { index: col, label: i };
      }) };
    })
    label2inner = cluster_data.rowClusters.map((cluster, i) => {
      return {data: cluster.map((row) => {
        return { index: row, label: i, word: cluster_data.words[row] };
      })};
    })
    // compute links data and inner data
    let links = [];
    console.log(cluster_data);
    Object.keys(label2inner).forEach( (inner_k) => {
      label2inner[inner_k]['type'] = 'inner';
      Object.keys(label2arc).forEach( (arc_k) => {
        label2arc[arc_k]['type'] = 'arc';
        // calculate strength between arc_cluster and inner_cluster
        let cluster_strength = 0;
        label2inner[inner_k]['data'].forEach((inner_item) => {
          label2arc[arc_k]['data'].forEach((arc_item) => {
            let strength_item = cluster_data.correlation[inner_item.index][arc_item.index];
            switch(cluster_mode) {
              case 'positive':
                cluster_strength += strength_item > 0 ? strength_item : 0;
                break;
              case 'negative':
                cluster_strength += strength_item < 0 ? Math.abs(strength_item) : 0;
                break;
              case 'abs':
                cluster_strength += Math.abs(strength_item);
                break;
              case 'raw':
                cluster_strength += strength_item;
                break;
            }
          });
        });
        let link = {
          source: label2arc[arc_k],
          target: label2inner[inner_k],
          strength: cluster_strength,
        }
        links.push(link);
      //   if (label2arc[inner_k].links === undefined) {
      //     label2arc[inner_k].links = [];
      //   }
      //   label2arc[inner_k].links.push(link);
      })
    });

    //normalize_strength
    links = self.strength_normalize(links, self.strength_range);

    console.log("finished preparing data");
    this.graph = {
      links: links,
      label2arc: label2arc,
      label2inner: label2inner,
      arc_data: arc_data,
      inner_data: inner_data,
      nodes: label2arc.concat(label2inner),
    };
  }

  strength_normalize(links, range) {
    let strength_extent = d3.extent(links.map( (l) => {return l.strength; }));
    console.log(strength_extent);
    // console.log(range);
    let scale = d3.scaleLinear()
        .domain(strength_extent)
        .range(range);
    links.forEach( (l) => {l.strength = scale(l.strength); });
    return links;
  }

  draw_states() {
    let self = this;
    let width = this.width, height = this.height;
    let radius = 0.9 * Math.min(width/2, height/2);
    this.svg.append('g')
        .selectAll('circle')
        .data(this.graph.states)
        .enter()
        .append('circle')
        .attr('r', 3)
        .attr('cx', function(d) {return width / 2 + radius * Math.cos(d.angle)})
        .attr('cy', function(d) {return height / 2 + radius * Math.sin(d.angle)})
        .style('fill', function(d) {return self.color(d.label);})
  }

  insert_element() {
    let self = this;

    // draw arc path
    let g_arc = self.svg.append('g')
        .attr('transform', 'translate(' + self.width / 2 + ',' + self.height / 2 + ')');

    let arc = d3.arc()
        .innerRadius(self.arc_radius - 10)
        .outerRadius(self.arc_radius)

    let arc_data = d3.pie()(self.graph.label2arc.map((d) => {return d.data.length}));

    self.graph.label2arc.forEach((d, i) => {
      g_arc.append('path')
        .datum(arc_data[i])
        .style('fill', self.color(i))
        .attr('d', arc)
      let arc_centroid = arc.centroid(arc_data[i]);
      d.fx = arc_centroid[0] + self.width / 2;
      d.fy = arc_centroid[1] + self.height / 2;
    });

    // this.links = this.svg.append('g')
    //   .attr('class', 'links')
    //     .selectAll('line')
    //     .data(this.graph.links)
    //     .enter()
    //     .append('line')
    //     .classed('active', false)
    //     .style('opacity', (d) => {return this.normal_opacity_line;})
    //     .style('stroke', (d) => {return self.color(0)});

    
    
    // let lines_data = [];
    // self.graph.links.forEach((l) => {
    //   lines_data.push([l.source, l.target]);
    // });


    // this.graph.links.forEach( (l) => {
    //   console.log(`bundle line : ${line(l)}`)
    // })
    // let g_path = this.svg.append('g');
    // this.graph.links.forEach( (l) => {
    //   g_path.append('path')
    //     .datum(l)
    //     .each(()
    //     .attr('d', line)
    //     .attr('stroke', 'green')
    //     .attr('stroke-width', 3)
    //     .attr('fill', 'orange')
    // })
    // let path = this.svg.append('g')
    //   .selectAll('path')
    //   .data(lines_data)
    //   .enter()
    //   .append('path')
    //   .each((d) => {d.source = d[0], d.target = d[d.length - 1]})
    //   .attr('d', line)
    //   .attr('stroke', 'green')
    //   .attr('stroke-width', 3)
    //   .attr('fill', 'orange')

    // this.links.each(function(d) {
    //   d['el'] = this;
    // });

    let words_list = [];
    self.graph.label2inner.forEach((d) => {
      let words = d.data.map((w) => {
        return { text: w.word, size: (300 - w.index) / 15}
      });
      const radius = Math.sqrt(words.length) * 10 + 1;
      console.log(words);
      d['el_wc'] = this.svg.append('g');
      let myWordCloud = new WordCloud(d['el_wc'], radius);
      myWordCloud.update(words);
      self.word_clouds.push(d['el_wc']);
    });

    // this.innerNodes = this.svg.append('g')
    //   .attr('class', 'innerNodes')
    //     .selectAll('text')
    //     .data(self.graph.label2inner)
    //     .enter()
    //     .append('text')
    //     .classed('active', false)
    //     .text(function(d, i) {return i })
    //     .attr('x', self.width / 2)
    //     .attr('y', self.height / 2)
    //     .call(d3.drag()
    //       .on('start', d => {
    //         if (!d3.event.active) self.simulation.alphaTarget(self.defaultAlpha).restart();
    //         d.fx = d.x;
    //         d.fy = d.y;
    //       })
    //       .on('drag', d => {
    //         d.fx = d3.event.x;
    //         d.fy = d3.event.y;
    //       })
    //       .on('end', d => {
    //         if (!d3.event.active) self.simulation.alphaTarget(0);
    //         d.fx = null;
    //         d.fy = null;
    //     }))

    // this.innerNodes.each(function(d) {
    //   d['el'] = this;
    // })
  }

  start_simulation() {
    let self = this;
    var repelForce = d3.forceCollide(20)
      .radius(d => Math.sqrt(d.data.length) * 10 + 10);
    var init = repelForce.initialize;
    repelForce.initialize = function(nodes) {
      init(nodes.filter(function(d) {return d.type === 'inner'; }));
    }
    this.simulation = d3.forceSimulation().alpha(this.defaultAlpha)
        .force('link', d3.forceLink()
          .distance(function(d) {return d.strength > 0 ? 50 / d.strength : 300; })
          .strength(function (d) {return 1; }))
        .force('repel', repelForce)
    this.simulation
      .nodes(self.graph.nodes)
      .on('tick', () => self.ticked());

    this.simulation.force('link')
      .links(self.graph.links);
  }

  ticked() {
    let self = this;
    // self.links
    //   .attr('x1', function (d) {return d.source.fx; })
    //   .attr('y1', function (d) {return d.source.fy; })
    //   .attr('x2', function (d) {return d.target.x; })
    //   .attr('y2', function (d) {return d.target.y; })

    self.graph.label2inner.forEach((d) => {
      d['el_wc'].attr('transform', 'translate(' + d.x + ',' + d.y + ')');
    });

    let line = d3.line()
      .curve(d3.curveBundle.beta(1))
      .x((d) => {return d.x})
      .y((d) => {return d.y})
    
    let line_data = [];
    let data_center = [];
    self.graph.label2arc.forEach((d, i) => {
      self.graph.label2inner.forEach((w) => {
        if (data_center[i] === undefined) {
          data_center[i] = {x: 0, y: 0};
        }
        data_center[i].x = data_center[i].x + w.x / self.graph.label2inner.length;
        data_center[i].y = data_center[i].y + w.y / self.graph.label2inner.length;
      });
    });
    // console.log(data_center);
    self.graph.label2arc.forEach((d, i) => {
      self.graph.label2inner.forEach((w) => {
        line_data.push([d, data_center[i], w]);
      });
    });

    if (self.g_links) {
      self.g_links.remove();
    }
    self.g_links = this.svg.append('g');
    self.g_links
      .selectAll('path')
      .data(line_data).enter()
      .append('path')
      .attr('class', 'links')
      .attr('d', line)
      .attr('stroke', self.color(0))

  }

  destroy() {
      console.log(`Destroying Graph`);
      this.simulation.nodes([]);
      this.simulation.force("link").links([]);
      // this.links.remove();
      this.g_links.remove();
      // this.word_clouds.remove();
      this.word_clouds.forEach((d) => {d.remove()});
      // this.innerNodes.remove();
      // this.arcNodes.remove();
    }

};

</script>

<style>
  .links {
    fill: none;
    stroke-width: 2;
    opacity: 0.3;
  }

  .links .active {
    stroke-width: 3;
  }

  .words {

  }

  .words .active {

  }
</style>
