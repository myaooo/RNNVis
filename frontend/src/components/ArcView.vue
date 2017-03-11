<template>
  <div class="arc">
    <el-tabs v-model="selectedState">
      <el-tab-pane v-for="state in states" :label="state" :name="state">
        <svg :id="paneId(model, state)" :width="width" :height="height"> </svg>
        <div class="arc_config">
          <el-col :span="4">
            <span>Cluster No.</span>
          </el-col>
          <el-col :span="6">
            <el-slider v-model="clusterNum" :min="1" :max="20"></el-slider>
          </el-col>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import {bus, SELECT_MODEL} from 'event-bus.js';
  import { kmeans } from '../algorithm/cluster';


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
         clusterNum: 1,
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
          this.reset();
        }
      },
      clusterNum: function (newClusterNum) {
        console.log(`cluster number changed to ${newClusterNum}`);
        this.clusterNum = newClusterNum;
        let p = null;
        if (this.clusterNum > 1 && !Object.prototype.hasOwnProperty.call(this, 'signature')) {
            p = dataService.getStateSignature(this.model, this.selectedState, {}, response => {
            this.signature = response.data;
            console.log('signature data loaded');
          });
        }
        Promise.resolve(p).then(() => {
          this.clusterAssignments = kmeans(this.signature, this.clusterNum, 1000)
          this.reset();
        })
        
      }
    },
    methods: {
      paneId(model, state) {
        return `${model}--${state}--svg`;
      },
      reset() {
        if (this.fdGraph) {
          this.fdGraph.destroy();
        }
        let raw_data = []
        const svg = document.getElementById(this.paneId(this.model, this.selectedState));
        let p1 = dataService.getProjectionData(this.model, this.selectedState, {}, response => {
          let states_data = response.data;
          if (this.clusterNum > 1) {
            states_data.forEach((d, i) => {d.label = this.clusterAssignments[i]});
          }
          raw_data.push(states_data);
          console.log('state data loaded');
        });
        let p2 = dataService.getStrengthData(this.model, this.selectedState, {top_k: 50}, response => {
          raw_data.push(response.data);
          console.log('word data loaded');
        });
        Promise.all([p1, p2]).then( values => {
          this.fdGraph = new ForceDirectedGraph(svg);
          this.fdGraph.process_data(raw_data[0], raw_data[1]);
          this.fdGraph.insert_element();
          this.fdGraph.start_simulation();

        });
        
      }, 
    },
    mounted() {
        bus.$on(SELECT_MODEL, (modelName) => {
            console.log(`Selected model in ArcView: ${modelName}`);
            this.model = modelName;
            bus.loadModelConfig(modelName).then( () => {
              const config = bus.state.modelConfigs[modelName];
              const cell_type = config.model.cell_type;
              if (cell2states.hasOwnProperty(cell_type)){
                this.states = cell2states[cell_type]
                console.log(this.states);
              }
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
    this.stateNodes = null;
    this.wordNodes = null;
    this.links = null;
    this.simulation = null;
    this.strengthfn = strengthfn || (v => {return v; });
    this.graph = null;
    this.rScale = 0.1;
    this.radius = 1.0;
    this.defaultAlpha = 1;
    this.scale = {
      x: null,
      y: null,
    };
    this.color = d3.scaleOrdinal(d3.schemeCategory10);
    this.strength_threshold = 0.5;
    this.normal_opacity_line = 0.01;
    this.high_opacity_line = 0.1;
    this.low_opacity_line = 0.01;
    this.normal_opacity_node = 0.5;
    this.high_opacity_node = 1;
    this.low_opacity_node = 0.01;
    this.arc_gap_angle = Math.PI / 8;
  }

  process_data(states, words) {
    let self = this;
    let label2state = {};
    let tmp = new Set();
    let angles = states.map(function(s) {
        tmp.add(s.layer);
        if (label2state[s.label] === undefined) {
          label2state[s.label] = []
        }
        label2state[s.label].push(s);
        s.angle = Math.atan2(s.coords[1], s.coords[0]);
        return Math.atan2(s.coords[1], s.coords[0]);
      });
    let layers = Array.from(tmp).sort();

    let label_num = Object.keys(label2state).length;
    this.arc_gap_angle = Math.PI / 8 / (label_num / 2 ) ** 3;
    let available_angle = 2 * Math.PI - (label_num === 1 ? 0 : label_num) * this.arc_gap_angle;
    
    let id2state = {};
    let links = [];
    let width = this.width, height = this.height;
    let radius = 0.9 * Math.min(width/2, height/2);

    let offset = 0;

    Object.keys(label2state).forEach(function (d) {
      let angle_extent = d3.extent(label2state[d].map((a) => {return a.angle}));
      let scale = d3.scaleLinear()
        .domain(angle_extent)
        .range([offset, offset + label2state[d].length / states.length * available_angle]);
      
      label2state[d].forEach(function (e) {
        e.id = '' + e.layer + '--' + e.state_id;
        e.angle = scale(e.angle);
        e.fx = width / 2 + radius * Math.cos(e.angle);
        e.fy = height / 2 + radius * Math.sin(e.angle);
        e.links = [];
        id2state[e.id] = e;
      });
      offset += self.arc_gap_angle + label2state[d].length / states.length * available_angle;
    });

    words.forEach(function (word) {
      word.id = word.word;
      word.links = [];
      word.x = width / 2;
      word.y = height / 2;
      word.strength.forEach(function(strengths, i) {
        strengths.forEach(function(s, j) {
          // console.log(`strength is ${s}`);
          if (s > self.strength_threshold) {
            let link = {
              source: id2state['' + layers[i] + '--' + j],
              target: word,
              strength: self.strengthfn(s),
            };
            link.source.links.push(link);
            word.links.push(link);
            links.push(link);
          }
        });
      });
    });

    console.log("finished preparing data");
    this.graph = {
      links: links,
      id2state: id2state,
      words: words,
      states: states,
    };
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
    this.stateNodes = self.svg.append('g')
      .selectAll('circle')
      .data(self.graph.states)
      .enter()
      .append('circle')
      .attr('r', function(d) {
        if (d.links.length > 0)
          return d.links.length * self.rScale + self.radius;
        else
          return 0;
      })
      .classed('active', false)
      .classed('stateNode', true)
      .style('fill', function(d) { return self.color(d.label); })
      .style('fill-opacity', self.low_opacity)
      .on('mouseover', function(d) {
          let links = new Set(d['links']);
          let words = new Set(d['links'].map((l) => {return l.target;}));
          let hide_links = self.graph.links.filter((x) => {return !links.has(x)})
          let hide_words = self.graph.words.filter((x) => {return !words.has(x)});
          hide_links.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'hidden');
          })
          hide_words.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'hidden');
          })
      })
      .on('mouseout', function(d) {
          self.graph.links.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'visible');
          })
          self.graph.words.forEach(function (d) {
            d3.select(d['el']).attr('visibility', 'visible');
          })
        })
    
    this.stateNodes.each( function (d) {
      d['el'] = this
    })

    this.stateNodes.append('title')
      .text(function (d) {return d.id});
    
    this.links = this.svg.append('g')
      .attr('class', 'links')
        .selectAll('line')
        .data(this.graph.links)
        .enter()
        .append('line')
        .classed('active', false)
        .style('opacity', (d) => {return Math.abs(d.strength) * self.normal_opacity_line;})
        .style('stroke', (d) => {return d.strength > 0 ? self.color(1) : self.color(0)});
    
    this.links.each(function(d) {
      d['el'] = this;
    });
    
    this.wordNodes = this.svg.append('g')
      .attr('class', 'words')
        .selectAll('text')
        .data(self.graph.words)
        .enter()
        .append('text')
        .classed('active', false)
        .text(function(d) {return d.id; })
        .call(d3.drag()
          .on('start', d => {
            if (!d3.event.active) self.simulation.alphaTarget(self.defaultAlpha).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', d => {
            d.fx = d3.event.x;
            d.fy = d3.event.y;
          })
          .on('end', d => {
            if (!d3.event.active) self.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }))
        .on('mouseover', function(d) {
          let links = new Set(d['links']);
          let states = new Set(d['links'].map((l) => {return l.source;}));
          let hide_links = self.graph.links.filter((x) => {return !links.has(x)})
          let hide_states = self.graph.states.filter((x) => {return !states.has(x)});
          hide_links.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'hidden');
          })
          hide_states.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'hidden');
          })
        })
        .on('mouseout', function(d) {
          self.graph.links.forEach(function(d){
            d3.select(d['el']).attr('visibility', 'visible');
          })
          self.graph.states.forEach(function (d) {
            d3.select(d['el']).attr('visibility', 'visible');
          })
        })
    this.wordNodes.each(function(d) {
      d['el'] = this;
    }) 
  }

  start_simulation() {
    let self = this;
    var repelForce = d3.forceManyBody().strength(10);
    var init = repelForce.initialize;
    repelForce.initialize = function(nodes) {
      init(nodes.filter(function(d) {d.hasOwnProperty('word')}));
    }
    this.simulation = d3.forceSimulation().alpha(this.defaultAlpha)
        .force('link', d3.forceLink()
          // .id(function (d) { return d.id; })
          .distance(function(d) {return d.strength > 0 ? 1 : 100; })
          .strength(function (d) {return Math.abs(d.strength); }))
        // .force('charge', repelForce)
        // .stop();
    this.simulation
      .nodes(self.graph.words.concat(self.graph.states))
      .on('tick', () => self.ticked());
    
    this.simulation.force('link')
      .links(self.graph.links);
    
    // for (var i = 0, n = Math.ceil(Math.log(this.simulation.alphaMin()) / Math.log(1 - this.simulation.alphaDecay())); i < n; ++i) {
    //   console.log(`Step ${i / n}`);
    //   this.simulation.tick();
    //   self.ticked();
    // }
  }

  ticked() {
    let self = this;
    self.links
      .attr('x1', function (d) {return d.source.fx; })
      .attr('y1', function (d) {return d.source.fy; })
      .attr('x2', function (d) {return d.target.x; })
      .attr('y2', function (d) {return d.target.y; })
    
    self.wordNodes
      .attr('x', function (d) {return d.x; })
      .attr('y', function (d) {return d.y; })
    
    self.stateNodes
      .attr('cx', function (d) {return d.fx; })
      .attr('cy', function (d) {return d.fy; })
  }

  ticked_2() {
    let self = this;

    self.links
      .attr('x1', function (d) {return self.scale.x(d._source.fx); })
      .attr('y1', function (d) {return self.scale.y(d._source.fy); })
      .attr('x2', function (d) {return self.scale.x(d._target.x); })
      .attr('y2', function (d) {return self.scale.y(d._target.y); })
    
    self.wordNodes
      .attr('x', function (d) {return self.scale.x(d.x); })
      .attr('y', function (d) {return self.scale.y(d.y); })
    
    self.stateNodes
      .attr('cx', function (d) {return self.scale.x(d.fx); })
      .attr('cy', function (d) {return self.scale.x(d.fy); })
  }

  update_scale() {
    let width = this.width;
    let height = this.height;
    let x_extent = d3.extent(this.graph.states, function(d) {return d.fx; })
    let y_extent = d3.extent(this.graph.states, function(d) {return d.fy; })
    let x_center = (x_extent[0] + x_extent[1]) / 2;
    let y_center = (y_extent[0] + y_extent[1]) / 2;
    let scale_factor = 0.9 * Math.min(width / (x_extent[1] - x_extent[0]), height / (y_extent[1] - y_extent[0]));
    this.scale.x = function(_x) {
      return (_x - x_center) * scale_factor + width / 2;
    }
    this.scale.y = function(_y) {
      return (_y - y_center) * scale_factor + height / 2;
    }
  }

  destroy() {
      console.log(`Destroying Graph`)
      this.simulation.nodes([]);
      this.simulation.force("link").links([]);
      this.links.remove();
      this.stateNodes.remove();
      this.wordNodes.remove();
    }

};
</script>

<style>
  .links {
    stroke-width: 1;
  }
  
  .links .active {
    stroke-width: 3;
  }

  .words {

  }

  .words .active {

  }
</style>