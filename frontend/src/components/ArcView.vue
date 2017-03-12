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
            <el-slider v-model="clusterNum" :min="2" :max="20"></el-slider>
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
         wordNum: 20,
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
        if (newClusterNum > 1 && !Object.prototype.hasOwnProperty.call(this.cluster_data, this.clusterNum)) {
          var p = this.loadClusterData();
          Promise.resolve(p).then( values => { this.reset() });
        } else {
          this.reset();
        }
      }
    },
    methods: {
      paneId(model, state) {
        return `${model}--${state}--svg`;
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
        var p = dataService.getCoCluster(this.model, this.selectedState, this.clusterNum, {
            top_k: this.wordNum, 
            mode: this.cluster_mode,
            }, response => {
            this.cluster_data[this.clusterNum] = response.data;
            console.log(`co_cluster_data ${this.clusterNum} loaded`);
          });
        return p;
      },
      reset() {
        if (this.fdGraph) {
          this.fdGraph.destroy();
        }
        const svg = document.getElementById(this.paneId(this.model, this.selectedState));
        this.fdGraph = new ForceDirectedGraph(svg);
        this.fdGraph.process_data(this.state_data, this.word_data, this.cluster_data[this.clusterNum], this.cluster_mode);
        this.fdGraph.insert_element();
        this.fdGraph.start_simulation();
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
    this.arcNodes = null;
    this.innerNodes = null;
    this.links = null;
    this.simulation = null;
    this.strengthfn = strengthfn || (v => {return v; });
    this.graph = null;
    this.rScale = 0.1;
    this.radius = 1.5;
    this.defaultAlpha = 1;
    this.scale = {
      x: null,
      y: null,
    };
    this.color = d3.scaleOrdinal(d3.schemeCategory10);
    this.strength_threshold = 0.5;
    this.normal_opacity_line = 0.1;
    this.high_opacity_line = 0.1;
    this.low_opacity_line = 0.01;
    this.normal_opacity_node = 0.5;
    this.high_opacity_node = 1;
    this.low_opacity_node = 0.01;
    this.strength_range = [0, 5];
    // this.arc_gap_angle = Math.PI / 8;
  }

  process_data(arc_data, inner_data, cluster_data, cluster_mode) {
    let self = this;
    let label2arc = [];
    let label2inner = [];
    let circle_radius = 0.9 * Math.min(self.width/2, self.height/2)

    arc_data.forEach( (d, i) => {
      if (label2arc[cluster_data.col[i]] === undefined) {
        label2arc[cluster_data.col[i]] = {data: [], };
      }

      d.index = i;
      d.label = cluster_data.col[i];
      label2arc[cluster_data.col[i]].data.push(d);
    });

    inner_data.forEach( (d, i) => {
      if (label2inner[cluster_data.row[i]] === undefined) {
        label2inner[cluster_data.row[i]] = {data: [], };
      }
      d.index = i;
      d.label = cluster_data.row[i];
      label2inner[cluster_data.row[i]].data.push(d);
    });

    let arc_offset = 0;
    Object.keys(label2arc).forEach( (k) => {

      let arc_length = 2 * Math.PI * label2arc[k].data.length / arc_data.length;
      // set angle for this cluster
      label2arc[k]['angle'] = (2 * arc_offset + arc_length) / 2;
      label2arc[k]['fx'] = circle_radius * Math.cos(label2arc[k]['angle']) + self.width / 2;
      label2arc[k]['fy'] = circle_radius * Math.sin(label2arc[k]['angle']) + self.height / 2;
      label2arc[k]['type'] = 'arc';
      let scale = d3.scaleLinear()
        .domain([0, label2arc[k].data.length - 1])
        .range([arc_offset, arc_offset + arc_length]);
      // set angle for individual point in the cluster
      label2arc[k].data.forEach( (d, i) => {
        d.angle = scale(i);
        d.fx = circle_radius * Math.cos(d.angle) + self.width / 2;
        d.fy = circle_radius * Math.sin(d.angle) + self.height / 2;
      });
      arc_offset += arc_length;
    });

    // compute links data and inner data
    let links = [];
    Object.keys(label2inner).forEach( (inner_k) => {
      console.log(inner_k);
      label2inner[inner_k]['type'] = 'inner';
      Object.keys(label2arc).forEach( (arc_k) => {
        // calculate strength between arc_cluster and inner_cluster
        let cluster_strength = 0;
        label2inner[inner_k]['data'].forEach((inner_item) => {
          label2arc[arc_k]['data'].forEach((arc_item) => {
            let strength_item = cluster_data['data'][inner_item.index][arc_item.index];
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
            }
          });
        });
        let link = {
          source: label2arc[arc_k],
          target: label2inner[inner_k],
          strength: cluster_strength,
        }
        links.push(link);
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
    console.log(range);
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
    this.arcNodes = self.svg.append('g')
      .selectAll('circle')
      .data(self.graph.arc_data)
      .enter()
      .append('circle')
      .attr('r', function(k) {
          return 1;
      })
      .classed('active', false)
      .classed('stateNode', true)
      .style('fill', function(d) { return self.color(d.label); })
      .style('fill-opacity', self.low_opacity)
    
    this.arcNodes.each( function (d) {
      d['el'] = this
    })

    this.arcNodes.append('title')
      .text(function (d) {return d.index});
    
    this.links = this.svg.append('g')
      .attr('class', 'links')
        .selectAll('line')
        .data(this.graph.links)
        .enter()
        .append('line')
        .classed('active', false)
        .style('opacity', (d) => {return this.normal_opacity_line;})
        .style('stroke', (d) => {return self.color(0)});
    
    this.links.each(function(d) {
      d['el'] = this;
    });
    
    this.innerNodes = this.svg.append('g')
      .attr('class', 'innerNodes')
        .selectAll('text')
        .data(self.graph.label2inner)
        .enter()
        .append('text')
        .classed('active', false)
        .text(function(d, i) {return i })
        .attr('x', self.width / 2)
        .attr('y', self.height / 2)
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
    
    this.innerNodes.each(function(d) {
      d['el'] = this;
    }) 
  }

  start_simulation() {
    let self = this;
    var repelForce = d3.forceCollide(20);
    var init = repelForce.initialize;
    repelForce.initialize = function(nodes) {
      init(nodes.filter(function(d) {return d.type === 'inner'; }));
    }
    this.simulation = d3.forceSimulation().alpha(this.defaultAlpha)
        .force('link', d3.forceLink()
          .distance(function(d) {return d.strength > 0 ? 10 : 100; })
          .strength(function (d) {return Math.abs(d.strength); }))
        .force('repel', repelForce)
    this.simulation
      .nodes(self.graph.nodes)
      .on('tick', () => self.ticked());
    
    this.simulation.force('link')
      .links(self.graph.links);
  }

  ticked() {
    let self = this;
    self.links
      .attr('x1', function (d) {return d.source.fx; })
      .attr('y1', function (d) {return d.source.fy; })
      .attr('x2', function (d) {return d.target.x; })
      .attr('y2', function (d) {return d.target.y; })
    
    self.innerNodes
      .attr('x', function (d) {return d.x; })
      .attr('y', function (d) {return d.y; })
    
    self.arcNodes
      .attr('cx', function (d) {return d.fx; })
      .attr('cy', function (d) {return d.fy; })
  }

  destroy() {
      console.log(`Destroying Graph`)
      this.simulation.nodes([]);
      this.simulation.force("link").links([]);
      this.links.remove();
      this.innerNodes.remove();
      this.arcNodes.remove();
    }

};

class ForceDirectedGraph_2{
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
        if (label2state['' + s.label] === undefined) {
          label2state['' + s.label] = []
        }
        label2state['' + s.label].push(s);
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