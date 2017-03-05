<template>
  <div class="project">
    <el-tabs v-model="selectedState">
    <el-tab-pane v-for="state in states" :label="state" :name="state">
      <svg :id="paneId(model, state)" :width="width" :height="height"> </svg>
    </el-tab-pane>

    </el-tabs>
  </div>
</template>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import {bus, SELECT_MODEL} from 'event-bus.js';

  const defaultParams = {
    rScale: 0.5,
    radius: 3.0,
    opacity: 0.4,
    opacityHigh: 1.0,
    defaultAlpha: 0.05,
    repel: -5,
    strength_thred: 0.5,
    color: d3.scaleOrdinal(d3.schemeCategory10)
  };

  const cell2states = {
    'GRU': ['state'],
    'BasicLSTM': ['state_c', 'state_h'],
    'BasicRNN': ['state'],
  };

  export default {
    name: 'ProjectView',
    data() {
      return {
        fdGraph: null,
        params: defaultParams,
        model: '',
        selectedState: '',
        states: '',
        // statesData: null,
        // strengthData: null
      };
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

    },
    watch: {
      width: function (newWidth) {
        this.fdGraoh.updateScale();
      },
      height: function (newWidth) {
        this.fdGraph.updateScale();
      },
      selectedState: function (newState) {
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h'){
          this.reset();
        }
      }
    },
    methods: {
      paneId(model, state) {
        return `${model}-${state}-svg`;
      },

      reset() {
        if (this.fdGraph){
          this.fdGraph.destroy();
        }
        // const svg = d3.select(`#${this.paneId}`)//.append('svg')
        //   .attr('width', this.width)
        //   .attr('height', this.height);
          // .attr('id', `${this.paneId}-svg`);
        const svg = document.getElementById(this.paneId(this.model, this.selectedState));
        this.fdGraph = new ForceDirectedGraph(svg, this.params);
        let statesData, strengthData;
        let p1 = dataService.getProjectionData(this.model, this.selectedState, {}, response => {
          statesData = response.data;
          // console.log(this.statesData)
          // console.log(statesData);
          console.log('states data loaded')
          // return 'done'
        });
        let p2 = dataService.getStrengthData(this.model, this.selectedState, { top_k: 200 }, response => {
          strengthData = response.data;
          strengthData = [strengthData[163]].concat(strengthData.slice(15, 50)).concat(strengthData.slice(170, 190));
          console.log('strength data loaded')
          // console.log(this.strengthData)
        });
        return Promise.all([p1, p2]).then(values => {
          normalize(statesData);
          let extents = calExtent(statesData);
          this.fdGraph.updateScale(extents);
          this.fdGraph.buildGraph(strengthData, statesData);
          this.fdGraph.insertElements();
          this.fdGraph.initSimulation();
          this.fdGraph.startSimulation();
        }, errResponse => {
          console.log("Failed to build force graph!");
        });
      }
    },
    mounted() {
      bus.$on(SELECT_MODEL, (modelName) => {
        console.log(`selected model: ${modelName}`);
        this.model = modelName;
        bus.loadModelConfig(modelName).then( (_) => {
          const config = bus.state.modelConfigs[modelName];
          const cell_type = config.model.cell_type;
          if (cell2states.hasOwnProperty(cell_type)){
            this.states = cell2states[cell_type];
            console.log(this.states);
            // this.selectedState = this.states[0];
            // this.reset();
          }
        })
      });
      // this.reset();
    }
  };

  function normalize(points) {
    const extents = calExtent(points);
    const factor_x = 100 / (extents[0][1] - extents[0][0]);
    const factor_y = 100 / (extents[1][1] - extents[1][0]);
    const factor = Math.max(factor_x, factor_y)
    for (let i = 0; i < points.length; i++){
      points[i].coords[0] *= factor;
      points[i].coords[1] *= factor;
    }
  }

  class ForceDirectedGraph {
    constructor(svg, params, strengthfn) {
      let self = this;
      this.svg = d3.select(`#${svg.id}`);
      this.svgEl = svg;
      this.graph = null;
      this.simulation = null;
      this.stateNodes = null;
      this.wordNodes = null;
      this.links = null;
      this.scale = {x: null, y: null, invert_x: null, invert_y: null};
      this.extents = null;
      this.strengthfn = strengthfn || (v => { return (v/2) ** 3; });
      Object.keys(params).forEach((p) => { self[p] = params[p]; });
    }

    get width() {
      return this.svgEl.clientWidth;
    }

    get height() {
      return this.svgEl.clientHeight;
    }

    buildGraph(words, states) {
      let self = this;
      // initialize Nodes and Links
      // console.log("initializing stateNodes")
      states.forEach(function (e) {
        e.id = "" + e.layer + "-" + e.state_id;
        e.fx = e.coords[0];
        e.fy = e.coords[1];
        e.links = [];
        // e.force = -200;
      });
      // console.log("initializing wordNodes")
      words.forEach(function (e) {
        e.id = e.word;
        e.links = [];
        // e.force = 0;
      });
      // console.log("initializing nodeLinks")

      let layers = Array.from(new Set(states.map(function (d) { return d.layer }))).sort();
      // console.log(layers)
      let links = [];
      let id2states = {};
      states.forEach(function (d) { id2states[d.id] = d });
      words.forEach(function (node) {
        let i = 0;
        node.strength.forEach(function (strengths) {  // the strength is stored as a list per layer's states
          let j = 0;  // state_id counter
          strengths.forEach(function (f) {
            let intensity = f;
            if (intensity > self.strength_thred) {  // a threshold strength
              // create link
              let link = {
                source: "" + layers[i] + "-" + j,  // the id of the stateNode
                target: node.id,
                strength: self.strengthfn(intensity),
                type: Math.sign(f)  // negative or positive strength
              };
              link._source = id2states[link.source];
              link._target = node;
              // add link to links array
              links.push(link);
              // add link to source and target nodes for reference
              id2states[link.source].links.push(link);
              node.links.push(link);
            }
            j++;
          });
          i++;
        });
      });
      console.log("done building graph")
      this.graph = {
        nodes: words.concat(states),
        links: links,
        id2states: id2states,
        wordNodes: words,
        stateNodes: states
      };
    }

    initSimulation() {
      var repelForce = d3.forceManyBody().strength(0); // the force that repel words apart
      var init = repelForce.initialize;
      repelForce.initialize = function (nodes) {
        init(nodes.filter(function (d) { d.hasOwnProperty("word") })); // only apply between word Nodes
      }
      var collideForce = d3.forceManyBody().strength(this.repel).distanceMax(5);
      this.simulation = d3.forceSimulation().alpha(this.defaultAlpha)
        .force("link", d3.forceLink() //.iterations(4)
          .id(function (d) { return d.id; })
          .strength(function (d) { return d.strength }))
        .force("collide", collideForce)
        .force("charge", repelForce);
    }

    insertElements() {
      let self = this;
      var g_states = this.svg.append('g')
        .attr('class', 'states');
      this.stateNodes = g_states.selectAll('circle')
        .data(this.graph.stateNodes)
        .enter().append("circle")
        // .attr("cx", function(d) { return d.fx; })
        // .attr("cy", function(d) { return d.fy; })
        .attr("r", function (d) {
          if (d.links.length > 0)
            return d.links.length * self.rScale + self.radius;
          return 0;
        })
        .style("fill", function (d, i) { return self.color(d.label); })
        .classed("active", false)
        .on("mouseover", d => mouseover(d, self))
        .on("mouseout", d => mouseout(d, self))
        .on("click", d => clickState(d, self));

      this.stateNodes.append("title")
        .text(function (d) { return d.id });
      //.style({opacity:'1.0'});

      // append a group for links
      this.links = this.svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(this.graph.links)
        .enter().append("line")
        .classed("active", false);

      // append a group for text nodes
      this.wordNodes = this.svg.append("g")
        .attr("class", "words")
        .selectAll("text")
        .data(this.graph.wordNodes)
        .enter().append("text")
        .classed("active", false)
        .call(d3.drag()
          .on("start", d => {
            if (!d3.event.active) self.simulation.alphaTarget(self.defaultAlpha).restart();
            d.fx = d.x;
            d.fy = d.y;
            // graph.links.forEach( function(d) {d.active = false; });
            d.links.forEach(function (d) { d.active = true; });
            refreshLinkStyle(self);
          })
          .on("drag", d => {
            d.fx = self.scale.invert_x(d3.mouse(self.svgEl)[0]);
            d.fy = self.scale.invert_y(d3.mouse(self.svgEl)[1]);
          })
          .on("end", d => {
            if (!d3.event.active) self.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
            d.links.forEach(function (d) { d.active = false; });
            refreshLinkStyle(self);
          }))
        .text(function (d) { return d.id });
      // console.log(this.links)

      refreshLinkStyle(self);
      refreshWordStyle(self);
      refreshStateStyle(self);
    }

    startSimulation() {
      let self = this;
      self.simulation
        .nodes(self.graph.nodes)
        .on("tick", () => self.ticked(self.links, self.wordNodes, self.stateNodes));

      self.simulation.force("link")
        .links(self.graph.links);
    }

    ticked(links, wordNodes, stateNodes) {
      // console.log(this.scale)
      let _graph = this;
      links
        .attr("x1", function (d) { return _graph.scale.x(d.source.x); })
        .attr("y1", function (d) { return _graph.scale.y(d.source.y); })
        .attr("x2", function (d) { return _graph.scale.x(d.target.x); })
        .attr("y2", function (d) { return _graph.scale.y(d.target.y); });

      wordNodes
        .attr("x", function (d) { return _graph.scale.x(d.x); })
        .attr("y", function (d) { return _graph.scale.y(d.y); });

      stateNodes
        .attr("cx", function (d) { return _graph.scale.x(d.x); })
        .attr("cy", function (d) { return _graph.scale.y(d.y); });
    }

    updateScale(extents) {
      if (extents) {
        this.extents = extents;
      }
      let width = this.width, height = this.height;
      let xExtent = this.extents[0],
        yExtent = this.extents[1];
      let xCenter = (xExtent[0] + xExtent[1]) / 2,
        yCenter = (yExtent[0] + yExtent[1]) / 2;
      let scaleFactor = 0.9 * Math.min(width / (xExtent[1] - xExtent[0]), height / (yExtent[1] - yExtent[0]));
      this.scale.x = function (_x) {
        return (_x - xCenter) * scaleFactor + width / 2;
      };
      this.scale.y = function (_y) {
        return (yCenter - _y) * scaleFactor + height / 2;
      };
      this.scale.invert_x = function (_x) {
        return (_x - width / 2) / scaleFactor + xCenter;
      };
      this.scale.invert_y = function (_y) {
        return yCenter - (_y - height / 2) / scaleFactor;
      };
    }

    destroy() {
      console.log(`Destroying Graph ${this.svgEl.id}`)
      this.simulation.nodes([]);
      this.simulation.force("link").links([]);
      this.links.remove();
      this.stateNodes.remove();
      this.wordNodes.remove();
      // this.graph;
    }

  }

  function calExtent(points) {
    var xExtent = d3.extent(points, function (d) { return d.coords[0] }),
      yExtent = d3.extent(points, function (d) { return d.coords[1] });
    return [xExtent, yExtent]
  }

  function mouseover(d, g) {
    if (d.active) return;
    d3.select(this).classed("active", true);
    d.links.forEach(function (l) {
      l.active = true;
      l._target.active = true;
    });
    refreshLinkStyle(g);
    refreshWordStyle(g);
  }

  function mouseout(d, g) {
    if (d.active) return;
    d3.select(this).classed("active", false);
    d.links.forEach(function (l) {
      l.active = false;
      l._target.active = false;
    });
    refreshLinkStyle(g);
    refreshWordStyle(g);
  }


  function clickState(d, g) {
    let isActive = d.active;
    if (isActive) {
      d.active = false;
      mouseout(d, g);
    }
    else {
      mouseover(d, g);
      d.active = true;
    }
    refreshLinkStyle(g);
    refreshWordStyle(g);
    d3.select(this).classed("active", d.active);
  }

  function refreshLinkStyle(g) {
    g.links.classed("active", function (d) { return d.active; })
      .style("opacity", function (d) {
        return d.strength * (d.active ? g.opacityHigh : g.opacity);
      })
      .style("stroke", function (d) {
        if (d.active)
          return "#6b7"
        if (d.type < 0)
          return "#68b";
        return "#c66";
      });
  }

  function refreshStateStyle(g) {
    g.stateNodes.classed("active", function (d) { return d.active; });
    // stateNodes.attr('r', function(d){ return d.links.length * rScale + radius})
  }

  function refreshWordStyle(g) {
    g.wordNodes.classed("active", function (d) { return d.active; })
  }

</script>
<style>
  .links .active {
    stroke-width: 3;
  }

  .links {
    stroke-width: 1.5;
    pointer-events: none;
  }

  .states .active {
    stroke-width: 2.0;
    stroke: black;
    opacity: 1.0;
  }

  .states {
    stroke-width: 0.0;
    opacity: 0.7;
  }

  .words .active {
    stroke-width: 0.5;
    stroke: black;
    opacity: 1.0;
  }

  .words {
    font-size: 13;
    opacity: 0.8;
    stroke: black;
    stroke-width: 0;
  }
</style>
