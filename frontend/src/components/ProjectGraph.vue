<template>
  <svg :id="svgId" :width="width" :height="height" class="project"> </svg>
</template>
<script>
  import * as d3 from 'd3';
  import { bus } from 'event-bus';
  import { kmeans } from '../algorithm/cluster';

  const defaultConfig = {
    rScale: 0.3,
    radius: 3.0,
    opacity: 0.4,
    opacityHigh: 1.0,
    defaultAlpha: 0.25,
    repel: -5,
    strength_thred: 0.5,
    color: d3.scaleOrdinal(d3.schemeCategory10),
    color2: d3.scaleOrdinal(d3.schemeCategory20),
    clusterNum: 0,
  };

  export default {
    name: 'ProjectGraph',
    defaultConfig: defaultConfig,
    data() {
      return {
        fdGraph: null,
        // defaultConfig: defaultConfig,
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
      graphData: {
        type: Object,
        required: true,
      },
      svgId: {
        type: String,
        required: true,
      },
      config: {
        type: Object,
        default: null,
      },
      ready: { // a small workaround of getting ready signal form parent
        type: Boolean,
        default: false,
      },
    },
    watch: {
      width: function (newWidth) {
        this.fdGraoh.updateScale();
      },
      height: function (newWidth) {
        this.fdGraph.updateScale();
      },
      ready: function (newReady) {
        if (newReady) {
          this.reset();
        }
      },
    },
    methods: {
      reset() {
        // const config = this.config === null ? defaultConfig : this.config;
        if (this.fdGraph) {
          this.fdGraph.destroy();
        }
        const svg = document.getElementById(this.svgId);
        this.fdGraph = new ForceDirectedGraph(svg, this.config);
        normalize(this.graphData.states);
        let extents = calExtent(this.graphData.states);
        this.fdGraph.updateScale(extents);
        this.fdGraph.buildGraph(this.graphData.strength, this.graphData.states);
        this.fdGraph.insertElements();
        this.fdGraph.initSimulation();
        this.fdGraph.startSimulation();
      },
      refreshGraph(ready) {
        if (ready){
          this.fdGraph.refresh();
          this.fdGraph.clusterStates(this.graphData.signature);
        }
      }
    },
    mounted() {
      // register event listeners
      bus.$on('REFRESH_PROJECT_GRAPH', this.refreshGraph);
      // this.reset();
    }
  };

  function normalize(points) {
    const extents = calExtent(points);
    const factor_x = 100 / (extents[0][1] - extents[0][0]);
    const factor_y = 100 / (extents[1][1] - extents[1][0]);
    const factor = Math.max(factor_x, factor_y)
    for (let i = 0; i < points.length; i++) {
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
      this.scale = { x: null, y: null, invert_x: null, invert_y: null };
      this.extents = null;
      this.strengthfn = strengthfn || (v => { return (v / 2) ** 3; });
      this.params = params;
      // Object.keys(params).forEach((p) => { self[p] = params[p]; });
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
            if (Math.abs(intensity) > self.params.strength_thred) {  // a threshold strength
              // create link
              let link = {
                source: "" + layers[i] + "-" + j,  // the id of the stateNode
                target: node.id,
                strength: self.strengthfn(Math.abs(intensity)),
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
      var repelForce = d3.forceManyBody().strength(-30).distanceMax(3); // the force that repel words apart
      var init = repelForce.initialize;
      repelForce.initialize = function (nodes) {
        init(nodes.filter(function (d) { d.hasOwnProperty("word") })); // only apply between word Nodes
      }
      // var collideForce = d3.forceManyBody().strength(this.params.repel).distanceMax(10);
      var collideForce = d3.forceCollide().strength(0.7).radius(2);
      this.simulation = d3.forceSimulation().alpha(this.params.defaultAlpha)
        .force("link", d3.forceLink() //.iterations(4)
          .id((d) => { return d.id; })
          .distance((l) => { return l.type < 0 ? 50 : 5; })
          .strength((d) => { return d.strength }))
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
            return d.links.length * self.params.rScale + self.params.radius;
          return 0;
        })
        .style("fill", function (d, i) { return self.params.color(d.label); })
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
            if (!d3.event.active) self.simulation.alphaTarget(self.params.defaultAlpha).restart();
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

    clusterStates(stateSignature) {
      const clusterNum = this.params.clusterNum;
      if (clusterNum < 2)
        return;
      const clusterAssignments = kmeans(stateSignature, clusterNum, 1000);
      this.graph.stateNodes.forEach( (node, i) => {
        node.cluster = clusterAssignments[i];
      });
      this.stateNodes.style('fill', (d, i) => {
        if (clusterNum < 11)
          return this.params.color(d.cluster);
        return this.params.color2(d.cluster);
      });
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

    refresh() {
      refreshLinkStyle(this);
      refreshStateStyle(this);
      refreshWordStyle(this);
    }

  };

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
        return d.strength * (d.active ? g.params.opacityHigh : g.params.opacity);
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
