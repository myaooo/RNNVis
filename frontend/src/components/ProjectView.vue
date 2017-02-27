<template>
  <div class="project">
    <svg id="state_project"> </svg>
  </div>
</template>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';

  var rScale = 0.5;
  var radius = 3.0;
  var opacity = 0.4;
  var opacityHigh = 1.0;
  var defaultAlpha = 0.15;
  var repel = -300;
  var strength_thred = 0.5;
  var color = d3.scaleOrdinal(d3.schemeCategory10);

  export default {
    name: 'ProjectView',
    data() {
      return {
        msg:'hello vue'
      }
    },
    beforeMount() {

    },
    mounted() {
      // set the dimensions and margins of the diagram
      var margin = {
          top: 20,
          right: 120,
          bottom: 20,
          left: 120
        },
        width = 1200 - margin.left - margin.right,
        height = 1200 - margin.top - margin.bottom;

      var statesData = dataService.getProjectionData('lstm', 'state_h');
      var strengthData = dataService.getStrengthData('lstm', 'state_h').slice(0,200);
      strengthData = [strengthData[163]].concat(strengthData.slice(15,50)).concat(strengthData.slice(170,190));

      // initialize scales
      const scale = getScale(statesData, width, height);
      var graph = buildGraph(strengthData, statesData, scale);
      // console.log(allNodes.length)
      // Setting forces
      var simulation = initSimulation();

      // append the svg object to the body of the page
      var svg = d3.select('#state_project');
      svg.attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

      // append a group for the states projection points
      var g_states = svg.append('g')
        .attr('id', 'states'),
        stateNodes = g_states
          .selectAll('circle')
          .data(graph.stateNodes)
          .enter().append("circle")
          .attr("cx", function(d) { return d.fx; })
          .attr("cy", function(d) { return d.fy; })
          .attr("r", function(d) {
            if (d.links.length > 0)
              return d.links.length * rScale + radius;
            return 0;
          })
          .style("fill", function(d, i) { return color(d.label); })
          .classed("active", false)
          .on("mouseover", mouseover)
          .on("mouseout", mouseout)
          .on("click", clickState);

      stateNodes.append("title")
        .text(function(d) { return d.id });
          //.style({opacity:'1.0'});

      // append a group for links
      var linkLines = svg.append("g")
        .attr("id", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .classed("active", false);

      // append a group for text nodes
      var wordNodes = svg.append("g")
        .attr("id", "words")
        .selectAll("text")
        .data(graph.wordNodes)
        .enter().append("text")
        .classed("active", false)
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))
        .text(function(d) { return d.id })

      refreshLinkStyle(linkLines);
      refreshWordStyle(wordNodes);
      refreshStateStyle(stateNodes);

      console.log("configuring forces")
      simulation
        .nodes(graph.nodes)
        .on("tick", function() { ticked(linkLines, wordNodes); });

      simulation.force("link")
        .links(graph.links);
    }
  }

  // get a scale mapping fucntion from data to window
  function getScale(points, width, height) {
    var xExtent = d3.extent(points, function(d) { return d.coords[0] }),
      yExtent = d3.extent(points, function(d) { return d.coords[1] });
    var xCenter = (xExtent[0] + xExtent[1]) / 2,
      yCenter = (yExtent[0] + yExtent[1]) / 2;
    var scaleFactor = 0.9 * Math.min(width / (xExtent[1] - xExtent[0]), height / (yExtent[1] - yExtent[0]));

    return {
      x: function(_x) {
        return (_x - xCenter) * scaleFactor + width / 2;
      },
      y: function(_y) {
        return (yCenter - _y) * scaleFactor + height / 2;
      }
    }
  }

  function buildGraph(words, states, scale) {
    // initialize Nodes and Links
    console.log("initializing stateNodes")
    states.forEach(function(e) {
      e.id = "" + e.layer + "-" + e.state_id;
      e.fx = scale.x(e.coords[0]);
      e.fy = scale.y(e.coords[1]);
      e.links = [];
      e.force = -200;
    });
    console.log("initializing wordNodes")
    words.forEach(function(e) {
      e.id = e.word;
      e.links = [];
      e.force = 0;
    });
    console.log("initializing nodeLinks")

    let layers = Array.from(new Set(states.map(function(d) { return d.layer }))).sort();
    // console.log(layers)
    let links = [];
    let id2states = {};
    states.forEach(function(d) { id2states[d.id] = d});
    words.forEach(function(node) {
      let i = 0;
      node.strength.forEach(function(strengths) {  // the strength is stored as a list per layer's states
        let j = 0;  // state_id counter
        strengths.forEach(function(f) {
          let intensity = f;
          if (intensity > strength_thred){  // a threshold strength
            // create link
            let link = {
              source: "" + layers[i] + "-" + j,  // the id of the stateNode
              target: node.id,
              strength: (intensity/2)**3,
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
          j ++;
        });
        i ++;
      });
    });
    return {
      nodes: words.concat(states),
      links: links,
      id2states: id2states,
      wordNodes: words,
      stateNodes: states
    };
  }

  function initSimulation() {
    var repelForce = d3.forceManyBody().strength(0); // the force that repel words apart
    var init = repelForce.initialize;
    repelForce.initialize = function(nodes) {
      init(nodes.filter(function(d) {d.hasOwnProperty("word")})); // only apply between word Nodes
    }
    var collideForce = d3.forceManyBody().strength(repel).distanceMax(200);
    return d3.forceSimulation().alpha(defaultAlpha)
      .force("link", d3.forceLink().iterations(4)
        .id(function(d) { return d.id; })
        .strength(function(d) { return d.strength}))
      .force("collide", collideForce)
      .force("charge", repelForce);
  }

  function ticked(linkLines, wordNodes) {
    linkLines
      .attr("x1", function(d) { return d.source.fx; })
      .attr("y1", function(d) { return d.source.fy; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

    wordNodes
      .attr("x", function(d) { return d.x; })
      .attr("y", function(d) { return d.y; });

    // stateNodes
    //   .attr("cx", function(d) { return d.fx; })
    //   .attr("cy", function(d) { return d.fy; });
  }

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(defaultAlpha).restart();
    d.fx = d.x;
    d.fy = d.y;
    // graph.links.forEach( function(d) {d.active = false; });
    d.links.forEach(function(d) { d.active = true; });
    refreshLinkStyle();
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
    d.links.forEach( function(d) {d.active = false; });
    refreshLinkStyle();
  }

  function mouseover(d) {
    if (d.active) return;
    d3.select(this).classed("active", true);
    d.links.forEach(function(l) {
      l.active = true;
      l._target.active = true;
    });
    refreshLinkStyle();
    refreshWordStyle();
  }

  function mouseout(d) {
    if (d.active) return;
    d3.select(this).classed("active", false);
    d.links.forEach(function(l) {
      l.active = false;
      l._target.active = false;
    });
    refreshLinkStyle();
    refreshWordStyle();
  }

  function clickState(d){
    let isActive = d.active;
    if (isActive) {
      d.active = false;
      mouseout(d);
    }
    else {
      mouseover(d);
      d.active = true;
    }
    refreshLinkStyle();
    refreshWordStyle();
    d3.select(this).classed("active", d.active);
  }

  function refreshLinkStyle(linkLines) {
    linkLines.classed("active", function(d) { return d.active; })
      .style("opacity", function(d) {
        return d.strength * (d.active ? opacityHigh : opacity);
      })
      .style("stroke", function(d) {
        if (d.active)
          return "#6b7"
        if (d.type < 0)
          return "#68b";
        return "#c66";
      });
  }

  function refreshStateStyle(stateNodes) {
    stateNodes.classed("active", function(d) { return d.active; });
    // stateNodes.attr('r', function(d){ return d.links.length * rScale + radius})
  }

  function refreshWordStyle(wordNodes) {
    wordNodes.classed("active", function(d) { return d.active; })
  }

</script>
<style>

#links .active {
  stroke-width: 3;
}

#links {
  stroke-width: 1.5;
  pointer-events: none;
}

#states .active {
  stroke-width: 2.0;
  stroke: black;
  opacity: 1.0;
}

#states {
  stroke-width: 0.0;
  opacity: 0.7;
}

#words .active {
  stroke-width: 0.5;
  stroke: black;
  opacity: 1.0;
}

#words {
  font-size: 13;
  opacity: 0.8;
  stroke: black;
  stroke-width: 0;
}

</style>
