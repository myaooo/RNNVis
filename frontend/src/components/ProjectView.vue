<template>
  <div class="project">
    <svg id="state_project"> </svg>
  </div>
</template>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
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

      var radius = 3.0;
      var opacity = 0.3;
      var color = d3.scaleOrdinal(d3.schemeCategory10);
      // set the dimensions and margins of the diagram
      var margin = {
          top: 20,
          right: 120,
          bottom: 20,
          left: 120
        },
        width = 1200 - margin.left - margin.right,
        height = 1200 - margin.top - margin.bottom;

      var statesData = dataService.getProjectionData();
      var strengthData = dataService.getStrengthData().slice(0,200);
      strengthData = [strengthData[163]].concat(strengthData.slice(10,50));

      // get a scale mapping fucntion from data to window
      function getScale(points) {
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

      function buildGraph(words, states) {
        let layers = Array.from(new Set(states.map(function(d) { return d.layer }))).sort();
        let links = [];
        let id2states = {};
        states.forEach( function(d) { id2states[d.id] = d});

      }

      // create links between stateNode and wordNode
      function createLinks(nodes, states, layers=[]) {
        var links = [];
        nodes.forEach(function(node) {
          if (layers == null)
            layers = Array(node.strength.length).map(function(d, i) { return i+1; })
          let i = 0;  // layer counter
          node.strength.forEach(function(strengths) {  // the strength is stored as a list per layer's states
            let j = 0;  // state_id counter
            strengths.forEach(function(f) {
              let intensity = f;
              if (intensity > 1.0){  // a threshold strength
                links.push({
                  source: "" + layers[i] + "-" + j,  // the id of the stateNode
                  target: node.id,
                  strength: (intensity/2)**2/2,
                  type: Math.sign(f)  // negative or positive strength
                });

              }
              j ++;
            });
            i ++;
          });
        });
        return links;
      }

      // initialize scales
      var scale = getScale(statesData)
      // initialize Nodes and Links
      console.log("initializing stateNodes")
      statesData.forEach(function(e) {
        e.id = "" + e.layer + "-" + e.state_id;
        e.fx = scale.x(e.coords[0]);
        e.fy = scale.y(e.coords[1]);
        e.links = [];
      });
      console.log("initializing wordNodes")
      strengthData.forEach(function(e) {
        e.id = e.word;
        e.links = [];
      });
      console.log("initializing nodeLinks")
      var allNodes = statesData.concat(strengthData),
        links = createLinks(strengthData, statesData, [2]);
      // console.log(allNodes.length)
      // Setting forces
      var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().iterations(4)
          .id(function(d) { return d.id; })
          .strength(function(d) { return d.strength}))
        .force("collide", d3.forceCollide(radius+2))
        .force("charge", d3.forceManyBody()
          .strength( function(d) {
            if (d.hasOwnProperty('state_id'))
              return -30;
            else
              return -300;
          }));

      // append the svg object to the body of the page
      var svg = d3.select('#state_project')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

      // append a group for the states projection points
      var g_states = svg.append('g')
        .attr('class', 'states'),
        stateNodes = g_states
          .selectAll('circle')
          .data(statesData)
          .enter().append("circle")
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; })
          .attr("r", radius)
          .style("fill", function(d, i) { return color(d.label); })
          .style("stroke", "#fff")
          .style("stroke", "1px")
          .style("opacity", 0.7)
          .on("mouseover",
            function(d) {
              d3.select(this)
                .style("stroke", "#000")
                .style("stroke", "2px")
                .style("opacity", 1.0);
            })
          .on("mouseout",
            function(d) {
              d3.select(this)
                .style("stroke", "#fff")
                .style("stroke", "1px")
                .style("opacity", 0.7);
            });

      stateNodes.append("title")
        .text(function(d) { return d.id });
          //.style({opacity:'1.0'});

      // append a group for links
      var linkLines = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .style("opacity", function(d) { return opacity * d.strength })
        .style("stroke", function(d) { if (d.type < 0) return "#22a"; return "#a22"});

      // append a group for text nodes
      var wordNodes = svg.append("g")
        .attr("class", "words")
        .selectAll("text")
        .data(strengthData)
        .enter().append("text")
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))
        .text(function(d) { return d.id })
        .style("font-size", "13")
        .style("opacity", 0.7);

      console.log("configuring forces")
      simulation
        .nodes(allNodes)
        .on("tick", ticked);

      simulation.force("link")
        .links(links);

      function ticked() {
        linkLines
          .attr("x1", function(d) { return d.source.fx; })
          .attr("y1", function(d) { return d.source.fy; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

        wordNodes
          .attr("x", function(d) { return d.x; })
          .attr("y", function(d) { return d.y; });

        stateNodes
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
      }

      function dragstarted(d) {
        if (!d3.event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
      }

      function dragended(d) {
        if (!d3.event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }

      function mouseover(d) {
        d3.select(this)
          .style("stroke", "#000")
          .style("stroke", "2px")
          .style("opacity", 1.0);
      }

      function mouseout(d){
        d3.select(this)
          .style("stroke", "#fff")
          .style("stroke", "1.5px")
          .style("opacity", 0.7);
      }

    }
  }
</script>
<style>

</style>
