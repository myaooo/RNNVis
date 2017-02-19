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

      var radius = 4.0;
      var statesData = dataService.getProjectionData();

      var xExtent = d3.extent(statesData, function(d) { return d.coords[0]})
      var yExtent = d3.extent(statesData, function(d) { return d.coords[1]})

      // set the dimensions and margins of the diagram
      var margin = {
          top: 20,
          right: 120,
          bottom: 20,
          left: 120
        },
        width = 800 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

      var scale_x = d3.scaleLinear()
        .range([0, width])
        .domain(xExtent),
        scale_y = d3.scaleLinear()
          .range([height, 0])
          .domain(yExtent)

      // append the svg object to the body of the page
      var svg = d3.select('#state_project')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

      // append a group for the states projection points
      var g_states = svg.append('g')
        .attr('class', 'states'),
        states = g_states
          .selectAll('circle')
          .data(statesData)
          .enter();

      var color = d3.scaleOrdinal(d3.schemeCategory10);

      var state = states.append("circle")
        .attr("cx", function(d) { return scale_x(d.coords[0]); })
        .attr("cy", function(d) { return scale_x(d.coords[1]); })
        .attr("r", radius)
        .style("fill", function(d, i) { return color(d.label); })
        .style("opacity", 0.7)
        .on("mouseover", mouseover)
        .on("mouseout", mouseout);

      state.append("title")
        .text(function(d) { return "" + d.layer + "-" + d.state_id });
          //.style({opacity:'1.0'});

      function mouseover(d) {
        d3.select(this)
          .style("stroke", "#000")
          .style("stroke", "2px")
          .style("opacity", 1.0);
      }

      function mouseout(d) {
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
