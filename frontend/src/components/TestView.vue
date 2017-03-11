<template>
  <div>
    <svg :id="svgId"> </svg>
  </div>
</template>
<style>

</style>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import {bus, SELECT_MODEL} from 'event-bus.js';
  import {Chart} from '../layout/chart.js'

  export default {
    name: 'TestView',
    data() {
      return {
        chart: null,
      };
    },
    props: {},
    computed: {
      svgId: function(){
        return 'svg-test';
      }
    },
    methods: {
      draw() {
        dataService.getWordStatistics('PTB-LSTM', 'state_c', -1, 'he', response => {
          const data = response.data;
          const mean = data.sort_idx.map((d, i) => { return data.mean[d]; });
          const range  = data.sort_idx.map((d, i) => { return [data.low1[d], data.high1[d]]; });
          const range2  = data.sort_idx.map((d, i) => { return [data.low2[d], data.high2[d]]; });

          console.log(data);
          const svg = d3.select(`#${this.svgId}`)
            .attr('width', 500)
            .attr('height', 200);
          this.chart = new Chart(svg, 500, 200)
            .background('lightgray', 0.3);
          const subchart = this.chart.subChart(250, 200)
            .margin(5,5,20,30)
            .xAxis()
            .yAxis();
          subchart.line(mean, (d, i) => { return i; }, (d) => { return d; });
          subchart.area(range, (d, i) => i, (d) => d[0], (d) => d[1])
            .attr('opacity', 0.4);
          subchart.area(range2, (d, i) => i, (d) => d[0], (d) => d[1])
            .attr('opacity', 0.2);
          // subchart.draw();

          const boxData = mean.map((m, i) => {
            return {mean: m, range1: range[i], range2: range2[i]};
          })
          const subchart2 = this.chart.subChart(250, 200)
            .translate(250, 0)
            .margin(5, 30, 20, 5)
            .xAxis()
            .yAxis('right')
            .rotate()
            .rotate();
          // subchart2
          //   .group.attr('transform', 'rotate(90)');
          subchart2.box(boxData.slice(0,10).concat(boxData.slice(boxData.length-10)), 5, (d, i) => i, (d) => d.mean, (d) => d.range1, (d) => d.range2)
            .attr('fill', 'steelblue')
            .attr('stroke', 'gray')
            .attr('fill-opacity', 0.5);
          this.chart.draw();

            // .line([[0.1, 0.1], [0.3, 0.8], [0.9,0.9]]);

        });
      }
    },
    mounted() {
      this.draw();
    }

  }

</script>
