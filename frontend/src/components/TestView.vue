<template>
  <div>
    <svg :id='svgId'> </svg>
  </div>
</template>
<style>

</style>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import { bus, SELECT_MODEL } from 'event-bus.js';
  import { Chart } from '../layout/chart.js'
  import { WordCloud } from '../layout/cloud.js'
  import { sentence } from '../layout/sentence.js'

  export default {
    name: 'TestView',
    data() {
      return {
        chart: null,
        model: 'PTB-LSTM',
        state: 'state_c',
      };
    },
    props: {},
    computed: {
      svgId: function () {
        return 'svg-test';
      }
    },
    methods: {
      init() {
        const svg = d3.select(`#${this.svgId}`)
          .attr('width', 500)
          .attr('height', 500);
        this.chart = new Chart(svg, 500, 500)
          .background('lightgray', 0.0);
      },
      draw2() {
        var words = ['you', 'want', 'more',
          'words', 'than', 'this', 'he', 'is', 'she', 'they', 'what', 'million',
          'it', '$', '<unk>', 'good', 'i', 'by']
          .map(function (d, i) {
            return { text: d, size: 5 + Math.random()*25, type: 0 + Math.round(Math.random()) };
          });
        //This method tells the word cloud to redraw with a new set of words.
        //In reality the new words would probably come from a server request,
        // user input or some other source.
        function showNewWords(vis, i) {
          i = i || 0;

          vis.update(words)
          setTimeout(function () { showNewWords(vis, i + 1) }, 2000)
        }

        //Create a new instance of the word cloud visualisation.
        var myWordCloud = new WordCloud(d3.select(`#${this.svgId}`), 120, 120).transform('translate(200,200)');
        // myWordCloud.update(words);

        //Start cycling through the demo data
        showNewWords(myWordCloud);
        // myWordCloud.update(words);
      },
      draw() {
        dataService.getWordStatistics(this.model, this.state, -1, 'he', response => {
          const data = response.data;
          const mean = data.sort_idx.map((d, i) => { return data.mean[d]; });
          const range = data.sort_idx.map((d, i) => { return [data.low1[d], data.high1[d]]; });
          const range2 = data.sort_idx.map((d, i) => { return [data.low2[d], data.high2[d]]; });

          console.log(data);
          const subchart = this.chart.subChart(250, 200)
            .margin(5, 5, 20, 30)
            .xAxis()
            .yAxis();
          subchart.line(mean, (d, i) => { return i; }, (d) => { return d; });
          subchart.area(range, (d, i) => i, (d) => d[0], (d) => d[1])
            .attr('opacity', 0.4);
          subchart.area(range2, (d, i) => i, (d) => d[0], (d) => d[1])
            .attr('opacity', 0.2);
          // subchart.draw();
        });
        let processed;
        const p1 = dataService.getStateStatistics(this.model, this.state, -1, 200, response => {
          const data = response.data;
          // console.log(data);
          processed = data.mean[0].map((_, i) => {
            return {
              freqs: data.freqs,
              mean: data.mean.map((m) => m[i]),
              low1: data.low1.map((m) => m[i]),
              low2: data.low2.map((m) => m[i]),
              high1: data.high1.map((m) => m[i]),
              high2: data.high2.map((m) => m[i]),
              rank: data.sort_idx.map((indice) => {
                return indice.findIndex((idx) => (idx === i));
              })
            };
          });
        });
        let vocab;
        const p2 = dataService.getVocab(this.model, 200, response => {
          vocab = response.data;
        })
        Promise.all([p1, p2]).then(() => {
          console.log(processed);
          console.log(vocab);
          const dim = 10;
          let data = processed[dim];
          data = data.freqs.map((f, i) => {
            return {
              freqs: f,
              mean: data.mean[i],
              range1: [data.low1[i], data.high1[i]],
              range2: [data.low2[i], data.high2[i]],
              rank: data.rank[i],
              word_id: i,
              word: vocab[i],
            };
          })
          const boxData = data.sort((a, b) => {
            return a.mean - b.mean;
          })
          // const boxData = mean.map((m, i) => {
          //   return {mean: m, range1: range[i], range2: range2[i], idx: data.sort_idx[i]};
          // });
          const boxes = boxData.slice(0, 10).concat(boxData.slice(boxData.length - 10));
          const subchart2 = this.chart.subChart(250, 200)
            .translate(250, 0)
            .margin(5, 30, 40, 5)
            .xAxis()
            .yAxis('right');
          subchart2.axis.x.tickFormat((d, i) => {
            return boxes[d].word;
          }).ticks(20);
          //   .group.attr('transform', 'rotate(90)');
          subchart2.box(boxes, 5, (d, i) => i, (d) => d.mean, (d) => d.range1, (d) => d.range2)
            .attr('fill', 'steelblue')
            .attr('stroke', 'gray')
            .attr('fill-opacity', 0.5);
          this.chart.draw();

          // formatting x axis labels
          subchart2.axisHandles.x.selectAll('text')
            .attr('y', 0)
            .attr('x', -9)
            .attr('dy', '.35em')
            .attr('transform', 'rotate(-90)')
            .style('text-anchor', 'end');
          // subchart2.axis.x.ticks(10);

          // .line([[0.1, 0.1], [0.3, 0.8], [0.9,0.9]]);

        });
      },
      draw_arc() {
        let color = d3.scaleOrdinal(d3.schemeCategory10);
        let width = 500, height = 500;
        const svg = d3.select(`#${this.svgId}`)
          .attr('width', width)
          .attr('height', height);

        let data = [
          { start: 0, end: 2 * Math.PI / 8},
          { start: 2 * Math.PI / 8, end: 2 * Math.PI / 2},
          { start: 2 * Math.PI / 2, end: 2 * Math.PI},
        ];

        let g = svg.append('g')
          .attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')');
        var arc = d3.arc()
          .innerRadius(230)
          .outerRadius(240)
          // .startAngle(0)

        data.forEach((d, i) => {
          g.append('path')
          // .attr('transform', 'rotate(' + 180 + ')')
            .datum({startAngle: d.start, endAngle: d.end})
            .style('fill', color(i))
            .attr('d', arc)
            // .attr('cent', arc.centroid())
          console.log(arc.centroid({startAngle: d.start, endAngle: d.end}));
        })

      },
      draw4() {
        let color = d3.scaleOrdinal(d3.schemeCategory10);
        let width = 500, height = 500;
        const rectHeight = 20;
        const rectWidth = 50;
        const rectInterval = 10;
        const data = ['this', 'is', 'foy', 'test', '<eos>'];
        
        var x = d3.scaleTime()
          .domain([new Date(2013, 7, 1), new Date(2013, 7, 15) - 1])
          .rangeRound([0, width]);
        const g = d3.select(`#${this.svgId}`)
          .attr('width', width)
          .attr('height', height)
          .append('g')
          .attr("transform", "translate(100," + height / 3  + ")");

        let texts = g
          .selectAll('text')
          .data(data).enter()
          .append('text')
          .text((d) => {return d})
          .style('text-anchor', 'middle')
          // .attr('font-size', 12)
          .attr('transform', function(d, i) {
            return 'translate(' + [(rectWidth + rectInterval) * i + rectWidth / 2, rectHeight/1.3] + ')';
          });
        
        const rect_g = g.append('g');
        data.forEach((d, i) => {
          if (i > 0) {
            rect_g.append('rect')
              .attr('x', i * (rectWidth + rectInterval) - rectInterval)
              .attr('y', '0')
              .attr('width', rectInterval)
              .attr('height', rectHeight)
              .attr('fill', 'lightgray')
              .attr('stroke-width', 2)
              .attr('stroke', 'blue')
              .attr('opacity', 0.2);
          }
          rect_g
            .append('rect')
            .attr('x', i * (rectWidth + rectInterval))
            .attr('y', '0')
            .attr('width', rectWidth)
            .attr('height', rectHeight)
            .attr('fill', 'lightgray')
            .attr('stroke-width', 2)
            .attr('stroke', 'blue')
            .attr('opacity', 0.2);
        });
        
        g.append('g')
          .call(d3.brushX()
              .extent([[0, 0], [(rectWidth + rectInterval) * data.length - rectInterval, rectHeight]])
              .on('end', brushended)
          );
        
        function brushended () {
          if (!d3.event.sourceEvent) return;
          if (!d3.event.selection) return;
          let extent = d3.event.selection;
          extent[0] = Math.round(extent[0] / (rectWidth + rectInterval)) * (rectWidth + rectInterval);
          extent[1] = Math.max(Math.round(extent[1] / (rectWidth + rectInterval)) * (rectWidth + rectInterval) - rectInterval, extent[0]);
          console.log(d3.event.selection);
          d3.select(this).transition().call(d3.event.target.move, extent)
        }
      },

      drawRect(g, dataLength, func) {
        const self = this;
        const rectSize = [20, 50];
        const minBrushLength = 3;
        g.selectAll('.wordRect')
          .data(d3.range(dataLength)).enter()
          .append('rect')
          .attr('x', 0)
          .attr('y', d => d * rectSize[1])
          .attr('width', rectSize[0])
          .attr('height', rectSize[1])
          .attr('fill', 'lightgray')
          .attr('stroke-width', 2)
          .attr('stroke', 'blue')
          .attr('opacity', 0.2);
        let brush = d3.brushY()
          .extent([[0, 0], [rectSize[0], rectSize[1] * dataLength]])
          .on('end', function() {
            if (!d3.event.sourceEvent) return;
            if (!d3.event.selection) return;
            let extent = d3.event.selection;
            extent[0] = Math.round(extent[0] / rectSize[1]) * rectSize[1];
            extent[1] = Math.round(extent[1] / rectSize[1]) * rectSize[1];
            d3.select(this).transition().call(d3.event.target.move, extent);
            console.log(d3.event.selection);
            func([Math.round(extent[0] / rectSize[1]), Math.round(extent[1] / rectSize[1])])
          });
        g.append('g')
          .call(brush)
          .call(brush.move, [0, rectSize[1] * dataLength])
      },

      draw5() {
        const p1 = bus.loadCoCluster('PTB-LSTM', 'state_c', 10, {top_k: 300, mode: 'raw'});
        const record = bus.evalSentence('What can I do for you?', 'PTB-LSTM');
        const p2 = record.evaluate();
        const clientHeight = 500;
        let sent = null;
        const svg = d3.select(`#${this.svgId}`);
        const sent_g = svg.append('g');
        
        Promise.all([p1, p2]).then((values) => {
          const coCluster = bus.getCoCluster('PTB-LSTM', 'state_c', 10, {top_k: 300, mode: 'raw'});
          const sentenceRecord = record.getRecords('state_c', -1);
          console.log(record);
          const rect_g = svg.append('g')
            .attr('transform', 'translate(0, 50)');
          sent = sentence(sent_g)
            .transform('translate(100, 0)')
            .size([50, clientHeight])
            .sentence(sentenceRecord)
            .coCluster(coCluster)
            .words(record.tokens)
            .draw()
            // .transform('scale('  + scaleFactor + ')translate(' + [200/scaleFactor,10/scaleFactor] + ')');;
          console.log(sent.strengthByCluster);
          this.drawRect(rect_g, record.tokens.length, updateSentence);
      });

      function updateSentence(extent_) {
        console.log(`extent_ is ${extent_}`);
        const words = record.tokens.slice(...extent_);
        console.log(`words is ${words}`);
        let scaleFactor = record.tokens.length / words.length;
        scaleFactor = Math.min(scaleFactor, 2);
        const newHeight = scaleFactor * clientHeight;
        let translateY = sent.getWordPos(extent_[0])[1];
        sent.transform('scale('  + scaleFactor + ')translate(' + [100/scaleFactor, -translateY] + ')');
      }
    },
    
    draw3() {
      const p1 = bus.loadCoCluster('PTB-LSTM', 'state_c', 10, {top_k: 300, mode: 'raw'});
      const record = bus.evalSentence('What can I do for you?', 'PTB-LSTM');
      const p2 = record.evaluate();
      Promise.all([p1, p2]).then((values) => {
        const coCluster = bus.getCoCluster('PTB-LSTM', 'state_c', 10, {top_k: 300, mode: 'raw'});
        const sentenceRecord = record.getRecords('state_c', -1);
        console.log(record);
        const a = sentence(d3.select(`#${this.svgId}`).append('g'))
          .transform('translate(100, 10)')
          .size([50, 450])
          .sentence(sentenceRecord)
          .coCluster(coCluster)
          .words(record.tokens)
          .draw();
        console.log(a.strengthByCluster);
      })


        // .layout();
    },
  },
    mounted() {
      // let coClusterData;
      // const p = dataService.getCoCluster(this.model, this.state, 10, {}, response => {
      //   coClusterData = response.data;
      //   // console.log('co-cluster data:');
      //   // console.log(coClusterData);
      // })
      this.init();
      // this.draw2();
      // this.draw_arc();
      this.draw5();
      // this.draw4();
    }

  }

</script>
