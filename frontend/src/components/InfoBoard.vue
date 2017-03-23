<template>
  <div>
    <h4 class="normal">{{type === 'state' ? 'Hidden State' : 'Word'}}</h4>
    <hr>
    <svg :id='svgId' :width='width' :height='height'> </svg>
    <svg :id='svgId+"2"' :width='width' :height='30'> </svg>
  </div>
</template>

<style>
  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>

<script>
  import * as d3 from 'd3';
  import { Chart } from '../layout/chart';
  import { bus, SELECT_UNIT, SELECT_WORD, SELECT_LAYER } from '../event-bus';

  const layoutParams = {
    lineLength: 30,
    wordWidth: 50,
    interval: 20,
    color: d3.scaleOrdinal(d3.schemeCategory10),
  };

  export default {
    name: 'InfoBoard',
    data(){
      return {
        width: 0,
        chart: null,
        labelBoard: null,
        shared: bus.state,
        statistics: null,
        layoutParams: layoutParams,
      };
    },
    props: {
      height: {
        type: Number,
        default: 600,
      },
      id: {
        type: String,
        default: 'info-board',
      },
      type: {
        type: String,
        default: 'state',
      },
      compare: {
        type: Boolean,
        default: false,
      },
    },
    computed: {
      svgId: function () {
        return this.id + '-svg';
      },
      selectedLayer: function () {
        return this.compare ? this.shared.selectedLayer2 : this.shared.selectedLayer;
      },
      selectedModel: function () {
        return this.compare ? this.shared.selectedModel2 : this.shared.selectedModel;
      },
      selectedState: function () {
        return this.compare ? this.shared.selectedState2 : this.shared.selectedState;
      },
      selectedUnits: function () {
        return this.type === 'state' ? (this.compare ? this.shared.selectedUnits2 : this.shared.selectedUnits) : 0;
      },
      selectedWords: function () {
        return this.type === 'word' ? (this.comapre ? this.shared.selectedWords2 : this.shared.selectedWords) : 0;
      },
    },
    watch: {
      width: function () {
        if (this.selectedLayer && this.selectedModel && this.selectedState) {
          if (this.type === 'word' && this.selectedWords) this.repaintWord();
          else if (this.type === 'state' && this.selectedUnits) this.repaintState();
        }
      },
      selectedUnits: function () {
        if (this.selectedUnits) {
          let model = this.selectedModel,
            state = this.selectedState,
            layer = this.selectedLayer;
          const p = bus.loadStatistics(model, state, layer)
            .then(() => {
              this.statistics = bus.getStatistics(model, state, layer);
              this.repaintState();
            });
        }
      },
      selectedWords: function() {
        if (this.selectedWords) {
          let model = this.selectedModel,
            state = this.selectedState,
            layer = this.selectedLayer;
          const p = bus.loadStatistics(model, state, layer)
            .then(() => {
              this.statistics = bus.getStatistics(model, state, layer);
              // console.log(this.statistics);
              this.repaintWord();
              // const wordsStatistics = this.statistics.statOfWord(this.selectedWords[0]).mean;
            });
        }
      }
    },
    methods: {
      init() {
        this.chart = new Chart(d3.select(`#${this.svgId}`), this.width, this.height)
          .background('lightgray', 0.0);
        this.labelBoard = d3.select(`#${this.svgId}2`);
      },
      repaintWord() {
        this.chart.clean();
        this.labelBoard.selectAll('rect, text, path').remove();
        // console.log(this.statistics);
        if (!this.selectedWords.length){
          console.log('Painting no words');
          return;
        }
        const layout = this.layoutParams;
        const color = layout.color;
        const labelLength = layout.lineLength + layout.wordWidth + layout.interval;
        const wordsStatistics = this.selectedWords.map((word, i) => {
          return this.statistics.statOfWord(word.text);
        });
        this.chart
          .margin(5, 5, 20, 30)
          .xAxis()
          .yAxis();
        let sortIdx = wordsStatistics[0].sort_idx;
        const interval = ~~(sortIdx.length / 100)
        const ranges = range(0, sortIdx.length, interval);
        sortIdx = ranges.map((i) => sortIdx[i]);
        // console.log(range);
        // console.log(sortIdx);
        this.chart.line([[0,0], [wordsStatistics[0].mean.length,0]])
          .attr('stroke', '#000');

        wordsStatistics.forEach((wordData, i) => {
          this.chart
            .line(sortIdx.map((i) => wordData.mean[i]), (d, i) => i*interval, (d) => { return d; })
            .attr('stroke-width', 1)
            .attr('stroke', this.selectedWords[i].color);
          this.chart
            .area(sortIdx.map((i) => wordData.range1[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
            .attr('fill', this.selectedWords[i].color)
            .attr('fill-opacity', 0.2);
          this.chart
            .area(sortIdx.map((i) => wordData.range2[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
            .attr('fill', this.selectedWords[i].color)
            .attr('fill-opacity', 0.1);
          // draw labels
          this.labelBoard.append('rect')
            .attr('x', labelLength*i + 20).attr('y', 10).attr('width', layout.lineLength).attr('height', 1)
            .attr('fill', this.selectedWords[i].color);
          this.labelBoard.append('text')
            .attr('x', labelLength*i + 30 + layout.lineLength).attr('y', 15)
            .text(this.selectedWords[i].text)
            .style('font-size', 12);
          if (i === 0) {
            this.labelBoard.append('path')
              .attr('d', 'M' + (30 + layout.lineLength) + ' 20 ' + 'H ' + (this.selectedWords[i].text.length*7 + 30 + layout.lineLength))
              .style('stroke', this.selectedWords[i].color);
          }
        });
        this.chart.draw();

      },
      repaintState() {
        this.chart.clean();
        if (!this.selectedUnits.length){
          console.log('Painting no words');
          return;
        }
        const top_k = 8;
        // const units = this.selectedUnits;
        const unitsStatistics = this.selectedUnits.map((unit, i) => {
          const data = this.statistics.statesData[unit];
          const dataArray = data.mean.map((_, j) => {
            return {
              mean: data.mean[j],
              range1: [data.low1[j], data.high1[j]],
              range2: [data.low2[j], data.high2[j]],
              word: data.words[j],
            };
          });

          dataArray.sort((a, b) => a.mean - b.mean);
          dataArray.splice(top_k, dataArray.length - 2 * top_k);
          return dataArray;
        });
        // console.log(unitsStatistics);
        const subChartWidth = this.width/3;

        this.labelBoard.selectAll('path, text').remove();
        unitsStatistics.forEach((unitData, i) => {
          const subchart = this.chart.subChart(subChartWidth, this.height)
            .xAxis()
            .yAxis();
          subchart.axis.y.tickFormat((j) => {
            // console.log(j);
            if (-1 < j && j < top_k * 2)
              return unitData[j].word;
          }).tickValues(range(0,20,1));
          subchart.axis.x.ticks(7);
          subchart
            .margin(10,10,20,60)
            .translate(subChartWidth*i, 0)
            .rotate();
          subchart
            .box(unitData, 6, (d, j) => j, (d) => d.mean, (d) => d.range1, (d) => d.range2)
            .attr('fill', 'steelblue')
            .attr('stroke', 'gray')
            .attr('fill-opacity', 0.5);
          // horizontal line
          subchart.line([[top_k-0.5, subchart.extents[1][0]], [top_k-0.5, subchart.extents[1][1]]])
            .style('stroke', '#333').style('stroke-dasharray', '5 5').style('stroke-opacity', 0.6);
          const labelPos = subChartWidth*i + subChartWidth/2 - 6;
          this.labelBoard.append('text')
            .attr('x', labelPos).attr('y', 15)
            .text('Dim: ' + this.selectedUnits[i])
            .style('font-size', 12);
          if (i === 0){
            this.labelBoard.append('path')
              .attr('d', 'M' + labelPos + ' 20 ' + 'H ' + (this.selectedUnits[0].toString().length*6 + 30 + labelPos))
              .style('stroke', '#a36');
          }
        })
        this.chart.draw();
      },

    },
    mounted() {
      // console.log(this.$el);
      this.width = this.$el.clientWidth;
      this.init();
      // register event listener
      // this.register();

      // test event
      bus.$on(SELECT_LAYER, () => {
        setTimeout(() => {
          if (this.type === 'word')
            bus.$emit(SELECT_UNIT, 10, false);
          // if (this.type === 'state')
          //   bus.$emit(SELECT_WORD, 'he', false);
        }, 1000);
        setTimeout(() => {
          if (this.type === 'word')
            bus.$emit(SELECT_UNIT, 20, false);
          // if (this.type === 'state')
          //   bus.$emit(SELECT_WORD, 'she', false);
        }, 4000);

      });

    }
  }

  function getSortedStatesData(words) {

  }

  function range(start, end, interval = 1) {
    const num = ~~((end - start -1) / interval) + 1;
    return Array.from({length: num}, (v, i) => start + i * interval);
  }
</script>
