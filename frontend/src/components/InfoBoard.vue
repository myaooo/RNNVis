<template>
  <div>
    <h4 class="normal">{{ header }}</h4>
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
  import dataServices from '../services/dataService.js'

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
        wordsStatistics: null,
        unitsStatistics: null,
        top_k: 4,
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
      header: function() {
        const typeStr = this.type === 'state' ? 'Hidden State' : 'Word';
        return 'Info: ' + (this.shared.compare ? ('[' + this.selectedModel + '] ' + typeStr) : typeStr);
      },
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
        return this.type === 'word' ? (this.compare ? this.shared.selectedWords2 : this.shared.selectedWords) : 0;
      },
      selectedNode: function() {
        return this.type === 'word' ? (this.compare ? this.shared.selectedNode2 : this.shared.selectedNode) : 0;
      }
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
              this.unitsStatistics = this.selectedUnits.map((unit, i) => {
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
                dataArray.splice(this.top_k, dataArray.length - 2 * this.top_k);
                return dataArray;
              });
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
              this.wordsStatistics = this.selectedWords.map((word, i) => {
                const wordData = this.statistics.statOfWord(word.text);
                wordData.color = this.selectedWords[i].color;
                return wordData;
              });
              // console.log(this.statistics);
              this.repaintWord();
              // const wordsStatistics = this.statistics.statOfWord(this.selectedWords[0]).mean;
            });
        }
      },
      selectedNode: function() {
        if (this.selectedNode) {
          let model = this.selectedModel,
            state = this.selectedState,
            layer = this.selectedLayer;
          const p = dataServices.getWordStatistics(model, state, layer, this.selectedNode.word, (response) => {
            if (response.status === 200) {
              const data = response.data;
              data.word = data.words;
              data.range1 = data.low1.map((low1, i) => [low1, data.high1[i]]);
              data.range2 = data.low2.map((low2, i) => [low2, data.high2[i]]);
              data.color = this.layoutParams.color(0);
              const line = {mean: this.selectedNode.response, color: this.layoutParams.color(1), word: data.word + '(sentence)'};
              this.wordsStatistics = [data, line];
              console.log(this.selectedNode);
              this.repaintWord();
            }
          });
        }
      },
      compare: function () {
        this.init();
      },
      type: function () {
        this.init();
      }
    },
    methods: {
      init() {
        if (this.chart) {
          this.chart.clean();
        }
        if (this.labelBoard) {
          this.labelBoard.selectAll('rect, text, path').remove();
        }
        this.chart = new Chart(d3.select(`#${this.svgId}`), this.width, this.height)
          .background('lightgray', 0.0);
        this.labelBoard = d3.select(`#${this.svgId}2`);
      },
      repaintWord() {
        this.chart.clean()
          .resize(this.width, this.height);
        this.labelBoard.selectAll('rect, text, path').remove();
        if (!this.wordsStatistics.length){
          console.log('Painting no words');
          return;
        }
        const layout = this.layoutParams;
        const color = layout.color;
        const labelLength = layout.lineLength + layout.wordWidth + layout.interval;

        this.chart
          .margin(20, 40, 20, 30)
          .xAxis('dims')
          .yAxis('response');
        let sortIdx = this.wordsStatistics[0].sort_idx;
        const interval = ~~(sortIdx.length / 200)
        const ranges = range(0, sortIdx.length, interval);
        sortIdx = ranges.map((i) => sortIdx[i]);
        // console.log(range);
        // console.log(sortIdx);
        this.chart.line([[0,0], [this.wordsStatistics[0].mean.length,0]])
          .attr('stroke', '#000');

        this.wordsStatistics.forEach((wordData, i) => {
          this.chart
            .line(sortIdx.map((i) => wordData.mean[i]), (d, i) => i*interval, (d) => { return d; })
            .attr('stroke-width', 1)
            .attr('stroke', wordData.color);
          if(wordData.range1){
            this.chart
              .area(sortIdx.map((i) => wordData.range1[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
              .attr('fill', wordData.color)
              .attr('fill-opacity', 0.2);
          }
          if(wordData.range2){
            this.chart
              .area(sortIdx.map((i) => wordData.range2[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
              .attr('fill', wordData.color)
              .attr('fill-opacity', 0.1);
          }
          // draw labels
          this.labelBoard.append('rect')
            .attr('x', labelLength*i + 20).attr('y', 10).attr('width', layout.lineLength).attr('height', 1)
            .attr('fill', wordData.color);
          this.labelBoard.append('text')
            .attr('x', labelLength*i + 30 + layout.lineLength).attr('y', 15)
            .text(wordData.word)
            .style('font-size', 12);
          if (i === 0) {
            this.labelBoard.append('path')
              .attr('d', 'M' + (30 + layout.lineLength) + ' 20 ' + 'H ' + (wordData.word.length*7 + 30 + layout.lineLength))
              .style('stroke', wordData.color);
          }
        });
        this.chart.draw();

      },
      repaintState() {
        this.chart.clean()
          .resize(this.width, this.height);
        this.labelBoard.selectAll('path, text').remove();
        if (!this.selectedUnits.length){
          console.log('Painting no states');
          return;
        }
        const subChartWidth = (this.width-50)/this.unitsStatistics.length;
        const top_k = this.top_k;

        this.unitsStatistics.forEach((unitData, i) => {
          const xLabel = i === this.unitsStatistics.length - 1 ? 'response' : ' '
          const marginRight = i === this.unitsStatistics.length - 1 ? 60 : 10;
          const subWidth = i === this.unitsStatistics.length - 1 ? subChartWidth + 50 : subChartWidth;
          const subchart = this.chart.subChart(subWidth, this.height)
            .xAxis(xLabel)
            .yAxis('words');
          subchart.axis.y.tickFormat((j) => {
            // console.log(j);
            if (-1 < j && j < top_k * 2)
              return unitData[j].word;
          }).tickValues(range(0,top_k*2,1));
          subchart.axis.x.ticks(6);
          subchart
            .margin(20,marginRight,20,60)
            .translate(subChartWidth*i, 0)
            .rotate();
          subchart
            .box(unitData, 6, (d, j) => j, (d) => d.mean, (d) => d.range1, (d) => d.range2)
            .attr('fill', 'steelblue')
            .attr('stroke', 'black')
            .attr('stroke-width', 0)
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
      this.width = this.$el.clientWidth - 10;
      this.init();
      // register event listener
      // this.register();

      // test event
      // bus.$on(SELECT_LAYER, () => {
      //   setTimeout(() => {
      //     if (this.type === 'word')
      //       bus.$emit(SELECT_UNIT, 10, false);
      //     // if (this.type === 'state')
      //     //   bus.$emit(SELECT_WORD, 'he', false);
      //   }, 1000);
      //   setTimeout(() => {
      //     if (this.type === 'word')
      //       bus.$emit(SELECT_UNIT, 20, false);
      //     // if (this.type === 'state')
      //     //   bus.$emit(SELECT_WORD, 'she', false);
      //   }, 4000);

      // });

    }
  }

  function getSortedStatesData(words) {

  }

  function range(start, end, interval = 1) {
    const num = ~~((end - start -1) / interval) + 1;
    return Array.from({length: num}, (v, i) => start + i * interval);
  }
</script>
