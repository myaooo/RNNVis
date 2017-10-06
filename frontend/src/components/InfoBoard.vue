<template>
  <div>
    <el-tooltip placement="top" :open-delay="500">
    <div slot="content">{{ headerText }}</div>
    <h4 class="normal">{{ header }}</h4>
    </el-tooltip>
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
  import { mapActions, mapState } from 'vuex';
  import { Chart } from '../layout/chart';
  import { SELECT_UNIT, SELECT_WORD, SELECT_LAYER, GET_STATE_STATISTICS, GET_WORD_STATISTICS } from '../store';
  // import dataServices from 'src/service/dataService.js'

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
        // width: 0,
        chart: null,
        labelBoard: null,
        layoutParams: layoutParams,
        wordsStatistics: [],
        unitsStatistics: null,
        top_k: 4,
        width: 300,
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
        return 'Info: ' + (this.$store.state.compare ? ('[' + this.selectedModel + '] ' + typeStr) : typeStr);
      },
      headerText: function() {
        switch (this.type) {
          case 'state': return "Highly sensitive words of selected hidden units";
          default: return "The distribution of model's updates on hidden states";
        }
      },
      svgId: function () {
        return this.id + '-svg';
      },
      model: function () {
        return this.compare ? this.$store.state.selectedModel2 : this.$store.state.selectedModel;
      },
      selectedLayer: function () {
        return this.model ? this.model.selectedLayer : null;
      },
      selectedState: function () {
        return this.model ? this.model.selectedState : null;
      },
      selectedUnits: function () {
        if (!this.model) return [];
        return this.type === 'state' ? (this.model.selectedUnits) : [];
      },
      selectedWords: function () {
        if (!this.model) return [];
        return this.type === 'word' ? (this.model.selectedWords) : [];
      },
      selectedNode: function() {
        if (!this.model) return null;
        return this.type === 'word' ? (this.model.selectedNode) : null;
      },
      ...mapState({
        color: 'color',
      }),
    },
    watch: {
      width: function () {
        if (this.selectedLayer && this.model && this.selectedState) {
          if (this.type === 'word' && this.wordsStatistics) this.repaintWord();
          else if (this.type === 'state' && this.unitsStatistics) this.repaintState();
        }
      },
      selectedUnits: function () {
        if (this.selectedUnits.length) {
          let model = this.model;
          this.getStateStatistics({ modelName: model.name })
            .then(() => {
              this.unitsStatistics = this.selectedUnits.map((unit, i) => {
                const data = model.stateStats.statesData[unit];
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
        if (this.selectedWords.length) {
          this.getStateStatistics({ modelName: this.model.name })
            .then(() => {
              this.wordsStatistics = this.selectedWords.map((word, i) => {
                const wordData = this.model.stateStats.statOfWord(word);
                wordData.color = this.color(i);
                return wordData;
              });
              this.repaintWord();
            });
        } else {
          this.wordsStatistics = [];
          this.repaintWord();
        }
      },
      selectedNode: function() {
        if (this.selectedNode) {
          this.getWordStatistics({ modelName: this.model.name, word: this.selectedNode.word })
            .then(wordStatistics => {
              const data = wordStatistics.data;
              data.word = data.words;
              data.range1 = data.low1.map((low1, i) => [low1, data.high1[i]]);
              data.range2 = data.low2.map((low2, i) => [low2, data.high2[i]]);
              data.color = this.color(0);
              const line = {mean: this.selectedNode.response, color: this.color(1), word: data.word + '(sentence)'};
              this.wordsStatistics = [data, line];
              // console.log(this.selectedNode);
              console.log(this.wordsStatistics);
              console.log(line);
              this.repaintWord();
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
        const color = this.color;
        const labelLength = layout.lineLength + layout.wordWidth + layout.interval;

        this.chart
          .margin(20, 30, 20, 30)
          .xAxis('dim')
          .yAxis('response');
        let sortIdx = this.wordsStatistics[0].sort_idx;
        const interval = ~~(sortIdx.length / 200) + 1;
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
            .attr('stroke', wordData.color)
            .attr('stroke-opacity', 0.6);
          if(wordData.range1){
            this.chart
              .area(sortIdx.map((i) => wordData.range1[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
              .attr('fill', wordData.color)
              .attr('fill-opacity', 0.15)
              .attr('stroke', wordData.color)
              .attr('stroke-opacity', 0.1);
          }
          if(wordData.range2){
            this.chart
              .area(sortIdx.map((i) => wordData.range2[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
              .attr('fill', wordData.color)
              .attr('fill-opacity', 0.15)
              .attr('stroke', wordData.color)
              .attr('stroke-opacity', 0.05);
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
        const xLabelWidth = 20;
        const subChartWidth = (this.width-xLabelWidth)/this.unitsStatistics.length;
        const top_k = this.top_k;

        this.unitsStatistics.forEach((unitData, i) => {
          const xLabel = i === this.unitsStatistics.length - 1 ? 'r' : ' '
          const marginRight = i === this.unitsStatistics.length - 1 ? xLabelWidth : 0;
          const subWidth = i === this.unitsStatistics.length - 1 ? subChartWidth + xLabelWidth : subChartWidth;
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
            .margin(20,marginRight,20,55)
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

      ...mapActions({
        getStateStatistics: GET_STATE_STATISTICS,
        getWordStatistics: GET_WORD_STATISTICS,
      }),
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
