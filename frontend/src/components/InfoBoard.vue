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

  export default {
    name: 'InfoBoard',
    data(){
      return {
        width: 0,
        chart: null,
        labelBoard: null,
        shared: bus.state,
        selectedUnits: null, // []
        selectedWords: null, // []
        compare: null,
        statistics: null,
        color: d3.scaleOrdinal(d3.schemeCategory10),
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
      }
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
      // width: function () {

      // }
    },
    watch: {
      width: function () {
        if (this.selectedLayer && this.selectedModel && this.selectedState) {
          if (this.type === 'word' && this.selectedWords) this.repaintWord();
          else if (this.type === 'state' && this.selectedState) this.repaintState();
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
        const wordsStatistics = this.selectedWords.map((word, i) => {
          return this.statistics.statOfWord(word);
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
        this.labelBoard.selectAll('rect, text').remove();
        wordsStatistics.forEach((wordData, i) => {
          this.chart
            .line(sortIdx.map((i) => wordData.mean[i]), (d, i) => i*interval, (d) => { return d; })
            .attr('stroke-width', 1)
            .attr('stroke', this.color(i));
          this.chart
            .area(sortIdx.map((i) => wordData.range1[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
            .attr('fill', this.color(i))
            .attr('fill-opacity', 0.2);
          this.chart
            .area(sortIdx.map((i) => wordData.range2[i]), (d, i) => i*interval, (d) => d[0], (d) => d[1])
            .attr('fill', this.color(i))
            .attr('fill-opacity', 0.1);
          this.labelBoard.append('rect')
            .attr('x', 80*i+20).attr('y', 10).attr('width', 30).attr('height', 1)
            .attr('fill', this.color(i))
          this.labelBoard.append('text')
            .attr('x', 80*i + 60).attr('y', 15)
            .text(this.selectedWords[i])
        });
        this.chart.draw();

      },
      repaintState() {

      },
      register() {
        if(this.type === 'state') {
          bus.$on(SELECT_UNIT, (unitDims, compare) => {
            this.selectedUnits = unitDims;
            this.compare = compare;
            let model = this.selectedModel,
              state = this.selectedState,
              layer = this.selectedLayer;
            const p = bus.loadStatistics(model, state, layer)
              .then(() => {
                this.statistics = bus.getStatistics(model, state, layer);
                this.repaintState();
              });
          });
        } else if (this.type === 'word') {
          bus.$on(SELECT_WORD, (words, compare) => {
            this.selectedWords = words.slice();
            this.compare = compare;
            let model = this.selectedModel,
              state = this.selectedState,
              layer = this.selectedLayer;
            const p = bus.loadStatistics(model, state, layer)
              .then(() => {
                this.statistics = bus.getStatistics(model, state, layer);
                this.repaintWord();
              });
          });
        }
      }
    },
    mounted() {
      // console.log(this.$el);
      this.width = this.$el.clientWidth;
      this.init();
      // register event listener
      this.register();

      // test event
      bus.$on(SELECT_LAYER, () => {
        setTimeout(() => {
          if (this.type === 'word')
            bus.$emit(SELECT_UNIT, [10, 20], false);
          if (this.type === 'state')
            bus.$emit(SELECT_WORD, ['he', 'she'], false);
        }, 5000);
        setTimeout(() => {
          if (this.type === 'word')
            bus.$emit(SELECT_UNIT, [10, 20], false);
          if (this.type === 'state')
            bus.$emit(SELECT_WORD, ['he', 'for'], false);
        }, 8000);

      });

    }
  }

  function range(start, end, interval) {
    const num = ~~((end - start -1) / interval) + 1;
    return Array.from({length: num}, (v, i) => start + i * interval);
  }
</script>
