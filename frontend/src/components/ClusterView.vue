<style>
:root {
  --positive-color: '#ff5b09';
  --negative-color: '#09adff';
}
.hidden-cluster {
  stroke: black;
  stroke-opacity: 0.2;
  stroke-width: 1;
  fill: gray;
  fill-opacity: 0.1;
}
.hidden-cluster.active {
  stroke-width: 1.5;
  stroke-opacity: 0.7;
  stroke: black;
  fill-opacity: 0.1;
}
.active > .hidden-cluster{
  stroke-width: 1.5;
  stroke-opacity: 0.7;
  stroke: black;
  fill-opacity: 0.1;
}
#middle_line {
  stroke: lightskyblue;
  stroke-width: 2;
  stroke-dasharray: 3, 3;
  fill: none;
  opacity: 0.5;
}
.link.active {
  opacity: 1;
}
.link {
  fill: none;
  opacity: 0.2;
}
.link.positive {
  stroke: #ff5b09;
}
.link.negative {
  stroke: #09adff;
}
.unit {
  stroke: 'none';
}
.unit-active {
  stroke: black;
  stroke-width: 1.0;
}
.wordcloud.active {
  stroke: 'black';
  stroke-width: 1.5;
  stroke-opacity: 0.7;
}

.axis path, .axis tick, .axis line {
  fill: none;
  stroke: none;
}

</style>
<template>

    <svg :id='svgId' :width='width' :height='height'>
      <defs>
        <linearGradient id="state-legend">
          <stop class="stop1" offset="0%" stop-color='rgba(9, 173, 255, 1)'/>
          <stop class="stop2" offset="50%" stop-color='rgba(128, 128, 128, 0.1)'/>
          <stop class="stop3" offset="100%" stop-color='rgba(255, 91, 9, 1)'/>
        </linearGradient>
      </defs>
    </svg>
  <!--</div>-->
</template>

<script>
  import * as d3 from 'd3';
  import { mapState, mapActions, mapMutations } from 'vuex';
  import {
    SELECT_MODEL,
    SELECT_STATE,
    CHANGE_LAYOUT,
    EVALUATE_SENTENCE,
    SELECT_UNIT,
    DESELECT_UNIT,
    CLOSE_SENTENCE,
    SELECT_SENTENCE_NODE,
    SELECT_COLOR,
    GET_STATE_STATISTICS,
    RENDER_GRAPH,
    GRAPH_RENDERED,
  } from '../store';
  // import { WordCloud } from '../layout/cloud.js';
  // import { sentence } from '../layout/sentence.js';
  import {
    LayoutParamsConstructor,
    Painter
  } from '../layout/cocluster';

  const colorHex = ['#33a02c', '#1f78b4', '#b15928', '#fb9a99', '#e31a1c', '#6a3d9a', '#ff7f00', '#cab2d6', '#ffff99', '#a6cee3', '#b2df8a', '#fdbf6f'];
  const colorScheme = (i) => colorHex[i];
  const positiveColor = '#ff5b09';
  const negativeColor = '#09adff';

  const pos2tag = {
    "VERB": 0,
    "NOUN": 1,
    "PRON": 2,
    "ADJ": 3,
    "ADV": 4,
    "ADP": 5,
    "CONJ": 6,
    "DET": 7,
    "NUM": 8,
    "PRT": 9,
    "X": 10,
    ".": 11,
  };

  const labelParams = {
    colorScheme: colorScheme,
    radius: 4,
    fontSize: 11,
    interval: 8
  }

  class PosLabel {
    constructor(selector, params, compare = False) {
      this.g = selector;
      this.params = params;
      this.compare = compare;
    }
    draw(tags) {
      const params = this.params;
      const color = params.colorScheme;
      const fontSize = params.fontSize;
      const radius = params.radius;
      const interval = params.interval;
      const labels = [];
      Object.keys(tags).forEach((key) => {
        labels[tags[key]] = key;
      });
      const gs = this.g.selectAll('g')
        .data(labels).enter()
        .append('g');
      gs.append('circle')
        .attr('cx', 0).attr('cy', (d, i) => i * (interval + 2 * radius))
        .attr('r', radius)
        .style('fill', (d, i) => color(i));
      gs.append('text')
        .attr('x', radius * 2).attr('y', (d, i) => i * (interval + 2 * radius) + fontSize / 2)
        .attr('text-anchor', 'start').style('font-size', fontSize)
        .text((d) => d);
      return this;
      // pos = this.g.selectAll
    }
    transform(transStr) {
      if (this.compare)
        transStr = transStr + 'scale(-1, 1)';
      this.g.attr('transform', transStr);
      return this;
    }
    clean() {
      this.g.selectAll('g').remove();
    }
  }


  export default {
    name: 'ClusterView',
    data() {
      return {
        // params: new LayoutParamsConstructor(this.width, this.height),
        clusterData: null,
        painter: null,
        svgId: this.compare ? 'cluster-svg2' : 'cluster-svg',
        // changingFlag: false,
        rendering: false,
        posLabel: null,
        rootGroup: null,
      }
    },
    props: {
      compare: {
        type: Boolean,
        defautl: false,
      },
      width: {
        type: Number,
        default: 800,
      },
      height: {
        type: Number,
        default: 800,
      },
    },
    computed: {
      style: function () {
        return this.model.style;
      },
      clusterNum: function () {
        return this.model.nCluster;
      },
      selectedWords: function () {
        return this.model.selectedWords;
      },
      selectedUnits: function () {
        return this.model.selectedUnits;
      },
      renderPos: function () {
        return this.model.renderPos;
      },
      sentences: function() {
        return this.model.sentences;
      },
      coClusterData: function() {
        return this.model.coCluster;
      },
      needRendering: function() {
        return this.model.needRendering;
      },
      model: function() {
        return this.compare ? this.$store.state.selectedModel2 : this.$store.state.selectedModel;
      }
    },
    watch: {
      style: function(newStyle, oldStyle) {
        if (!newStyle || !this.painter || !this.painter.hasData()) return;
        if (newStyle.mode !== oldStyle.mode) {
          this.maybeReload();
          return;
        }
        this.painter.styleParams(newStyle);
        if (newStyle.stateClip !== oldStyle.stateClip) {
          this.painter.changeStateClip(newStateClip);
        }
        if (newStyle.strokeControlStrength !== oldStyle.strokeControlStrength) {
          this.painter.refreshStroke(newStyle);
          return;
        }
        const linkFilter1 = newStyle.linkFilterThreshold;
        const linkFilter2 = oldStyle.linkFilterThreshold;
        if (linkFilter1[0] !== linkFilter2[0] || linkFilter1[1] !== linkFilter2[1]) {
          this.painter.refreshStroke(newStyle);
          return;
        }
      },
      model: function (newModel, oldModel) {
        console.log(`${this.svgId} > model changed to ${this.model.name}`);
        this.maybeReload();
      },
      needRendering: 'maybeReload',
      sentences: 'maybeReload',
      selectedWords: function (words) {
        if (words.length === 0) {
          this.painter.renderState([]);
        } else {
          this.getStatistics({
            modelName: this.model.name,
          }).then(() => {
              const statistics = this.model.stateStats;
              const wordsStatistics = statistics.statOfWord(this.selectedWords[this.selectedWords.length - 1]).mean;
              // const wordsStatistics = statistics.statOfWord(this.selectedWords[0].text).mean;
              console.log('click word ' + this.selectedWords[0]);
              this.painter.renderState(wordsStatistics);
            });
        }
      },
      renderPos: function (renderPos) {
        const data = {};
        if (renderPos) {
          bus.loadPosStatistics(this.selectedModel, undefined, (response) => {
            // console.log(response);
            if (response.status === 200) {
              const posStatistics = response.data;
              posStatistics.forEach((word, i) => {
                const posRatio = Object.keys(word.ratio).map((key, i) => {
                  return {
                    index: i,
                    pos: key,
                    value: word.ratio[key]
                  };
                });
                posRatio.sort((a, b) => b.value - a.value);
                data[word.word] = pos2tag[posRatio[0].pos];
              });
              // console.log(data);
              this.posLabel.draw(pos2tag).transform('translate(' + [this.width - 40, 10] + ')');
            }
            this.painter.renderWord(data);
          });
        } else {
          this.painter.renderWord(data);
          this.posLabel.clean();
        }
      },
      width: function (newWidth) {
        // this.params.updateWidth(newWidth);
        this.maybeReload();
      },
      height: function (newHeight) {
        // this.params.updateHeight(newHeight);
        this.maybeReload();
      }
    },
    methods: {
      maybeReload() {
      // this function might clogged.
        if (this.needRendering && !this.rendering) {
          this.rendering = true;

          this.reload()
            .then(() => {
              this.rendering = false;
              this.graphRendered({
                modelName: this.model.name,
              });
            });
        }
      },
      init() {
        // this.params.updateWidth(this.width);
        this.rootGroup = d3.select(`#${this.svgId}`).append('g');
        this.painter = new Painter(this.rootGroup, this.$store, this.compare);
        this.posLabel = new PosLabel(this.rootGroup.append('g'), labelParams, this.compare);
      },
      reload() {
        if (this.renderPos) {
          this.posLabel.clean();
          this.posLabel.draw(pos2tag).transform(`translate(${this.width - 40},10})`);
        }
        console.log(`${this.svgId} > reloading...`);
        // console.log(this.coClusterData.getColClusters().length);
        this.painter.layoutParams(new LayoutParamsConstructor({
          width: this.width,
          height: this.height,
          alignMode: this.mode,
          chipsSizes: this.coClusterData.colSizes,
          cloudsSizes: this.coClusterData.rowSizes,
        })).styleParams(this.style);
        this.rootGroup.attr('transform', this.compare ? `scale(-1,1)translate(${-this.width},0)` : '')
        // this.painter.destroy();
        console.log(this.coClusterData);
        return this.painter.draw(this.coClusterData);
      },
      changeStroke(controlStrength, linkFilterThreshold) {
        this.painter.refreshStroke(controlStrength, linkFilterThreshold);
      },
      ...mapActions({
        getStatistics: GET_STATE_STATISTICS,
      }),
      ...mapMutations({
        graphRendered: GRAPH_RENDERED,
      }),
    },
    mounted() {
      this.init();

      // bus.$on(EVALUATE_SENTENCE, (value, compare) => {
      //   if (compare !== this.compare)
      //     return;
      //   const record = bus.evalSentence(value, this.selectedModel);
      //   const p2 = record.evaluate();
      //   Promise.all([p2]).then((values) => {
      //     // TODO change -1 to something else
      //     const sentenceRecord = record.getRecords(this.selectedState, -1);

      //     this.painter.addSentence(value, record, sentenceRecord);
      //   })
      // });

      // bus.$on(CLOSE_SENTENCE, (sentence, compare) => {
      //   if (compare !== this.compare)
      //     return;
      //   this.painter.deleteSentence(sentence);
      // });

      // bus.$on(SELECT_COLOR, (newColor) => {
      //   console.log(`color has changed to ${newColor}`);
      //   d3.select(`#${this.svgId}`)
      //     .selectAll('.link.positive')
      //     .style('stroke', newColor[1]);
      //   d3.select(`#${this.svgId}`)
      //     .selectAll('.link.negative')
      //     .style('stroke', newColor[0]);
      // })
    }
  }


</script>
