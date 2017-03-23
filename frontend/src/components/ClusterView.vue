<style>
.hidden-cluster {
  stroke: gray;
  stroke-opacity: 0.5;
  stroke-width: 0.5;
  fill: lightgray;
  fill-opacity: 0.3;
}
.cluster-selected {
  stroke-width: 1.5;
  stroke-opacity: 0.5;
  stroke: black;
  fill-opacity: 0.3;
}
#middle_line {
  stroke: lightskyblue;
  stroke-width: 2;
  stroke-dasharray: 3, 3;
  fill: none;
  opacity: 0.5;
}
.little-triangle {
  fill: #1f77b4;
}
.link.active {
  opacity: 1;
}
.link {
  fill: none;
  opacity: 0.2;
}
.state_unit {

}
.state_unit .active {

}
.wordcloud {
  stroke: 'gray';
  stroke-width: 0.5;
  fill: 'white';
  fill-opacity: 0.0;
  stroke-opacity: 0.8;
}

.wordcloud-active {
  stroke: 'black';
  stroke-width: 1.5;
}

</style>
<template>
  <!--<div>-->
    <!--<div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>-->
    <svg :id='svgId' :width='width' :height='height' :transform='compare ? "scale(-1,1)" : ""'> </svg>
  <!--</div>-->
</template>

<script>
  import * as d3 from 'd3';
  import { bus, SELECT_MODEL, SELECT_STATE, CHANGE_LAYOUT, SELECT_WORD, EVALUATE_SENTENCE} from '../event-bus';
  import { WordCloud } from '../layout/cloud.js';
  import { sentence } from '../layout/sentence.js';

  const colorHex = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'];
  const colorScheme = (i) => colorHex[i];

  class LayoutParamsConstructor {
    constructor(){
      this.unitWidth = 3;
      this.unitHeight = 3;
      this.unitMarginSuppose = 2;
      this.unitMargin = 2;
      this.wordCloudArcDegree = 130;
      this.wordCloudNormalRadius = 60;
      this.wordCloudHightlightRatio = 1.5;
      this.wordCloudPaddingLength = 5;
      this.wordCloudChord2ClusterDistance = 50;
      this.wordCloudChordLength2ClientHeightRatio = 0.9;
      this.wordCloudChord2stateClusterHeightRatio = 1.1;
      this.wordCloudWidth2HeightRatio = 1 / 0.618;
      this.littleTriangleWidth = 5;
      this.littleTriangleHeight = 5;
      this.strengthThresholdPercent = 0.2;
      this.linkWidth2StrengthRatio = 0.01;
      this.wordSize2StrengthRatio = 3;
      this.middleLineX = 300;
      this.middleLineY = 50;
      this.sentenceWordThreshold = 0.2;
      this.posColor = colorScheme;
    }

    computeParams (clientHeight, clusterNum, clusterInterval2HeightRatio) {
      this.wordCloudChordLength = clientHeight * this.wordCloudChordLength2ClientHeightRatio;
      this.clusterHeight = (this.wordCloudChordLength / this.wordCloudChord2stateClusterHeightRatio) /
        (clusterNum + clusterNum * clusterInterval2HeightRatio - clusterInterval2HeightRatio);
      this.clusterInterval = this.clusterHeight * clusterInterval2HeightRatio;
      this.packNum = ~~(this.clusterHeight / (this.unitHeight + this.unitMargin));
      // this.unitMargin = this.clusterHeight / this.packNum - this.unitHeight;
      this.wordCloudChord2CenterDistance = this.wordCloudChordLength / 2 / Math.tan(this.wordCloudArcDegree / 2 * Math.PI / 180);
    }
    // clusterRectStyle: {
    //   'fill': '#eee',
    //   'fill-opacity': 0.5,
    //   'stroke': '#555',
    //   'stroke-width': 0.5,
    //   'stroke-opacity': 0.5,
    // },
  }
  const layoutParams = new LayoutParamsConstructor();
  // layoutParams.clusterHeight = layoutParams.unitHeight*layoutParams.packNum + layoutParams.unitMargin * (layoutParams.packNum + 1);
  // layoutParams.clusterWidth = layoutParams.clusterHeight / (layoutParams.packNum);

  const pos2tag = {
    "VERB": 0,
    "NOUN": 1,
    "PRON": 2,
    "ADJ" : 3,
    "ADV" : 4,
    "ADP" : 5,
    "CONJ": 6,
    "DET" : 7,
    "NUM" : 8,
    "PRT" : 9,
    "X" : 10,
    "." : 11,
  };

  const labelParams = {
    colorScheme: colorScheme,
    radius: 4,
    fontSize: 11,
    interval: 8
  }

  export default {
    name: 'ClusterView',
    data() {
      return {
        params: new LayoutParamsConstructor(),
        // svgId: 'cluster-svg',
        clusterData: null,
        // clusterNum: 10,
        painter: null,
        // painter2: null,
        shared: bus.state,
        // width: 800,
        changingFlag: false,
        posLabel: null,
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
      svgId: function () {
        return this.compare ? 'cluster-svg2' : 'cluster-svg';
      },
      selectedState: function() {
        return this.compare ? this.shared.selectedState2 : this.shared.selectedState;
      },
      selectedModel: function() {
        return this.compare ? this.shared.selectedModel2 : this.shared.selectedModel;
      },
      selectedLayer: function() {
        return this.compare ? this.shared.selectedLayer2 : this.shared.selectedLayer;
      },
      layout: function() {
        return this.compare ? this.shared.layout2 : this.shared.layout;
      },
      clusterNum: function() {
        return this.layout.clusterNum;
      },
      selectedWords: function() {
        return this.compare ? this.shared.selectedWords2 : this.shared.selectedWords;
      },
      selectedUnits: function() {
        return this.compare ? this.shared.selectedUnits2 : this.shared.selectedUnits;
      },
      renderPos: function() {
        return this.compare ? this.shared.renderPos2 : this.shared.renderPos;
      }
    },
    watch: {
      selectedState: function (state) {
        console.log(`${this.svgId} > state changed to ${this.selectedState}`);
        this.maybeReload();
      },
      selectedLayer: function (layer) {
        console.log(`${this.svgId} > layer changed to ${this.selectedLayer}`);
        this.maybeReload();
      },
      layout: function(layout) {
        console.log(`${this.svgId} > layout changed. clusterNum: ${this.clusterNum}`);
        this.maybeReload();
      },
      selectedModel: function (newModel, oldModel) {
        console.log(`${this.svgId} > model changed to ${this.selectedModel}`);
        this.maybeReload();
      },
      selectedWords: function (words) {
        if (words.length === 0) {
          this.painter.render_state([]);
          return;
        }
        // const words = words.map((word) => word.text);
        // this.compare = compare;
        let model = this.selectedModel,
          state = this.selectedState,
          layer = this.selectedLayer;
        const p = bus.loadStatistics(model, state, layer)
          .then(() => {
            const statistics = bus.getStatistics(model, state, layer);
            const wordsStatistics = statistics.statOfWord(this.selectedWords[0].text).mean;
            this.painter.render_state(wordsStatistics);
          });
      },
      renderPos: function(renderPos) {
        const data = {};
        if (renderPos) {
          bus.loadPosStatistics(this.selectedModel, undefined, (response) => {
            console.log(response);
            if (response.status === 200) {
              const posStatistics = response.data;
              posStatistics.forEach((word, i) => {
                const posRatio = Object.keys(word.ratio).map((key, i) => {
                  return {index: i, pos: key, value: word.ratio[key]};
                });
                posRatio.sort((a, b) => b.value - a.value);
                data[word.word] = pos2tag[posRatio[0].pos];
              });
              this.posLabel.draw(pos2tag).transform('translate(' + [this.width-40, 10] + ')');
            }
            this.painter.renderWord(data);
          });
        } else {
          this.painter.renderWord(data);
          this.posLabel.clean();
        }
      },
      width: function (newWidth, oldWidth) {
        console.log("width ${newWidth}");
        if (this.painter && typeof newWidth === 'number') {
          this.painter.translateX(newWidth/3 - this.painter.middle_line_x);
        }
        if (this.renderPos) {
          this.posLabel.clean();
          this.posLabel.draw(pos2tag).transform('translate(' + [this.width-40, 10] + ')');
        }
      },
    },
    methods: {
      checkLegality() {
        const state = this.selectedState;
        console.log(state);
        console.log(this.selectedLayer);
        console.log(this.layout);
        return (state === 'state' || state === 'state_c' || state === 'state_h')
          && ((typeof this.selectedLayer) === 'number') && (this.layout);
      },
      maybeReload() {
        // console.log(this.changingFlag);
        if (!this.changingFlag){
          this.changingFlag = true;
          console.log(this.changingFlag);
          if (this.checkLegality()){
            console.log(`${this.svgId} > reloading...`);
            this.reload(this.selectedModel, this.selectedState, this.selectedLayer, this.clusterNum)
              .then(() => {
                this.changingFlag = false;
              });
          } else {
            this.changingFlag = false;
          }
        }
      },
      init() {
        this.painter = new Painter(`#${this.svgId}`, this.params, this.compare);
        this.posLabel = new PosLabel(d3.select(`#${this.svgId}`), labelParams, this.compare);
      },
      reload(model, state, layer, clusterNum) {
        const params = {
          top_k: 300,
          mode: 'raw',
          layer: layer,
        };
        return bus.loadCoCluster(model, state, clusterNum, params)
          .then(() => {
            this.clusterData = bus.getCoCluster(model, state, clusterNum, params);
            this.painter.destroy();
            this.painter.draw(this.clusterData);
          });
      },
    },
    mounted() {
      // this.width = this.$el.clientWidth;
      this.init();
      // register events
      // bus.$on(SELECT_MODEL, (model) => {
      //   this.selectedModel = model;
      //   bus.loadModelConfig(model).then(() => {
      //     this.states = bus.availableStates(model);
      //   });
      // });
      bus.$on(EVALUATE_SENTENCE, (value, compare) => {
        // const p1 = bus.loadCoCluster(this.selectedModel, this.selectedState, this.clusterNum, {top_k: 300, mode: 'raw'});
        const record = bus.evalSentence(value, this.selectedModel);
        const p2 = record.evaluate();
        Promise.all([p2]).then((values) => {
          // const coCluster = bus.getCoCluster(this.selectedModel, this.selectedState, this.clusterNum, {top_k: 300, mode: 'raw'});
          // TODO change -1 to something else
          const sentenceRecord = record.getRecords(this.selectedState, -1);
          this.painter.drawSentence(record, sentenceRecord);

        })
      });
      // bus.$on(CHANGE_LAYOUT, (layout, compare) => {
      //   if (compare)
      //     return;
      //   console.log("cluster > Changing Layout...");
      //   // this.clusterNum = layout.clusterNum;
      // });
    }
  }

  class PosLabel {
    constructor(selector, params, compare=False) {
      this.g = selector.append('g');
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
      Object.keys(tags).forEach((key) => { labels[tags[key]] = key; });
      const gs = this.g.selectAll('g')
        .data(labels).enter()
        .append('g');
      gs.append('circle')
        .attr('cx', 0).attr('cy', (d, i) => i*(interval + 2*radius))
        .attr('r', radius)
        .style('fill', (d, i) => color(i));
      gs.append('text')
        .attr('x', radius*2).attr('y', (d, i) => i*(interval + 2*radius) + fontSize/2)
        .attr('text-anchor', 'start').style('font-size', fontSize)
        .text((d)=> d);
      return this;
      // pos = this.g.selectAll
    }
    transform(transStr){
      if (this.compare)
        transStr = transStr + 'scale(-1, 1)';
      this.g.attr('transform', transStr);
      return this;
    }
    clean(){
      this.g.selectAll('g').remove();
    }
  }

  class Painter {
    constructor(selector, params = layoutParams, compare=false) {
      this.svg = d3.select(selector);
      this.params = params;
      this.hwg = this.svg.append('g');
      this.hg = this.hwg.append('g');
      this.wg = this.hwg.append('g');

      this.client_width = this.svg.node().getBoundingClientRect().width;
      this.client_height = this.svg.node().getBoundingClientRect().height;
      this.middle_line_x = params.middleLineX;
      this.middle_line_y = params.middleLineY;
      this.triangle_height = 5;
      this.triangle_width = 5;

      this.dx = 0, this.dy = 0;
      this.graph = null;
      this.clusterSelected = [];

      this.state_elements = [];
      this.loc = null;
      this.wordClouds = [];

      this.unitNormalColor = '#ff7f0e';
      this.unitRangeColor = ['#09adff', '#ff5b09'];
      this.linkWidthRanage = [1, 5];
      this.linkColor = ['#09adff', '#ff5b09'];
      this.compare = compare;

    }

    drawSentence(record, sentenceRecord) {
      const sg = this.svg.append('g');
      const spg = sg.append('g');
      const translationX = 300;
      const sentenceTranslate = [100, 0];
      const self = this;
      this.translateX(translationX);

      console.log(record);

      const rectGroup = sg.append('g');
      this.drawRect(rectGroup, record.tokens.length, updateSentence);

      // TODO change -1 to something else
      const sent = sentence(spg)
        .size([translationX, this.client_height])
        .sentence(sentenceRecord)
        .coCluster(this.graph.coCluster)
        .words(record.tokens)
        .transform('translate(' + sentenceTranslate + ')')
        .draw();

      const links = [];
      console.log(sent.dataList);
      sent.dataList.forEach((d, i) => {
        const wordPos = sent.getWordPos(i);
        this.graph.state_info.state_cluster_info.forEach((s, j) => {

          let link = {
            source: {x: wordPos[0] + sent.nodeWidth, y: wordPos[1] + sent.nodeHeight/2},
            target: {x: s.top_left[0] + this.middle_line_x - sentenceTranslate[0], y: s.top_left[1] + this.middle_line_y + s.height / 2 - sentenceTranslate[1]},
          };
          links.push(link);
        });
      });
      console.log(links);
      console.log('there are ' + links.length + ' links');
      
      sg.append('g')
        .selectAll('.sentenceLink')
        .data(links).enter()
        .append('path')
        .attr('d', l => this.createLink(l))
        .attr('stroke', 'blue')
        .attr('stroke-width', 2)
        .attr('opacity', 0.2)
        .attr('fill', 'none')

      function updateSentence(extent_) {
        console.log(`extent_ is ${extent_}`);
        const words = record.tokens.slice(...extent_);
        console.log(`words is ${words}`);
        let scaleFactor = record.tokens.length / words.length;
        scaleFactor = Math.min(scaleFactor, 2);
        const newHeight = scaleFactor * self.client_height;
        let translateY = sent.getWordPos(~~((extent_[0] + extent_[1])/2))[1];
        sent.transform('scale('  + scaleFactor + ')translate(' + [sentenceTranslate[0]/scaleFactor, -translateY + self.client_height/2/scaleFactor] + ')');
        const links = [];

      }
    }

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
          let lower = Math.round(extent[0] / rectSize[1]);
          let upper = Math.round(extent[1] / rectSize[1]);
          extent[0] = lower * rectSize[1];
          extent[1] = upper * rectSize[1];
          d3.select(this).transition().call(d3.event.target.move, extent);
          console.log(d3.event.selection);
          func([lower, upper]);
        });
      g.append('g')
        .call(brush)
        .call(brush.move, [0, rectSize[1] * dataLength])
    }

    calculate_state_info(coCluster) {
      let state_loc = [];
      let state_cluster_loc = [];
      let little_triangle_loc = [];

      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;

      const stateClusters = coCluster.colClusters;
      const agg_info = coCluster.aggregation_info;
      const nCluster = coCluster.labels.length;
      const words = coCluster.words;

      stateClusters.forEach((clst, i) => {
        let width = Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) + unitMargin;
        let height = clusterHeight;
        let top_left = [-width / 2, i * (clusterHeight + clusterInterval)];

        state_cluster_loc[i] = {top_left: top_left, width: width, height: height};

        clst.forEach((c, j) => {
          let s_width = unitWidth;
          let s_height = unitHeight;
          let s_top_left = [(~~(j/packNum)) * (unitMargin + unitWidth) + unitMargin / 2 + top_left[0],
            j%packNum * (unitHeight + unitMargin) + unitMargin / 2 + top_left[1]];
          state_loc[c] = {top_left: s_top_left, width: s_width, height: s_height};
        });
      });
      return {state_cluster_info: state_cluster_loc, state_info: state_loc};
    }

    calculate_word_info(coCluster, selected_state_cluster_index=-1) {
      let self = this;
      const wordCloudPaddingLength = this.params.wordCloudPaddingLength;
      const wordSize2StrengthRatio = this.params.wordSize2StrengthRatio;
      const wordCloudWidth2HeightRatio = this.params.wordCloudWidth2HeightRatio;
      const wordClusters = coCluster.rowClusters;
      const words = coCluster.words;
      const nCluster = coCluster.labels.length;
      const agg_info = coCluster.aggregation_info;

      let chordLength = this.params.wordCloudChordLength;
      let availableLength = chordLength - nCluster * wordCloudPaddingLength;

      const wordCloudArcRadius = chordLength / 2 / Math.sin(this.params.wordCloudArcDegree / 2 * Math.PI / 180);
      this.params.wordCloudArcRadius = wordCloudArcRadius;
      // const wordCloudArcLength = wordCloudArcRadius * wordCloudArcDegree * Math.PI / 180;

      let highlight_clouds = [];
      if (selected_state_cluster_index >= 0) {
        self.graph.link_info[selected_state_cluster_index].forEach((d, j) => {
          if (d.strength > 0) {
            highlight_clouds.push(j);
          }
        });
      }
      highlight_clouds = new Set(highlight_clouds);
      let word_info = [];
      let wd_height = wordClusters.map((d, i) => {
        if (highlight_clouds.size === 0) {
          return Math.sqrt(d.length);
        } else if (!highlight_clouds.has(i)) {
          return Math.sqrt(d.length);
        } else {
          return Math.sqrt(d.length) * this.params.wordCloudHightlightRatio;
        }
      });

      let wd_height_sum = wd_height.reduce((acc, val) => {
        return acc + val;
      }, 0);

      let offset = -chordLength / 2;
      wordClusters.forEach((wdst, i) => {
        // let angle = wd_radius[i] / wd_radius_sum * availableDegree / 2;
        // let actual_radius = wordCloudArcRadius * angle * Math.PI / 180;
        // let angle_loc = angle + offset;
        const actual_height = wd_height[i] / wd_height_sum * availableLength;
        const actual_width = actual_height * wordCloudWidth2HeightRatio;
        const top_left_y = offset;
        const top_left_x = Math.sqrt(wordCloudArcRadius ** 2 - top_left_y ** 2);
        offset += actual_height + wordCloudPaddingLength;
        // if self.graph exist, then only update the location info
        if (self.graph && self.graph.word_info) {
          self.graph.word_info[i].top_left = [top_left_x, top_left_y];
          self.graph.word_info[i].width = actual_width;
          self.graph.word_info[i].height = actual_height;
          word_info[i] = self.graph.word_info[i];
        } else {
          const words_data = wdst.map((d) => {
            return {text: words[d], size: agg_info.row_single_2_col_cluster[d][i] * wordSize2StrengthRatio};
          });
          word_info[i] = {top_left: [top_left_x, top_left_y], width: actual_width,
            height: actual_height, words_data: words_data};
        }
      });
      return word_info;
    }

    calculate_link_info(state_info, word_info, coCluster, dx, dy) {
      const self = this;
      const strengthThresholdPercent = this.params.strengthThresholdPercent;
      let links = [];
      const row_cluster_2_col_cluster = coCluster.aggregation_info.row_cluster_2_col_cluster;
      console.log('row_cluster_2_col_cluster');
      console.log(row_cluster_2_col_cluster);
      state_info.state_cluster_info.forEach((s, i) => {
        let strength_extent = 0;
        if (!self.graph || !this.graph.link_info) {
          strength_extent = d3.extent(row_cluster_2_col_cluster[i]);
          // console.log(`threshold is ${strength_extent[0]*strengthThresholdPercent}, ${strength_extent[1]*strengthThresholdPercent}`);
        }
        word_info.forEach((w, j) => {
          if (links[i] === undefined) {
            links[i] = [];
          }
          // if self.graph exists, then only update the location info, keep
          if (this.graph && this.graph.link_info) {
            // console.log(self.graph.link_info[i][j].el);
            links[i][j] = {source: {x: s.top_left[0] + s.width,
              y: s.top_left[1] + s.height / 2},
              target: {x: w.top_left[0] + dx, y: w.top_left[1] + w.height/2 + dy},
              strength: self.graph.link_info[i][j].strength,
              el: self.graph.link_info[i][j].el,
            };
          } else {
            let tmp_strength = row_cluster_2_col_cluster[i][j];
            if (tmp_strength > 0) {
              tmp_strength = tmp_strength > strength_extent[1] * strengthThresholdPercent ? tmp_strength : 0;
            } else {
              tmp_strength = tmp_strength < strength_extent[0] * strengthThresholdPercent ? tmp_strength : 0;
            }
            links[i][j] = {source: {x: s.top_left[0] + s.width,
              y: s.top_left[1] + s.height / 2},
              target: {x: w.top_left[0] + dx, y: w.top_left[1] + w.height/2 + dy},
              strength: tmp_strength,
            };
          }

        });
      });
      return links;
    }

    ArcLength2Angle(length, radius) {
      return length / radius * 180 / Math.PI;
    }

    render_state(data) {
      if (data.length) {
        // console.log(this.graph.word_info);
        this.graph.state_info.state_info.forEach((s, i) => {
          d3.select(s['el'])
            .transition()
            .duration(300)
            .attr('fill', data[i] > 0 ? this.unitRangeColor[1] : this.unitRangeColor[0])
            .attr('fill-opacity', Math.abs(data[i]))
        });
      } else {
        this.graph.state_info.state_info.forEach((s, i) => {
          d3.select(s['el'])
            .transition()
            .duration(300)
            .attr('fill', this.unitNormalColor)
        });
      }
    }

    renderWord(data = {}) {
      // console.log(data);
      this.graph.word_info.forEach((wordCluster, i) => {
        wordCluster.words_data.forEach((word, j) => {
          word.type = data[word.text];
        })
      });
      // console.log(this.graph.word_info);
      this.draw_word(this.wg, this.graph);
    }

    redraw_word_link(selected_state_cluster_index = -1) {
      this.graph.word_info = this.calculate_word_info(this.graph.coCluster, selected_state_cluster_index);
      this.graph.link_info = this.calculate_link_info(this.graph.state_info, this.graph.word_info, this.graph.coCluster, this.dx, this.dy);
      this.draw_word(this.wg, this.graph);
      this.draw_link(this.hg, this.graph);
    }

    draw_state(g, graph) {
      let self = this;
      let coCluster = graph.coCluster;
      let state_info = graph.state_info;
      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterInterval = this.params.clusterInterval;
      const littleTriangleWidth = this.params.littleTriangleWidth;
      const littleTriangleHeight = this.params.littleTriangleHeight;

      if (this.clusterSelected.length !== coCluster.colClusters.length) {
        this.clusterSelected = coCluster.colClusters.map((d, i) => 0);
      }
      const clusterSelected = this.clusterSelected;
      g.append('line')
        .attr('id', 'middle_line')
        .attr('x1', 0)
        .attr('y1', -1000)
        .attr('x2', 0)
        .attr('y2', 1000)

      const hiddenClusters = g.selectAll('g rect')
        .data(coCluster.colClusters, (clst, i) => Array.isArray(clst) ? (String(clst.length) + String(i)) : this.id); // matching function

      const selectCluster = function (clst, i) {
        if (!clusterSelected[i]){
          clusterSelected[i] = 1;
          d3.select(this).select('rect').classed('cluster-selected', true);
          d3.select(this).property('selected', 'true');
          self.redraw_word_link(i)
          graph.link_info[i].forEach((l) => {d3.select(l['el']).classed('active', true);})
        } else {
          clusterSelected[i] = 0;
        }
      }

      const hGroups = hiddenClusters.enter()
        .append('g')
        .on('mouseover', function (clst, i) {
          if (clusterSelected[i]) return;
          // const selectedIdx = clusterSelected.indexOf(1);
          d3.select(this).select('rect').classed('cluster-selected', true);
          graph.link_info[i].forEach((l) => {d3.select(l['el']).classed('active', true);})
        })
        .on('mouseleave', function(clst, i) {
          if (clusterSelected[i]) return;
          if (d3.select(this).property('selected') === 'true') {
            d3.select(this).property('selected', 'false');
            self.redraw_word_link(-1);
          }
          d3.select(this).select('rect').classed('cluster-selected', false);
          graph.link_info[i].forEach((l) => {d3.select(l['el']).classed('active', false);})
        })
        .on('click', selectCluster)
        .attr('id', (clst, i) => (String(clst.length) + String(i)));

      hGroups.each(function (d, i) {
        graph.state_info.state_cluster_info[i]['el'] = this;
      });

      const clusterRect = hGroups.append('rect')
        .classed('hidden-cluster', true)
        .transition()
        .duration(400)
        .attr('width', (clst, i) => state_info.state_cluster_info[i].width)
        .attr('height', (clst, i) => state_info.state_cluster_info[i].height)
        .attr('x', (clst, i) => state_info.state_cluster_info[i].top_left[0])
        .attr('y', (clst, i) => state_info.state_cluster_info[i].top_left[1]);

      hGroups.append('path')
        .classed('little-triangle', true)
        .attr('d', 'M 0, 0 L ' + -littleTriangleWidth/2 + ', ' +
          littleTriangleHeight + ' L ' +  littleTriangleWidth/2 +
          ', ' + littleTriangleHeight + ' L 0, 0')
        .transition()
        .duration(400)
        .attr('transform', (k, i) => {
          return 'translate(' + [state_info.state_cluster_info[i].top_left[0] + state_info.state_cluster_info[i].width / 2,
            state_info.state_cluster_info[i].top_left[1] + state_info.state_cluster_info[i].height] + ')';
        });

      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      let tmp_units = units.enter()
        .append('rect')
        .on('mouseover', function(d, i) {
          if (d3.select(this.parentNode.parentNode).property('selected') === 'true') {
            d3.select(this).attr('fill-opacity', 1)
            console.log(d + 'is selected');
            bus.$emit(SELECT_UNIT, d);
            // fisheye in
          }
        })
        .on('mouseleave', function(d, i) {
          if (d3.select(this.parentNode.parentNode).property('selected') === 'true') {
            d3.select(this).attr('fill-opacity', 0.5)
            console.log(d + 'is deselected');
            bus.$emit(DESELECT_UNIT, d);
            // fisheye out
          }
        })
        .transition()
        .duration(400)
        .attr('width', (i) => state_info.state_info[i].width)
        .attr('height', (i) => state_info.state_info[i].height)
        .attr('x', (i) => state_info.state_info[i].top_left[0])
        .attr('y', (i) => state_info.state_info[i].top_left[1])
        .attr('fill', self.unitNormalColor)
        .attr('fill-opacity', 0.5)

      tmp_units.each(function(d) {
        graph.state_info.state_info[d]['el'] = this;
      })

      // add
      hiddenClusters.exit()
        .transition()
        .duration(400)
        .style('fill-opacity', 1e-6)
        .attr('width', 1)
        .attr('height', 1)
        .remove();
    }

    sum(arr) {
      return arr.reduce(function (acc, val) {
        return acc + val;
      }, 0);
    }

    draw_word(g, graph) {
      let self = this;
      let word_info = graph.word_info;
      word_info.forEach((wclst, i) => {
        if (wclst['wordCloud']) {
          wclst['wordCloud']
            .draw([wclst.width/2, wclst.height/2])
            .transform( 'translate(' + [wclst.top_left[0] + wclst.width/2, wclst.top_left[1] + wclst.height/2] + ')')
        } else {
          let tmp_g = g.append('g')
            .on('mouseover', function () {
              if(!wclst['wordCloud'].selected){
                self.graph.link_info.forEach((ls) => {
                  d3.select(ls[i]['el'])
                    .classed('active', true);
                });
                wclst['wordCloud'].bgHandle.classed('wordcloud-active', true);
              }
            })
            .on('mouseleave', function () {
              if(!wclst['wordCloud'].selected){
                self.graph.link_info.forEach((ls) => {
                  d3.select(ls[i]['el'])
                    .classed('active', false);
                });
                wclst['wordCloud'].bgHandle.classed('wordcloud-active', false);
              }
            })
            .on('click', function () {
              if(!wclst['wordCloud'].selected){
                wclst['wordCloud'].selected = true;
                wclst['wordCloud'].bgHandle.classed('wordcloud-active', true);
              } else {
                wclst['wordCloud'].selected = false;
                wclst['wordCloud'].bgHandle.classed('wordcloud-active', false);
              }
            });
          let myWordCloud = new WordCloud(tmp_g, wclst.width/2, wclst.height/2, 'rect', this.compare)
            .transform( 'translate(' + [wclst.top_left[0] + wclst.width/2, wclst.top_left[1] + wclst.height/2] + ')')
            .color(this.params.posColor);
          myWordCloud.update(word_info[i].words_data);

          // wclst['el'] = tmp_g.node();
          wclst['wordCloud'] = myWordCloud;
        }

      });
    }

    erase_link() {
      this.graph.link_info.forEach((ls) => {
        ls.forEach((l) => {
          d3.select(l['el'])
          .transition()
          .duration(500)
          .attr('opacity', 0)
          .remove()
        });
      });
    }

    erase_word () {
      this.graph.word_info.forEach((w) => {
        w['wordCloud'].destroy();

      });
    }

    erase_state() {
      this.graph.state_info.state_cluster_info.forEach((s) => {
        d3.select(s['el'])
          .transition()
          .duration(500)
          .attr('fill-opacity', 1e-6)
          .attr('opacity', 1e-6)
          .remove()
      });
    }

    createLink(d) {
        return "M" + d.source.x + "," + d.source.y
            + "C" + (d.source.x + d.target.x) / 2 + "," + d.source.y
            + " " + (d.source.x + d.target.x) / 2 + "," + d.target.y
            + " " + d.target.x + "," + d.target.y;
      }

    draw_link(g, graph) {
      const link_info = graph.link_info;
      const linkWidth2StrengthRatio = this.params.linkWidth2StrengthRatio;

      function flatten(arr) {
        return arr.reduce((acc, val) => {
          return acc.concat(Array.isArray(val) ? flatten(val) : val);
        }, []);
      }
      const strengthes = flatten(link_info).filter(d => {return d.strength > 0}).map(d => {return Math.abs(d.strength)});
      const scale = d3.scaleLinear()
        .domain(d3.extent(strengthes))
        .range(this.linkWidthRanage)

      link_info.forEach((ls, i) => {
        // const strengthRange = d3.extent(ls);
        ls.forEach((l, j) => {
          if (l['el']) {
            d3.select(l['el'])
              .transition()
              .duration(300)
              .attr('d', this.createLink(l))
          } else {
            let tmp_path = g.append('path')
              .classed('link', true)
              .classed('active', false)
              .attr('d', this.createLink(l))
              .attr('stroke-width', l.strength !== 0 ? scale(Math.abs(l.strength)) : 0)
              .attr('opacity', 0.2)
              .attr('stroke', l.strength > 0 ? this.linkColor[1] : this.linkColor[0])
            l['el'] = tmp_path.node();
          }

        });
      });
    }

    draw(coCluster) {
      let self = this;
      // console.log(coCluster.colClusters);
      let maxClusterSize = coCluster.colClusters.reduce((a, b) => Math.max(Array.isArray(a) ? a.length : a, b.length));
      const nCluster = coCluster.labels.length;
      let clusterInterval2HeightRatio = 2;
      console.log(`cluster number is ${nCluster}`);
      this.params.computeParams(this.client_height, coCluster.labels.length, clusterInterval2HeightRatio);

      let maxClusterWidth = Math.ceil(maxClusterSize / this.params.packNum) * (this.params.unitWidth + this.params.unitMargin);
      while (maxClusterWidth > 500) {
        clusterInterval2HeightRatio -= 0.05;
        maxClusterSize = coCluster.colClusters.reduce((a, b) => Math.max(Array.isArray(a) ? a.length : a, b.length));
        this.params.computeParams(this.client_height, coCluster.labels.length, clusterInterval2HeightRatio);
        maxClusterWidth = Math.ceil(maxClusterSize / this.params.packNum) * (this.params.unitWidth + this.params.unitMargin);
        // console.log(maxClusterWidth);
      }
      this.dx = this.params.wordCloudChord2ClusterDistance - (this.params.wordCloudChord2CenterDistance - maxClusterWidth / 2);
      this.dy = this.params.wordCloudChordLength / 2;
      console.log(`maxClusterWidth is ${maxClusterWidth}`);
      console.log(`wordCloudChord2CenterDistance is ${this.params.wordCloudChord2CenterDistance}`);
      console.log(`dx is ${this.dx}, dy is ${this.dy}`);

      // const clusterHeight = this.params.clusterHeight;
      // this.params.clusterInterval = (this.client_height / nCluster - clusterHeight) * 0.7;
      // const clusterInterval = this.params.clusterInterval;
      // let chordLength = nCluster * (clusterHeight + clusterInterval);
      // this.dx = 0, this.dy = chordLength / 2;
      const coClusterAggregation = coCluster.aggregation_info;
      let state_info = this.calculate_state_info(coCluster);

      // let word_info = this.calculate_word_info(coCluster);
      // let link_info = this.calculate_link_info(state_info, word_info, coCluster, this.dx, this.dy);
      self.graph = {
        state_info: state_info,
        // word_info: word_info,
        // link_info: link_info,
        coCluster: coCluster,
      }
      // }
      this.draw_state(this.hg, self.graph);

      this.redraw_word_link();
      // this.adjustdx(-10);

      this.translateX(0);
    }

    

    translateX(x) {
      this.middle_line_x += x;
      this.hg.attr('transform', 'translate(' + [this.middle_line_x, this.middle_line_y] + ')');
      this.wg.attr('transform', 'translate(' + [this.middle_line_x + this.dx, this.middle_line_y + this.dy] + ')');
    }

    adjustdx(newdx) {
      this.dx = newdx;
      this.redraw_word_link();
    }

    destroy() {
      if (!this.graph) {
        return;
      }
      this.erase_state();
      this.erase_link();
      this.erase_word();
      // self.graph.state_info.forEach(())
    }
  }

</script>
