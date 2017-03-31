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
  <!--<div>-->
    <!--<div class="header">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
    </div>-->

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
  import { bus, SELECT_MODEL, SELECT_STATE, CHANGE_LAYOUT, EVALUATE_SENTENCE, SELECT_UNIT, DESELECT_UNIT, CLOSE_SENTENCE, SELECT_SENTENCE_NODE, SELECT_COLOR} from '../event-bus';
  import { WordCloud } from '../layout/cloud.js';
  import { sentence } from '../layout/sentence.js';

  const colorHex = ['#33a02c','#1f78b4','#b15928','#fb9a99','#e31a1c','#6a3d9a','#ff7f00','#cab2d6','#ffff99','#a6cee3','#b2df8a','#fdbf6f'];
  const colorScheme = (i) => colorHex[i];
  const positiveColor = '#ff5b09';
  const negativeColor = '#09adff';

  class LayoutParamsConstructor {
    constructor(width=800, height=800){
      this.unitWidthRatio = 1.0;
      this.unitMarginRatio = 0.5;
      this.clusterMarginRatio = 0.7;
      this.wordCloudArcDegree = 110;
      this.wordCloudHightlightRatio = 1.5;
      this.wordCloudPaddingLength = 5;
      this.wordCloudChord2ClusterDistance = 50;
      this.wordCloudChordLength2ClientHeightRatio = 0.9;
      this.wordCloudChord2stateClusterHeightRatio = 1.1;
      this.wordCloudWidth2HeightRatio = 1 / 0.5;
      this.littleTriangleWidth = 5;
      this.littleTriangleHeight = 5;
      this.strengthThresholdPercent = [0.2, 1];
      this.wordSize2StrengthRatio = 3;
      this.dxShrinkFactor = 0.07;
      this.spacePerSentence = 2/20;
      this.sentenceNodeWidth = 100;
      this.sentenceInitTranslate = [50, 10]
      // this.middleLineX = 300;
      this.middleLineY = 50;
      // this.middleLineTranslationXAfterSentence = 200;
      this.brushTranslationX = -100;
      this.sentenceWordThreshold = 0.5;
      this.posColor = colorScheme;
      this.middleLineOffset = 0;
      this.width = width;
      this.height = height;
      this.sentenceBrushRectWidth = 10;
      this.mode = 'width';
    }
    // switchMode() {
    //   this.mode = this.mode === 'height' ? 'width' : 'height';
    // }
    get wordCloudWidth () {
      // return this.width*0.3;
      return this.width*0.15;
    }
    get unitHeight () {
      return this._unitHeight ? this._unitHeight : Math.max(3, Math.min(~~((this.width - 500)/500) + 4, 7));
    }
    updateWidth(width) {
      if (typeof width === 'number')
        this.width = Math.min(Math.max(500, width), 1400);
    }
    updateHeight(height) {
      if (typeof height === 'number')
        this.height = Math.min(Math.max(200, height), 1000);
    }
    get unitWidth() {
      return this.unitHeight * this.unitWidthRatio;
    }
    get unitMargin() {
      return this.unitHeight * this.unitMarginRatio;
    }
    get clusterMargin() {
      return this.unitHeight * this.clusterMarginRatio;
    }
    get maxClusterWidth() {
      // const width = Math.max(this.width, 500);
      return Math.min(this.width/3.5, 400);
    }
    get middleLineX() {
      // const width = Math.max(this.width, 500);
      return this.width * 0.25 + this.middleLineOffset;
    }
    computeUnitParams() {
      // const unitHeight = this.clusterHeight / this.cluster2UnitRatio;
      // this.unitHeight = ~~unitHeight;
    }
    get cluster2UnitRatio() {
      return (this.packNum + (this.packNum - 1 ) * this.unitMarginRatio + 2 * this.clusterMarginRatio );
    }
    computeParams (clusterNum, clusterInterval2HeightRatio, clusterSizes, callTime = 0) {
      if (callTime > 10) return;
      const maxClusterSize = clusterSizes.reduce((a, b) => Math.max(a, b), 0);
      const totalClusterSize = clusterSizes.reduce((a, b) => a+b, 0);
      this._unitHeight = Math.min(~~this.width/50, 3 + ~~(this.width / totalClusterSize));
      // console.log(this);
      this.wordCloudChordLength = this.height * this.wordCloudChordLength2ClientHeightRatio;
      this.clusterHeight = (this.wordCloudChordLength) /
        (clusterNum + clusterNum * clusterInterval2HeightRatio - clusterInterval2HeightRatio);
      this.packNum = ~~((this.clusterHeight / this.unitHeight + this.unitMarginRatio - 2 * this.clusterMarginRatio) / (1+this.unitMarginRatio));
      this.clusterHeight =  this.unitHeight * this.cluster2UnitRatio;
      this.clusterInterval = this.clusterHeight * clusterInterval2HeightRatio;

      if (this.middleLineY < 0) {
        // callTime++;
        this.computeParams(clusterNum, clusterInterval2HeightRatio-0.2, clusterSizes, callTime+1);
      }
      const maxClusterWidth = Math.ceil(maxClusterSize / this.packNum) * (this.unitWidth + this.unitMargin);
      if (maxClusterWidth < 0.8 * this.maxClusterWidth) {
        // callTime++;
        clusterInterval2HeightRatio += 0.2;
        this.computeParams(clusterNum, clusterInterval2HeightRatio, clusterSizes, callTime+1);
      }
      if (maxClusterWidth > this.maxClusterWidth) {
        // callTime++;
        clusterInterval2HeightRatio -= 0.2;
        this.computeParams(clusterNum, clusterInterval2HeightRatio, clusterSizes, callTime+1);
      }
      if (callTime > 0) return;
      if (this.mode === 'height')
        this.middleLineY = (this.height - this.clusterHeight * clusterNum - this.clusterInterval * (clusterNum - 1)) / 2;
      else{
        this.widthPackNum = ~~(this.width/5.5/(this.unitWidth+this.unitMargin));
        this.clusterWidth = this.widthPackNum * (this.unitWidth + this.unitMargin) - this.unitMargin + 2 * this.clusterMargin;
        let heightSum = 0;
        const heights = clusterSizes.forEach((size, i) => {

          const pack = Math.ceil(size / this.widthPackNum);
          heightSum += pack * (this.unitHeight + this.unitMargin) - this.unitMargin + 2 * this.clusterMargin;
          // console.log(`pack ${pack}, widthPack ${this.widthPackNum}, size ${size}`);
          // console.log(heightSum);

        });
        // const heightSum = heights.reduce((a, b) => a+b, 0);
        this.clusterInterval = Math.floor((this.height - heightSum) / (clusterNum+2));
        // console.log(`heightSum ${heightSum}, clusterInterval ${this.clusterInterval}`);
        this.middleLineY = (this.height - heightSum - (clusterNum)*this.clusterInterval)/2;
        // return;
      }


    }

  }

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
        params: new LayoutParamsConstructor(this.width, this.height),
        // svgId: 'cluster-svg',
        clusterData: null,
        // clusterNum: 10,
        painter: null,
        // painter2: null,
        shared: bus.state,
        // width: 800,
        changingFlag: false,
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
        // console.log(`${this.svgId} > cluster num: ${this.layout.clusterNum}`);
        return this.layout.clusterNum;
      },
      strokeControlStrength: function() {
        return this.layout.strokeControlStrength;
      },
      linkFilterThreshold: function() {
        return this.layout.linkFilterThreshold;
      },
      stateClip: function() {
        return this.layout.stateClip;
      },
      selectedWords: function() {
        return this.compare ? this.shared.selectedWords2 : this.shared.selectedWords;
      },
      selectedUnits: function() {
        return this.compare ? this.shared.selectedUnits2 : this.shared.selectedUnits;
      },
      renderPos: function() {
        return this.compare ? this.shared.renderPos2 : this.shared.renderPos;
      },
      mode: function() {
        return this.layout.mode;
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
      strokeControlStrength: function(strokeControlStrength) {
        this.changeStroke(this.strokeControlStrength, this.linkFilterThreshold);
      },
      linkFilterThreshold: function(linkFilterThreshold) {
        this.changeStroke(this.strokeControlStrength, this.linkFilterThreshold);
      },
      clusterNum: function(clusterNum) {
        console.log(`${this.svgId} > layout changed. clusterNum: ${this.layout.clusterNum}`);
        this.maybeReload();
      },
      stateClip: function(newStateClip) {
        console.log('state clip changed');
        this.painter.changeStateClip(newStateClip);
      },
      mode: function(newMode) {
        this.params.mode = newMode;
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
            // console.log(response);
            if (response.status === 200) {
              const posStatistics = response.data;
              posStatistics.forEach((word, i) => {
                const posRatio = Object.keys(word.ratio).map((key, i) => {
                  return {index: i, pos: key, value: word.ratio[key]};
                });
                posRatio.sort((a, b) => b.value - a.value);
                data[word.word] = pos2tag[posRatio[0].pos];
              });
              // console.log(data);
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
        this.params.updateWidth(newWidth);
        this.maybeReload();
      },
      height: function (newHeight) {
        this.params.updateHeight(newHeight);
        this.maybeReload();
      }
    },
    methods: {
      checkLegality() {
        const state = this.selectedState;
        // console.log(state);
        // console.log(this.selectedLayer);
        // console.log(this.layout);
        return (state === 'state' || state === 'state_c' || state === 'state_h')
          && ((typeof this.selectedLayer) === 'number') && (this.layout) && (this.painter)
          && (typeof this.width === 'number') && (typeof this.height === 'number');
      },
      maybeReload() {
        if (!this.changingFlag){
          this.changingFlag = true;
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
        this.params.updateWidth(this.width);
        this.rootGroup = d3.select(`#${this.svgId}`).append('g');
        this.painter = new Painter(this.rootGroup, this.params, this.compare);
        this.posLabel = new PosLabel(this.rootGroup.append('g'), labelParams, this.compare);
      },
      reload(model, state, layer, clusterNum) {
        const params = {
          top_k: 300,
          mode: 'raw',
          layer: layer,
        };
        if (this.renderPos) {
          this.posLabel.clean();
          this.posLabel.draw(pos2tag).transform('translate(' + [this.width-40, 10] + ')');
        }
        return bus.loadCoCluster(model, state, clusterNum, params)
          .then(() => {
            this.clusterData = bus.getCoCluster(model, state, clusterNum, params);
            console.log(this.clusterData.colClusters.length);
            this.rootGroup.attr('transform', this.compare ? 'scale(-1,1)translate(' + [-this.width, 0] + ')' : '')
            this.painter.destroy();
            this.painter.draw(this.clusterData);
          });
      },
      changeStroke(controlStrength, linkFilterThreshold) {
        this.painter.refreshStroke(controlStrength, linkFilterThreshold);
      }
    },
    mounted() {
      this.init();

      bus.$on(EVALUATE_SENTENCE, (value, compare) => {
        if (compare !== this.compare)
          return;
        const record = bus.evalSentence(value, this.selectedModel);
        const p2 = record.evaluate();
        Promise.all([p2]).then((values) => {
          // TODO change -1 to something else
          const sentenceRecord = record.getRecords(this.selectedState, -1);

          this.painter.addSentence(value, record, sentenceRecord);
        })
      });

      bus.$on(CLOSE_SENTENCE, (sentence, compare) => {
        if(compare !== this.compare)
          return;
        this.painter.deleteSentence(sentence);
      });

      bus.$on(SELECT_COLOR, (newColor) => {
        console.log(`color has changed to ${newColor}`);
        d3.select(`#${this.svgId}`)
          .selectAll('.link.positive')
          .style('stroke', newColor[1]);
        d3.select(`#${this.svgId}`)
          .selectAll('.link.negative')
          .style('stroke', newColor[0]);
      })
    }
  }

  class PosLabel {
    constructor(selector, params, compare=False) {
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
    constructor(selector, params, compare=false) {
      this.svg = selector;
      this.params = params;

      this.topGroupClass = 'topGroup';
      this.topg = this.svg.append('g').attr('class', this.topGroupClass);
      this.hg = this.topg.append('g');
      this.wg = this.topg.append('g');
      this.sentenceWordThreshold = params.sentenceWordThreshold;
      this.strengthThresholdPercent = params.strengthThresholdPercent;

      this.dx = 0, this.dy = 0;
      this.graph = null;
      this.clusterSelected = [];

      this.state_elements = [];
      this.loc = null;
      this.wordClouds = [];

      this.unitNormalColor = '#ff5b09';
      this.unitRangeColor = ['#09adff', '#ff5b09'];
      this.linkWidthRange = [1, 5];
      this.linkColor = ['#09adff', '#ff5b09'];
      this.compare = compare;
      this.stateTranslateHis = [];
      this.sentenceTranslateHis = [this.params.sentenceInitTranslate[0],];
      this.sentences = [];
      this.stateClip = 2;

      this.strokeWidth = function(t) { return Math.abs(t) * 0.01};
      this.colorLegendAxis = null;
    }
    get dxOffset() {
      const adaptive = (this.client_width - this.client_height) / 4;
      // return Math.max(adaptive, 0);
      return 0;
    }
    get stateClusterWordCloudDX () {
      return this.client_width * ( - this.params.dxShrinkFactor * this.sentences.length) + this.dxOffset;
    }
    transform(trans) {
      this.svg.attr('transform', trans);
    }
    get client_width () {
      return this.params.width;
    }
    get client_height () {
      return this.params.height;
    }
    get middle_line_x() {
      return this.params.middleLineX;
    }
    get middle_line_y() {
      return this.params.middleLineY;
    }

    get nSentence() {
      return this.sentences.length;
    }

    refreshStroke(controlStrength, linkFilterThreshold) {
      this.strokeWidth = function(t) {return Math.abs(t) * controlStrength};
      this.sentenceStrokeWidth = function(t) {return Math.abs(t) * controlStrength * 3};
      this.strengthThresholdPercent = linkFilterThreshold;
      if (this.graph) {
        this.draw_link(this.hg, this.graph);

        this.graph.sentence_info.forEach((si) => {
          const strength_extent = d3.extent(this.flatten(si.links).map((l) => l.strength));
          const max_strength = Math.max(Math.abs(strength_extent[0]), Math.abs(strength_extent[1]));
          const strength_bound = this.strengthThresholdPercent.map((t) => t * max_strength);
          si.links.forEach((ls) => {
            ls.forEach((l) => {
              l['el'].attr('display', (Math.abs(l.strength) < strength_bound[0] || Math.abs(l.strength) > strength_bound[1]) ? 'none' : '');
            })
          })
        })
      }
    }

    changeStateClip(clip) {
      if (!this.graph) {
        return;
      }
      this.stateClip = clip;
      const scale = d3.scaleLinear()
                        .domain([-this.stateClip, 0, this.stateClip])
                        .range([1, 0, 1]);
      if (d3.select(this.graph.state_info.state_info[0]['el']).property('colored')) {
        this.graph.state_info.state_info.forEach((s, i) => {
          let tmp_s = d3.select(s['el']);
          tmp_s
            .transition()
            .attr('fill', tmp_s.property('strength') > 0 ? positiveColor : negativeColor)
            .attr('fill-opacity', scale(Math.abs(tmp_s.property('strength'))));

        });
        this.axisScale.domain([-this.stateClip, this.stateClip]);
        this.colorLegendAxis = d3.axisBottom()
                          .tickValues([parseInt(-this.stateClip), 0, parseInt(this.stateClip)])
                          .scale(this.axisScale);
        // this.hg.select('#color-legend')
        //        .select('g')
        //        .remove();
        this.hg.select('#color-legend')
               .select('g')
               .call(this.colorLegendAxis);
      }

    }

    deleteSentence(value) {
      const sentence_to_add = this.sentences.filter((s) => {return s !== value});
      this.sentences = [];
      this.translateX(-d3.sum(this.stateTranslateHis));
      this.adjustdx(this.stateClusterWordCloudDX);
      this.stateTranslateHis = [];
      this.sentenceTranslateHis = [this.params.sentenceInitTranslate[0],];

      const tmpSentenceInfo = {};
      this.graph.sentence_info.forEach((s) => {
        tmpSentenceInfo[s.value] = s;
        s['group'].remove();
      });
      this.graph.sentence_info = [];
      sentence_to_add.forEach((s) => {
        this.addSentence(s, tmpSentenceInfo[s].record, tmpSentenceInfo[s].sentenceRecord)
      });
    }

    deleteAllSentences() {
      this.sentences = [];
      this.translateX(-d3.sum(this.stateTranslateHis));
      this.adjustdx(this.stateClusterWordCloudDX);
      this.stateTranslateHis = [];
      this.sentenceTranslateHis = [this.params.sentenceInitTranslate[0],];

      const tmpSentenceInfo = {};
      this.graph.sentence_info.forEach((s) => {
        tmpSentenceInfo[s.value] = s;
        s['group'].remove();
      });
      this.graph.sentence_info = [];
    }

    addSentence(value, record, sentenceRecord) {
      const self = this;
      const needTranslate = (this.params.spacePerSentence + this.params.dxShrinkFactor) * this.client_width;
      const translationForEachSentence = needTranslate;
      const translationForState = needTranslate * 0.5;
      const sentenceInitTranslate = this.params.sentenceInitTranslate;
      this.stateTranslateHis.push(translationForState);
      if (this.sentences.length) {
        this.sentenceTranslateHis.push(translationForEachSentence);
      }
      // const sentenceTranslationX = this.client_width * 0.01;
      const sg = this.svg.append('g')
                      .attr('transform', `translate(${sentenceInitTranslate[0]}, ${sentenceInitTranslate[1]})`)
      this.sentences.push(value);
      const spg = sg.append('g');
      const sentenceTranslate = [50, 10];
      this.translateX(d3.sum(this.stateTranslateHis));
      this.adjustdx(this.stateClusterWordCloudDX);

      this.graph.sentence_info.forEach((s, k) => {
        s.group
        .transition()
        .attr('transform', 'translate(' + [d3.sum(this.sentenceTranslateHis), sentenceInitTranslate[1]] + ')');
      });

      const rectGroup = sg.append('g').attr('transform', 'translate(' + [-this.params.sentenceNodeWidth/2, this.client_height/4] + ')');
      this.drawBrushRect(rectGroup, sentenceRecord.length, updateSentence);

      const sent = sentence(spg, this.compare)
        .size([this.params.sentenceNodeWidth, this.params.sentenceNodeWidth * sentenceRecord.length])
        .sentence(sentenceRecord)
        .coCluster(this.graph.coCluster)
        .words(record.tokens)
        .mouseoverCallback(highlightSentenceLinkByNodeIndex)
        .draw();

      const links = [];
      const strength = sent.strengthByCluster;
      sent.strengthByCluster.forEach((strengths, i) => {
        links[i] = [];
        const wordPos = sent.getWordPos(i);
        this.graph.coCluster.colClusters.forEach((clst, j) => {
          const strength = strengths[j];
          const s = this.graph.state_info.state_cluster_info[j];
          links[i][j] = {
            source: {x: wordPos[0] + sent.nodeWidth, y: wordPos[1] + sent.nodeHeight/2},
            source_init: {x: wordPos[0] + sent.nodeWidth, y: wordPos[1] + sent.nodeHeight/2},
            // source_init: {x: wordPos[0] + sent.nodeWidth + sentenceTranslate[0], y: wordPos[1] + sent.nodeHeight/2 + sentenceTranslate[1]},
            target: {x: s.top_left[0] + this.middle_line_x - sentenceInitTranslate[0] , y: s.top_left[1] + this.middle_line_y + s.height / 2 - sentenceInitTranslate[1]},
            strength: strength,
          };
        });
      });
      // links.forEach(function(ls) {
      //   const strengthExtent = d3.extent(ls.map(l => Math.abs(l.strength)));
      //   const filterStrength = strengthExtent[1] * self.sentenceWordThreshold;
      //   ls.forEach((l) => {
      //     l.strength = Math.abs(l.strength) < filterStrength ? 0 : l.strength;
      //   });
      // });

      // const strengthes = flatten(links).map(d => {return Math.abs(d.strength)});
      // const scale = d3.scaleLinear()
      //   .domain([0, d3.extent(strengthes)[1]])
      //   .range([0, this.linkWidthRange[1]])


      const strength_extent = d3.extent(this.flatten(links).map((l) => l.strength));
      const max_strength = Math.max(Math.abs(strength_extent[0]), Math.abs(strength_extent[1]));
      const strength_bound = this.strengthThresholdPercent.map((t) => t * max_strength);

      const lsg = sg.append('g');
      links.forEach(function(ls) {
        ls.forEach(function(l) {
          l['el'] = lsg.append('path')
                      .classed('active', false)
                      .classed('link', true)
                      .classed(l.strength > 0 ? 'positive' : 'negative', true)
                      .attr('d', self.createLink(l))
                      .attr('stroke-width', self.sentenceStrokeWidth(l.strength))
                      .attr('opacity', 0.2)
                      .attr('fill', 'none')
                      .attr('display', (Math.abs(l.strength) < strength_bound[0] || Math.abs(l.strength) > strength_bound[1]) ? 'none' : '')
                      .attr('hold', 'false');
        });
      });
      self.graph.sentence_info.push({sentence: sent, links: links, value: value, group: sg, record: record, sentenceRecord: sentenceRecord});
      if (this.sentences.length > 1) {
        self.graph.sentence_info.forEach((si) => {
          si.links.forEach((ls) => {
            ls.forEach((l) => {
              l['el'].attr('display', 'none');
            });
          });
        });
      }

      function updateSentence(extent_) {
        // console.log(`extent_ is ${extent_}`);
        const words = record.tokens.slice(...extent_);
        // console.log(`words is ${words}`);
        let scaleFactor = self.client_height / words.length / self.params.sentenceNodeWidth / 1.05;
        // console.log(`scaleFactor is ${scaleFactor}`);
        scaleFactor = Math.min(scaleFactor, 1.5);
        const newHeight = scaleFactor * self.client_height;
        let translateY = (sent.getWordPos(extent_[0])[1] + sent.getWordPos(extent_[1])[1]) / 2;
        // let translateY = sent.getWordPos(~~((extent_[0] + extent_[1])/2))[1];
        sent.transform('scale('  + scaleFactor + ')translate(' + [0, -translateY + self.client_height/2/scaleFactor] + ')');
        // sent.transform('scale('  + scaleFactor + ')translate(' + [d3.sum(self.sentenceTranslateHis)/scaleFactor, -translateY + self.client_height/2/scaleFactor] + ')');
        links.forEach((ls) => {
          ls.forEach((l) => {
            const actualY = (l.source_init.y - translateY) * scaleFactor + self.client_height / 2;
            console.log(`client height is ${self.client_height}`);
            console.log(`actualY is ${actualY}`);
            // const actualY = (l.source_init.y - translateY) * scaleFactor + self.client_height / 2;
            l.source.x = l.source_init.x * scaleFactor;
            l.source.y = actualY;
            l['el']
              .transition()
              .attr('d', self.createLink(l));
          })
        })
      }

      function highlightSentenceLinkByNodeIndex (data, t, highlight, changeCheckStatus=false) {
        // console.log(this);
        links[t].forEach((l) => {
          if (changeCheckStatus) {
            l['el'].attr('hold', l['el'].attr('hold') === 'false' ? 'true' : 'false');
          }
          if (l['el'].attr('hold') !== 'true') {
            l['el'].classed('active', highlight)
            .attr('display', highlight || self.sentences.length < 2 ? '' : 'none');
          } else {
            l['el'].classed('active', true)
            .attr('display', '');
          }

        });
        if(changeCheckStatus) {
          data.selected = !data.selected;
          bus.$emit(SELECT_SENTENCE_NODE, data, self.compare);
        }
        // console.log(data.el);
        data.bg.classed('active', data.selected ? true : highlight);
      }
    }

    drawBrushRect(g, dataLength, func) {
      const self = this;

      const maxHeight = this.params.height / 2;
      const unitHeight = Math.min( maxHeight / dataLength, 50);
      console.log(`unitHeight is ${unitHeight}`);
      const rectSize = [self.params.sentenceBrushRectWidth, unitHeight];
      console.log(`rect size is ${rectSize}`);
      const minBrushLength = 3;
      g.selectAll('.wordRect')
        .data(d3.range(dataLength)).enter()
        .append('rect')
        .attr('x', 0)
        .attr('y', d => d * rectSize[1])
        .attr('width', rectSize[0])
        .attr('height', rectSize[1])
        .attr('fill', 'black')
        .attr('stroke-width', 1)
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

    calculate_state_info(coCluster, mode='width') {
      let state_loc = [];
      let state_cluster_loc = [];
      // let little_triangle_loc = [];

      const clusterHeight = this.params.clusterHeight;
      const packNum = this.params.packNum;
      this.params.computeUnitParams();
      const unitHeight = this.params.unitHeight;
      const unitWidth = this.params.unitWidth;
      const unitMargin = this.params.unitMargin;
      const clusterMargin = this.params.clusterMargin;
      const clusterInterval = this.params.clusterInterval;

      const stateClusters = coCluster.colClusters;

      if (mode === 'height'){
        stateClusters.forEach((clst, i) => {
          let width = Math.ceil(clst.length / packNum) * (unitWidth + unitMargin) - unitMargin + 2 * clusterMargin;
          let height = clusterHeight;
          let top_left = [-width / 2, i * (height + clusterInterval)];

          state_cluster_loc[i] = {top_left: top_left, width: width, height: height};

          clst.forEach((c, j) => {
            let s_width = unitWidth;
            let s_height = unitHeight;
            let s_top_left = [(~~(j/packNum)) * (unitMargin + unitWidth) + clusterMargin,
              j%packNum * (unitHeight + unitMargin) + clusterMargin];
            state_loc[c] = {top_left: s_top_left, width: s_width, height: s_height};
          });
        });
      } else if (mode === 'width') {
        const width = this.params.clusterWidth;
        const widthPackNum = this.params.widthPackNum;
        let offset = 0;
        stateClusters.forEach((clst, i) => {
          let height = Math.ceil(clst.length / widthPackNum) * (unitHeight + unitMargin) - unitMargin + 2 * clusterMargin;
          let top_left = [-width / 2, offset];
          offset += (height + clusterInterval);
          // let packNum =
          state_cluster_loc[i] = {top_left: top_left, width: width, height: height};

          clst.forEach((c, j) => {
            let s_width = unitWidth;
            let s_height = unitHeight;
            let s_top_left = [(j%widthPackNum) * (unitMargin + unitWidth) + clusterMargin,
              ~~(j/widthPackNum) * (unitHeight + unitMargin) + clusterMargin];
            state_loc[c] = {top_left: s_top_left, width: s_width, height: s_height};
          });
        });
      }
      return {state_cluster_info: state_cluster_loc, state_info: state_loc};
    }

    calculate_word_info(coCluster, selected_state_cluster_index=-1) {
      let self = this;
      const wordCloudPaddingLength = this.params.wordCloudPaddingLength;
      const wordSize2StrengthRatio = this.params.wordSize2StrengthRatio;
      // const wordCloudWidth2HeightRatio = this.params.wordCloudWidth2HeightRatio;
      const baseWordCloudWidth = this.params.wordCloudWidth;
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
          // if (Math.abs(d.strength) > 0) {
          if (d3.select(d['el']).attr('display') !== 'none') {
            highlight_clouds.push(j);
          }
        });
      }
      // console.log("need to highlight ");
      // console.log(highlight_clouds);
      highlight_clouds = new Set(highlight_clouds);
      let word_info = [];
      let wd_height = wordClusters.map((d, i) => {
        if (highlight_clouds.size === 0) {
          return d.length;
          // return Math.sqrt(d.length);
        } else if (!highlight_clouds.has(i)) {
          // return Math.sqrt(d.length);
          return d.length;
        } else {
          return d.length * this.params.wordCloudHightlightRatio;
          // return Math.sqrt(d.length) * this.params.wordCloudHightlightRatio;
        }
      });

      let wd_height_sum = wd_height.reduce((acc, val) => {
        return acc + val;
      }, 0);
      const max_height = wd_height.reduce((a, b) => Math.max(a,b), 0);

      let offset = -chordLength / 2 ;
      wordClusters.forEach((wdst, i) => {
        // let angle = wd_radius[i] / wd_radius_sum * availableDegree / 2;
        // let actual_radius = wordCloudArcRadius * angle * Math.PI / 180;
        // let angle_loc = angle + offset;
        const actual_height = wd_height[i] / wd_height_sum * availableLength;
        const actual_width = baseWordCloudWidth; // * wd_height[i] / max_height;
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
      // console.log(word_info);
      return word_info;
    }

    calculate_link_info(state_info, word_info, coCluster, dx, dy) {
      const self = this;
      let links = [];
      const row_cluster_2_col_cluster = coCluster.aggregation_info.row_cluster_2_col_cluster;
      const colClusters = coCluster.colClusters;
      const rowClusters = coCluster.rowClusters;
      const labels = coCluster.labels;
      // console.log('row_cluster_2_col_cluster');
      // console.log(row_cluster_2_col_cluster);
      state_info.state_cluster_info.forEach((s, i) => {
        let max_strength = 0;
        // if (!self.graph || !this.graph.link_info) {
        //   const strength_extent = d3.extent(row_cluster_2_col_cluster.map((row_cluster) => row_cluster[i]));
        //   max_strength = Math.max(Math.abs(strength_extent[0]), Math.abs(strength_extent[1]));
        //   // console.log(`threshold is ${strength_extent[0]*strengthThresholdPercent}, ${strength_extent[1]*strengthThresholdPercent}`);
        // }
        // const filter_strength = max_strength * strengthThresholdPercent;
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
            let tmp_strength = row_cluster_2_col_cluster[labels[j]][labels[i]];
            // tmp_strength = Math.abs(tmp_strength) < filter_strength ? 0 : tmp_strength;
            links[i][j] = {source: {x: s.top_left[0] + s.width,
              y: s.top_left[1] + s.height / 2},
              target: {x: w.top_left[0] + dx, y: w.top_left[1] + w.height/2 + dy},
              strength: tmp_strength,
            };
          }

        });
      });
      // console.log(links);
      return links;
    }

    ArcLength2Angle(length, radius) {
      return length / radius * 180 / Math.PI;
    }

    render_state(data) {
      if (data.length) {
        if (!this.hg.select('#color-legend').node()) {
          const tmp_state = this.graph.state_info.state_cluster_info[this.graph.state_info.state_cluster_info.length - 1];
          const tmp_g = this.hg.append('g')
                            .attr('id', 'color-legend')
                            .attr('transform', 'translate(' + [0, tmp_state.top_left[1] + tmp_state.height + 2 * this.params.clusterInterval] + ')');
          const width = tmp_state.width;
          const height = this.params.unitHeight * 2;
          tmp_g.append('rect')
          .attr('transform', 'translate(' + [-width / 2, 0] + ')')
          .transition()
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', width)
          .attr('height', this.params.unitHeight * 2)
          .style('fill', 'url(#state-legend)');

          this.axisScale = d3.scaleLinear().range([0, width]).domain([-this.stateClip, this.stateClip]);
          this.colorLegendAxis = d3.axisBottom()
                          .tickValues([parseInt(-this.stateClip), 0, parseInt(this.stateClip)])
                          .scale(this.axisScale);
          tmp_g.append('g')
               .classed('axis', true)
               .attr('transform', 'translate(' + [-width/2, 0] + ')')
               .call(this.colorLegendAxis);
        }

        // const extent = d3.extent(data);
        const scale = d3.scaleLinear()
                        .domain([-this.stateClip, 0, this.stateClip])
                        .range([1, 0, 1]);
        // console.log(`the extent of render state is ${extent}`);
        this.graph.state_info.state_info.forEach((s, i) => {
          d3.select(s['el'])
            .property('colored', true)
            .property('strength', data[i])
            .transition()
            .duration(300)
            .attr('fill', data[i] > 0 ? positiveColor : negativeColor)
            .attr('fill-opacity', scale(Math.abs(data[i])));
        });

      } else {
        this.hg.select('#color-legend').remove();
        this.graph.state_info.state_info.forEach((s, i) => {
          d3.select(s['el'])
            .property('colored', false)
            .transition()
            .duration(400)
            .attr('fill', this.unitNormalColor)
            .attr('fill-opacity', 0.3)
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
      // console.log(this.graph.word_info);
      this.draw_word(this.wg, this.graph);
    }

    redraw_word_link(selected_state_cluster_index = -1) {
      this.graph.word_info = this.calculate_word_info(this.graph.coCluster, selected_state_cluster_index);
      this.graph.link_info = this.calculate_link_info(this.graph.state_info, this.graph.word_info, this.graph.coCluster, this.stateClusterWordCloudDX, this.dy);
      this.draw_word(this.wg, this.graph);
      this.draw_link(this.hg, this.graph);
    }

    update_ref(el, mode) {
      if (mode === "plus") {
        d3.select(el).property('ref', parseInt(d3.select(el).property('ref')) + 1);
        d3.select(el).classed('active', true);
      } else if (mode === "minus") {
        d3.select(el).property('ref', parseInt(d3.select(el).property('ref')) - 1);
        if (parseInt(d3.select(el).property('ref')) === 0) {
          d3.select(el).classed('active', false);
        }
      }
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
        .attr('display', 'none');

      // const color = ['#09adff', '#ffffff', '#ff5b09'];
      // g.append('defs')
      //   .append('linearGradient')
      //   .attr('id', 'legend-state')
      //   .attr('x1', '0').attr('y1', '0')
      //   .attr('x2', '1').attr('y2', '0')
      //   .selectAll('stop')
      //   .data(d3.range(3)).enter()
      //   .append('stop')
      //   .attr('offset', d => d/2)
      //   .attr('stop-color', d => color[d]);




      const hiddenClusters = g.selectAll('g rect')
        .data(coCluster.colClusters, (clst, i) => Array.isArray(clst) ? (String(clst.length) + String(i)) : this.id); // matching function

      const selectCluster = function (clst, i) {
        if (!clusterSelected[i]){
          clusterSelected[i] = 1;
          self.redraw_word_link(i);
          self.update_ref(d3.select(this).select('rect').node(), "plus");
          self.graph.link_info[i].forEach((l, j) => {
            self.update_ref(l['el'], "plus");
            if (d3.select(l['el']).attr('display') !== 'none') {
              self.update_ref(self.graph.word_info[j]['wordCloud'].bgHandle.node(), "plus");
            }
          });
        } else {
          clusterSelected[i] = 0;
          self.redraw_word_link(-1);
          self.update_ref(d3.select(this).select('rect').node(), "minus");
          self.graph.link_info[i].forEach((l, j) => {
            self.update_ref(l['el'], "minus");
            if (d3.select(l['el']).attr('display') !== 'none') {
              self.update_ref(self.graph.word_info[j]['wordCloud'].bgHandle.node(), "minus");
            }
          });
        }
      }

      const hGroups = hiddenClusters.enter()
        .append('g');
      hGroups
        .on('mouseenter', function (clst, i) {
          self.update_ref(d3.select(this).select('rect').node(), "plus");
          self.graph.link_info[i].forEach((l, j) => {
            self.update_ref(l['el'], "plus");
            if (d3.select(l['el']).attr('display') !== 'none') {
              self.update_ref(self.graph.word_info[j]['wordCloud'].bgHandle.node(), "plus");
              const ref = self.graph.word_info[j]['wordCloud'].bgHandle.property('ref');
              console.log(`word ${j}'s ref is ${ref}`);
            }
          })
        })
        .on('mouseleave', function(clst, i) {
          console.log('mouse leave');
          self.update_ref(d3.select(this).select('rect').node(), "minus");
          console.log(d3.select(this).select('rect').property('ref'));
          self.graph.link_info[i].forEach((l, j) => {
            self.update_ref(l['el'], "minus");
            if (d3.select(l['el']).attr('display') !== 'none') {
              self.update_ref(self.graph.word_info[j]['wordCloud'].bgHandle.node(), "minus");
              const ref = self.graph.word_info[j]['wordCloud'].bgHandle.property('ref');
              console.log(`word ${j}'s ref is ${ref}`);
            }
          })
        })
        .on('click', selectCluster)
      hGroups.attr('transform', 'translate(0,0)')
        .attr('id', (clst, i) => (String(clst.length) + String(i)))
        .transition()
        .duration(300)
        .attr('transform', (d, i) => 'translate(' + [state_info.state_cluster_info[i].top_left[0], state_info.state_cluster_info[i].top_left[1]] + ')');

      hGroups.each(function (d, i) {
        graph.state_info.state_cluster_info[i]['el'] = this;
      });

      const clusterRect = hGroups.append('rect')
        .classed('hidden-cluster', true)
        .classed('active', false)
        .property('ref', 0)
        .transition()
        .duration(400)
        .attr('width', (clst, i) => state_info.state_cluster_info[i].width)
        .attr('height', (clst, i) => state_info.state_cluster_info[i].height)
        .attr('x', 0) //(clst, i) => state_info.state_cluster_info[i].top_left[0])
        .attr('y', 0) //(clst, i) => state_info.state_cluster_info[i].top_left[1]);

      const units = hGroups.append('g')
        .selectAll('rect')
        .data(d => d);

      let tmp_units = units.enter()
        .append('rect')
        .on('mouseover', function(d, i) {
          if (state_info.state_info[d].selected) return;
            d3.select(this).classed('unit-active', true)
            // fisheye in
        })
        .on('mouseleave', function(d, i) {
          if (state_info.state_info[d].selected) return;
            d3.select(this).classed('unit-active', false)
            // fisheye out
        })
        .on('click', function(d, i) {
            if (!state_info.state_info[d].selected){
              state_info.state_info[d].selected = true;
              d3.select(this).classed('unit-active', true)
              bus.$emit(SELECT_UNIT, d, self.compare);
            } else {
              state_info.state_info[d].selected = false;
              d3.select(this).classed('unit-active', false)
              bus.$emit(DESELECT_UNIT, d, self.compare);
            }
          // }
        })
        .transition()
        .duration(400)
        .attr('width', (i) => state_info.state_info[i].width)
        .attr('height', (i) => state_info.state_info[i].height)
        .attr('x', (i) => state_info.state_info[i].top_left[0])
        .attr('y', (i) => state_info.state_info[i].top_left[1])
        .attr('fill', self.unitNormalColor)
        .attr('fill-opacity', 0.3)

      tmp_units.each(function(d) {
        graph.state_info.state_info[d]['el'] = this;
      })

      // add
      // hiddenClusters.exit()
      //   .transition()
      //   .duration(400)
      //   .style('fill-opacity', 1e-6)
      //   .attr('width', 1)
      //   .attr('height', 1)
      //   .remove();
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
          const ref = parseInt(wclst['wordCloud'].bgHandle.property('ref'));
          const selected = wclst['wordCloud'].bgHandle.property('selected');
          const active = wclst['wordCloud'].bgHandle.classed('active');
          wclst['wordCloud']
            .draw([wclst.width/2, wclst.height/2])
            .transform( 'translate(' + [wclst.top_left[0] + wclst.width/2, wclst.top_left[1] + wclst.height/2] + ')');
          wclst['wordCloud'].bgHandle.property('ref', ref)
                              .property('selected', selected)
                              .classed('wordcloud', true)
                              .classed('active', active);
        } else {
          let tmp_g = g.append('g')
            .on('mouseenter', function () {
              self.graph.link_info.forEach((ls, j) => {
                self.update_ref(ls[i]['el'], 'plus');
                if (d3.select(ls[i]['el']).attr('display') !== 'none') {
                  self.update_ref(d3.select(self.graph.state_info.state_cluster_info[j]['el']).select('rect').node(), 'plus');
                }
              });
              self.update_ref(wclst['wordCloud'].bgHandle.node(), 'plus');
            })
            .on('mouseleave', function () {
              self.graph.link_info.forEach((ls, j) => {
                self.update_ref(ls[i]['el'], 'minus');
                if (d3.select(ls[i]['el']).attr('display') !== 'none') {
                  self.update_ref(d3.select(self.graph.state_info.state_cluster_info[j]['el']).select('rect').node(), 'minus');
                }
              });
              self.update_ref(wclst['wordCloud'].bgHandle.node(), 'minus');
            })
            .on('click', function () {
              // wclst['wordCloud'].selected = ~wclst['wordCloud'].selected;
              wclst['wordCloud'].bgHandle.property('selected', ~wclst['wordCloud'].bgHandle.property('selected'))
              self.graph.link_info.forEach((ls, j) => {
                self.update_ref(ls[i]['el'], wclst['wordCloud'].bgHandle.property('selected') ? 'plus' : 'minus');
                if (d3.select(ls[i]['el']).attr('display') !== 'none') {
                  self.update_ref(d3.select(self.graph.state_info.state_cluster_info[j]['el']).select('rect').node(), wclst['wordCloud'].bgHandle.property('selected') ? 'plus' : 'minus');
                }
              });
              self.update_ref(wclst['wordCloud'].bgHandle.node(), wclst['wordCloud'].bgHandle.property('selected') ? 'plus' : 'minus');
            });
          let myWordCloud = new WordCloud(tmp_g, wclst.width/2, wclst.height/2, 'rect', this.compare)
            .transform( 'translate(' + [wclst.top_left[0] + wclst.width/2, wclst.top_left[1] + wclst.height/2] + ')')
            .color(this.params.posColor);
          myWordCloud.update(word_info[i].words_data);
          myWordCloud.bgHandle.property('ref', 0)
                              .property('selected', false)
                              .classed('wordcloud', true);
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
          .style('opacity', 0)
          .remove();
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
          .style('fill-opacity', 1e-6)
          .style('opacity', 1e-6)
          .remove();
      });
    }

    createLink(d) {
      return "M" + d.source.x + "," + d.source.y
          + "C" + (d.source.x + d.target.x) / 2 + "," + d.source.y
          + " " + (d.source.x + d.target.x) / 2 + "," + d.target.y
          + " " + d.target.x + "," + d.target.y;
    }

    flatten(arr) {
      return arr.reduce((acc, val) => {
        return acc.concat(Array.isArray(val) ? this.flatten(val) : val);
      }, []);
    }

    draw_link(g, graph) {
      const link_info = graph.link_info;

      const strength_extent = d3.extent(this.flatten(link_info).map((l) => l.strength));
      const max_strength = Math.max(Math.abs(strength_extent[0]), Math.abs(strength_extent[1]));
      const strength_bound = this.strengthThresholdPercent.map((t) => t * max_strength);

      link_info.forEach((ls, i) => {
        // const strengthRange = d3.extent(ls);
        // console.log(`strength threshold is ${this.strengthThresholdPercent}`);
        ls.forEach((l, j) => {
          if (l['el']) {
            d3.select(l['el'])
              .attr('display', (Math.abs(l.strength) < strength_bound[0] || Math.abs(l.strength) > strength_bound[1]) ? 'none' : '')
              .transition()
              .duration(300)
              .attr('stroke-width', this.strokeWidth(l.strength))
              // .attr('stroke-width', l.strength !== 0 ? this.strokeWidth(l.strength) : 0)
              .attr('d', this.createLink(l))
          } else {
            let tmp_path = g.append('path')
              .classed('link', true)
              .classed('active', false)
              .classed(l.strength > 0 ? 'positive' : 'negative', true)
              .property('ref', 0)
              .attr('d', this.createLink(l))
              .attr('stroke-width', this.strokeWidth(l.strength))
              .attr('opacity', 0.15)
              // .attr('stroke', l.strength > 0 ? this.linkColor[1] : this.linkColor[0])
              .attr('display', (Math.abs(l.strength) < strength_bound[0] || Math.abs(l.strength) > strength_bound[1]) ? 'none' : '')

            l['el'] = tmp_path.node();
          }

        });
      });
    }

    draw(coCluster) {
      let self = this;
      // console.log(coCluster.colClusters.length);
      const maxClusterSize = coCluster.colSizes.reduce((a, b) => Math.max(a,b));
      const nCluster = coCluster.labels.length;
      let clusterInterval2HeightRatio = 1.0;
      // console.log(coCluster);
      // console.log(`cluster number is ${nCluster}`);
      this.params.computeParams(coCluster.labels.length, clusterInterval2HeightRatio, coCluster.colSizes);
      // this.dx = this.params.wordCloudChord2ClusterDistance - (this.params.wordCloudChord2CenterDistance - maxClusterWidth / 2);
      this.dx = this.stateClusterWordCloudDX;
      this.dy = this.params.wordCloudChordLength / 2.0 - this.middle_line_y / 2;
      // console.log(`maxClusterWidth is ${maxClusterWidth}`);
      // console.log(`wordCloudChord2CenterDistance is ${this.params.wordCloudChord2CenterDistance}`);
      console.log(`dx is ${this.dx}, dy is ${this.dy}`);

      // this.dx = 0, this.dy = chordLength / 2;
      const coClusterAggregation = coCluster.aggregation_info;
      let state_info = this.calculate_state_info(coCluster, this.params.mode);

      self.graph = {
        state_info: state_info,
        coCluster: coCluster,
        sentence_info: [],
      }
      this.draw_state(this.hg, self.graph);

      this.redraw_word_link();

      this.translateX(0);
    }

    translateX(x) {
      this.params.middleLineOffset += x;
      this.hg.attr('transform', 'translate(' + [this.middle_line_x, this.middle_line_y] + ')');
      this.wg.attr('transform', 'translate(' + [this.middle_line_x + this.dx, this.middle_line_y + this.dy] + ')');
    }

    adjustdx(newdx) {
      this.dx = newdx;
      this.wg.attr('transform', 'translate(' + [this.middle_line_x + this.dx, this.middle_line_y + this.dy] + ')');
      this.redraw_word_link();
    }

    destroy() {
      if (!this.graph) {
        return;
      }
      this.deleteAllSentences();
      this.erase_state();
      this.erase_link();
      this.erase_word();
      this.topg.transition()
        // .attr('transform')
        .duration(300)
        .attr('transform', 'scale(0.01, 0.01)')
        .style('opacity', 1e-6)
        .remove();
      this.topg = this.svg.append('g');
      this.hg = this.topg.append('g');
      this.wg = this.topg.append('g');
    }
  }

</script>
