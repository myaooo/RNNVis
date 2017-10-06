import * as d3 from 'd3';
import {
  WordCloud,
} from '../layout/cloud';
import {
  sentence,
} from '../layout/sentence';
import {
  SELECT_UNIT,
  SELECT_WORD,
  SELECT_SENTENCE_NODE,
} from '../store';

const colorHex = ['#33a02c', '#1f78b4', '#b15928', '#fb9a99', '#e31a1c', '#6a3d9a',
  '#ff7f00', '#cab2d6', '#ffff99', '#a6cee3', '#b2df8a', '#fdbf6f'];
const colorScheme = (i) => colorHex[i];
const positiveColor = '#ff5b09';
const negativeColor = '#09adff';

// function arcLength2Angle(length, radius) {
//   return length / radius * 180 / Math.PI;
// }

function flatten(arr) {
  return arr.reduce((acc, val) =>
    acc.concat(Array.isArray(val) ? flatten(val) : val), []);
}

function updateRef(data, mode) {
  const el = data.el;
  if (mode === 'plus') {
    data.ref += 1;
    d3.select(el).classed('active', true);
  } else if (mode === 'minus') {
    data.ref -= 1;
  }
  if (data.ref === 0) {
    d3.select(el).classed('active', false);
  }
}

function boundLength(nUnits, unitLength, unitInterval, marginInterval) {
  let bound = nUnits * (unitLength + unitInterval);
  bound -= unitInterval;
  bound += marginInterval * 2;
  return bound;
}

export class LayoutParamsConstructor {
  constructor({
    width = 800,
    height = 800,
    alignMode = 'width',
    chipsSizes,
    cloudsSizes,
  }) {
    this.width = width;
    this.height = height;
    this.alignMode = alignMode;
    this.chipsSizes = chipsSizes;
    this.cloudsSizes = cloudsSizes;
    this.nChips = chipsSizes.length;
    this.nClouds = cloudsSizes.length;
    const nUnits = d3.sum(chipsSizes);
    this.posColor = colorScheme;

    // cloud related params
    this.cloudArcDegree = 110;
    this.cloudWidth = Math.floor(width * 0.25);
    this.cloudHighlightRatio = 1.5;
    this.cloudInterval = Math.max(5, Math.floor(height / 6 / this.nClouds));
    const remainHeight = this.height - (this.cloudInterval * (this.nClouds + 4));
    const nWords = d3.sum(cloudsSizes);
    this.cloudsHeights = cloudsSizes.map(size => Math.round(remainHeight * (size / nWords)));
    const cloud1Height = this.cloudsHeights[0];
    const cloud2Height = this.cloudsHeights[this.cloudsHeights.length - 1];
    this.cloudsTopSpace = Math.floor(this.cloudInterval * 2.5);
    // console.log(this.cloudsHeights);
    this.cloudChordLength = Math.floor(this.height - (2 * this.cloudsTopSpace) - ((cloud1Height + cloud2Height) / 2));
    this.cloudArcRadius = this.cloudChordLength / 2 / Math.sin((this.cloudArcDegree / 2 / 180) * Math.PI);
    this.cloudArcCenterY = this.height + (cloud1Height - cloud2Height);
    this.cloudArcCenterY /= 2;

    // sentence related params
    this.dxShrinkFactor = 0.07;
    this.spacePerSentence = 0.14;
    this.sentenceNodeWidth = 100;
    this.sentenceInitTranslate = [50, 10];
    this.brushTranslationX = -100;
    this.sentenceWordThreshold = 0.5;
    this.middleLineOffset = 0;
    this.sentenceBrushRectWidth = 10;

    this.maxChipWidth = Math.min(width / 3.5, 400);
    this.chipsCenterX = Math.floor(this.width / 3);
    this.unitHeight = 3 + Math.min(Math.floor(width*3 / (nUnits + 300)), 4);
    this.computeParams();
  }
  computeParams() {
    this.unitWidth = this.unitHeight;
    this.unitInterval = Math.floor(this.unitWidth * 0.5);
    this.chipMargin = Math.floor(this.unitWidth * 0.7);
    if (this.alignMode === 'width') {
      const nRowUnits = Math.floor(this.maxChipWidth / (this.unitWidth + this.unitInterval));
      this.chipWidth = boundLength(nRowUnits, this.unitWidth, this.unitInterval, this.chipMargin);
      this.nColUnits = this.chipsSizes.map(size => Math.ceil(size / nRowUnits));
      this.nRowUnits = Array.from({ length: this.nChips }, () => nRowUnits);
      this.chipsHeights = this.nColUnits.map(colUnits =>
        boundLength(colUnits, this.unitHeight, this.unitInterval, this.chipMargin));
      this.chipsWidths = Array.from({ length: this.nChips }, () => this.chipWidth);
    } else {
      const maxChipSize = this.chipsSizes[this.chipsSizes.length - 1];
      const nColUnits = Math.ceil(maxChipSize / (this.maxChipWidth / (this.unitWidth + this.unitInterval)));
      this.chipHeight = boundLength(nColUnits, this.unitHeight, this.unitInterval, this.chipMargin);
      this.nRowUnits = this.chipsSizes.map(size => Math.ceil(size / nColUnits));
      this.nColUnits = Array.from({ length: this.nChips }, () => nColUnits);
      this.chipsWidths = this.nRowUnits.map(rowUnits =>
        boundLength(rowUnits, this.unitWidth, this.unitInterval, this.chipMargin));
      this.chipsHeights = Array.from({ length: this.nChips }, () => this.chipHeight);
    }
    this.chipInterval = (this.height - d3.sum(this.chipsHeights)) / (this.nChips + 2);
    if (this.chipInterval * (this.nChips + 2) * 3 < this.height) {
      this.unitHeight -= 1;
      this.computeParams();
    }
    this.chipsTopSpace = Math.floor(this.chipInterval * 1.5);
    this.linkSpace = Math.floor(this.width / (this.alignMode === 'width' ? 8 : 10));
    // console.log(this.chipsTopSpace);
  }
}

export class Painter {
  constructor(selector, store, compare = false) {
    this.svg = selector;
    this.params = null;
    this.style = null;
    this.store = store;
    this.initGroups();
    this.cloudsX = 0;
    this.cloudsY = 0;
    this.graph = {};
    this.coCluster = null;
    this.clusterSelected = [];

    this.state_elements = [];
    this.loc = null;
    this.clouds = [];

    this.unitNormalColor = '#aaaaaa';
    this.unitRangeColor = ['#09adff', '#ff5b09'];
    this.linkWidthRange = [1, 5];
    this.linkColor = ['#09adff', '#ff5b09'];
    this.compare = compare;
    this.stateTranslateHis = [];
    this.sentences = [];
    this.stateClip = 2;

    this.strokeWidth = function (t) {
      return Math.abs(t);
    };
    this.sentenceStrokeWidth = function (t) {
      return Math.abs(t);
    };
    this.colorLegendAxis = null;
  }
  hasData() {
    return Boolean(this.coCluster);
  }
  layoutParams(params) {
    if (params) {
      this.params = params;
      return this;
    }
    return this.params;
  }
  styleParams(style) {
    if (style) {
      this.style = JSON.parse(JSON.stringify(style));
      return this;
    }
    return this.style;
  }
  initGroups() {
    this.topg = this.svg.append('g').attr('class', 'topGroup');
    this.hg = this.topg.append('g').attr('class', 'chipGroup');
    this.lg = this.topg.append('g').attr('class', 'linkGroup');
    this.wg = this.topg.append('g').attr('class', 'wordGroup');
    this.hg.append('line')
    .attr('id', 'chip_control_line')
    .attr('x1', 0)
    .attr('y1', -1000)
    .attr('x2', 0)
    .attr('y2', 1000);
    // .attr('display', 'none');
  }
  get dxOffset() {
    const adaptive = (this.clientWidth - this.clientHeight) / 4;
    return Math.max(adaptive, 0);
    // return 0;
  }
  get stateClusterWordCloudDX() {
    return this.clientWidth * (-this.params.dxShrinkFactor * this.sentences.length) + this.dxOffset;
  }
  transform(trans) {
    this.svg.attr('transform', trans);
  }
  get clientWidth() {
    return this.params.width;
  }
  get clientHeight() {
    return this.params.height;
  }
  get chipsCenterX() {
    return this.params.chipsCenterX;
  }
  get chipsTopSpace() {
    return this.params.chipsTopSpace;
  }

  get nSentence() {
    return this.sentences.length;
  }

  get linkFilter() {
    return this.style.linkFilterThreshold;
  }

  get cloudArcCenterY() {
    return this.params.cloudArcCenterY;
  }

  refreshStroke({ strokeControlStrength }) {
    this.sentenceStrokeWidth = function (t) {
      return Math.abs(t) * strokeControlStrength;
    };
    if (!this.graph) return;
    this.drawLink(this.graph);

    if (!this.graph.sentences) return;
    this.graph.sentences.forEach((si) => {
      const strengthExtent = d3.extent(flatten(si.edges).map((l) => l.strength));
      const maxStrength = Math.max(Math.abs(strengthExtent[0]), Math.abs(strengthExtent[1]));
      const strengthBound = this.linkFilter.map((t) => t * maxStrength);
      si.edges.forEach((ls) => {
        ls.forEach((l) => {
          l.el.attr('display',
              (Math.abs(l.strength) < strengthBound[0] || Math.abs(l.strength) > strengthBound[1]) ? 'none' : '')
            .attr('stroke-width', this.sentenceStrokeWidth(l.strength));
        });
      });
    });
  }

  changeStateClip(clip) {
    if (!this.graph) {
      return;
    }
    const { chips } = this.graph;
    this.stateClip = clip;
    const scale = d3.scaleLinear()
      .domain([-this.stateClip, 0, this.stateClip])
      .range([1, 0, 1]);
    if (d3.select(chips.units[0].el).property('colored')) {
      chips.forEach(chip => {
        chip.forEach(unit => {
          let tmp_s = d3.select(unit.el);
          tmp_s
            .transition()
            .attr('fill', tmp_s.property('strength') > 0 ? positiveColor : negativeColor)
            .attr('fill-opacity', scale(Math.abs(tmp_s.property('strength'))));
        });
      });
      this.axisScale.domain([-this.stateClip, this.stateClip]);
      this.colorLegendAxis = d3.axisBottom()
        .tickValues([parseInt(-this.stateClip, 10), 0, parseInt(this.stateClip, 10)])
        .tickFormat(d3.format(`.${d3.precisionFixed(1)}f`))
        .scale(this.axisScale);
      this.hg.select('#color-legend')
        .select('g')
        .call(this.colorLegendAxis);
    }
  }

  deleteSentence(value) {
    const sentenceToAdd = this.sentences.filter((s) => s !== value);
    this.sentences = [];
    this.translateX(-d3.sum(this.stateTranslateHis));
    this.adjustdx(this.stateClusterWordCloudDX);
    this.stateTranslateHis = [];
    this.sentenceTranslateHis = [this.params.sentenceInitTranslate[0], ];

    const tmpSentenceInfo = {};
    this.graph.sentences.forEach((s) => {
      tmpSentenceInfo[s.value] = s;
      s.group.remove();
    });
    this.graph.sentences = [];
    sentenceToAdd.forEach((s) => {
      this.addSentence(s, tmpSentenceInfo[s].record, tmpSentenceInfo[s].sentenceRecord);
    });
  }

  deleteAllSentences() {
    if (this.sentences.length === 0) return;
    this.sentences = [];
    this.translateX(-d3.sum(this.stateTranslateHis));
    this.adjustdx(this.stateClusterWordCloudDX);
    this.stateTranslateHis = [];
    this.sentenceTranslateHis = [this.params.sentenceInitTranslate[0]];

    const tmpSentenceInfo = {};
    this.graph.sentences.forEach((s) => {
      tmpSentenceInfo[s.value] = s;
      s.group.remove();
    });
    this.graph.sentences = [];
  }

  addSentence(value, record, sentenceRecord) {
    const self = this;
    const graph = this.graph;
    const needTranslate = (this.params.spacePerSentence + this.params.dxShrinkFactor) * this.clientWidth;
    const translationForEachSentence = needTranslate;
    const translationForState = needTranslate * 0.5;
    const sentenceInitTranslate = this.params.sentenceInitTranslate;
    this.stateTranslateHis.push(translationForState);
    if (this.sentences.length) {
      this.sentenceTranslateHis.push(translationForEachSentence);
    }
    // const sentenceTranslationX = this.clientWidth * 0.01;
    const sg = this.svg.append('g')
      .attr('transform', `translate(${sentenceInitTranslate[0]}, ${sentenceInitTranslate[1]})`);
    this.sentences.push(value);
    const spg = sg.append('g');
    // const sentenceTranslate = [50, 10];
    this.translateX(d3.sum(this.stateTranslateHis));
    this.adjustdx(this.stateClusterWordCloudDX);

    graph.sentences.forEach((s) => {
      s.group
        .transition()
        .attr('transform', `translate(${d3.sum(this.sentenceTranslateHis)},${sentenceInitTranslate[1]})`);
    });

    const rectGroup = sg.append('g')
      .attr('transform', `translate(${-this.params.sentenceNodeWidth / 2},${this.clientHeight / 4})`);
    this.drawBrushRect(rectGroup, sentenceRecord.length, updateSentence);

    const sent = sentence(spg, this.compare)
      .size([this.params.sentenceNodeWidth, this.params.sentenceNodeWidth * sentenceRecord.length])
      .sentence(sentenceRecord)
      .coCluster(this.coCluster)
      .words(record.tokens)
      .mouseoverCallback(highlightSentenceLinkByNodeIndex)
      .barMouseoverCallback(barHighlightState)
      .draw();

    const edges = [];
    // const strength = sent.strengthByCluster;
    sent.strengthByCluster.forEach((strengths, i) => {
      edges[i] = [];
      const wordPos = sent.getWordPos(i);
      this.coCluster.colClusters.forEach((clst, j) => {
        const strength = strengths[j];
        const s = graph.chips[j];
        edges[i][j] = {
          source: {
            x: wordPos[0] + sent.nodeWidth,
            y: wordPos[1] + sent.nodeHeight / 2
          },
          source_init: {
            x: wordPos[0] + sent.nodeWidth,
            y: wordPos[1] + sent.nodeHeight / 2
          },
          target: {
            x: s.topLeft[0] + this.chipsCenterX - sentenceInitTranslate[0],
            y: s.topLeft[1] + this.chipsTopSpace + s.height / 2 - sentenceInitTranslate[1],
          },
          strength,
        };
      });
    });

    const strengthExtent = d3.extent(flatten(edges).map((l) => l.strength));
    const maxStrength = Math.max(Math.abs(strengthExtent[0]), Math.abs(strengthExtent[1]));
    const strengthBound = this.linkFilter.map((t) => t * maxStrength);

    const lsg = sg.append('g');
    edges.forEach((ls) => {
      ls.forEach((l) => {
        l.el = lsg.append('path')
          .classed('active', false)
          .classed('link', true)
          .classed(l.strength > 0 ? 'positive' : 'negative', true)
          .attr('d', self.createLink(l))
          .attr('stroke-width', self.sentenceStrokeWidth(l.strength))
          .attr('opacity', 0.2)
          .attr('fill', 'none')
          .attr('display',
            (Math.abs(l.strength) < strengthBound[0] || Math.abs(l.strength) > strengthBound[1]) ? 'none' : '')
          .attr('hold', 'false');
      });
    });
    graph.sentences.push({
      sentence: sent,
      edges,
      value,
      group: sg,
      record,
      sentenceRecord,
    });
    if (this.sentences.length > 1) {
      graph.sentences.forEach((si) => {
        si.edges.forEach((ls) => {
          ls.forEach((l) => {
            l.el.attr('display', 'none');
          });
        });
      });
    }

    function updateSentence(extent_) {
      // console.log(`extent_ is ${extent_}`);
      const words = record.tokens.slice(...extent_);
      // console.log(`words is ${words}`);
      let scaleFactor = self.clientHeight / words.length / self.params.sentenceNodeWidth / 1.05;
      // console.log(`scaleFactor is ${scaleFactor}`);
      scaleFactor = Math.min(scaleFactor, 1.5);
      // const newHeight = scaleFactor * self.clientHeight;
      const translateY = (sent.getWordPos(extent_[0])[1] + sent.getWordPos(extent_[1])[1]) / 2;
      // let translateY = sent.getWordPos(~~((extent_[0] + extent_[1])/2))[1];
      sent.transform(`scale(${scaleFactor}) translate(0, ${-translateY + (self.clientHeight / 2 / scaleFactor)})`);
      // sent.transform('scale(' + scaleFactor + ')translate('
      // + [d3.sum(self.sentenceTranslateHis)/scaleFactor, -translateY + self.clientHeight/2/scaleFactor] + ')');
      edges.forEach((ls) => {
        ls.forEach((l) => {
          const actualY = (l.source_init.y - translateY) * scaleFactor + self.clientHeight / 2;
          console.log(`client height is ${self.clientHeight}`);
          console.log(`actualY is ${actualY}`);
          // const actualY = (l.source_init.y - translateY) * scaleFactor + self.clientHeight / 2;
          l.source.x = l.source_init.x * scaleFactor;
          l.source.y = actualY;
          l.el
            .transition()
            .attr('d', (actualY > 0 && actualY < self.clientHeight) ? self.createLink(l) : '');
        });
      });
    }

    const store = this.store;

    function highlightSentenceLinkByNodeIndex(data, t, highlight, changeCheckStatus = false) {
      // console.log(this);
      edges[t].forEach((l) => {
        if (changeCheckStatus) {
          l.el.attr('hold', l.el.attr('hold') === 'false' ? 'true' : 'false');
        }
        if (l.el.attr('hold') !== 'true') {
          l.el.classed('active', highlight)
            .attr('display', highlight || (self.sentences.length < 2 && l.el.attr('display') !== 'none') ? '' : 'none');
        } else {
          l.el.classed('active', true)
            .attr('display', '');
        }
      });
      if (changeCheckStatus) {
        data.selected = !data.selected;
        store.dispatch(SELECT_SENTENCE_NODE, {
          data,
          compare: self.compare,
        });
      }
      data.bg.classed('active', data.selected ? true : highlight);
    }

    function barHighlightState(stateIndex, highlight) {
      // console.log('state ' + stateIndex + ' is hover on or out');
      updateRef(
        d3.select(graph.chips[stateIndex].el)
        .select('rect')
        .node(), highlight ? 'plus' : 'minus');
    }
  }

  drawBrushRect(g, dataLength, func) {
    const self = this;

    const maxHeight = this.clientHeight / 2;
    const unitHeight = Math.min(maxHeight / dataLength, 50);
    console.log(`unitHeight is ${unitHeight}`);
    const rectSize = [self.params.sentenceBrushRectWidth, unitHeight];
    console.log(`rect size is ${rectSize}`);
    // const minBrushLength = 3;
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
    const brush = d3.brushY()
      .extent([
        [0, 0],
        [rectSize[0], rectSize[1] * dataLength],
      ])
      .on('end', function () {
        if (!d3.event.sourceEvent) return;
        if (!d3.event.selection) return;
        const extent = d3.event.selection;
        const lower = Math.round(extent[0] / rectSize[1]);
        const upper = Math.round(extent[1] / rectSize[1]);
        extent[0] = lower * rectSize[1];
        extent[1] = upper * rectSize[1];
        d3.select(this).transition().call(d3.event.target.move, extent);
        console.log(d3.event.selection);
        func([lower, upper]);
      });
    g.append('g')
      .call(brush)
      .call(brush.move, [0, rectSize[1] * dataLength]);
  }

  async buildChips(graph = this.graph, coCluster = this.coCluster, params = this.params) {
    const unitHeight = params.unitHeight;
    const unitWidth = params.unitWidth;
    const unitInterval = params.unitInterval;
    const chipInterval = params.chipInterval;

    const stateClusters = coCluster.getColClusters();
    let accumulateHeight = 0;
    const chips = stateClusters.map((stateCluster, i) => {
      const width = params.chipsWidths[i];
      const height = params.chipsHeights[i];
      const topLeft = [-width / 2, accumulateHeight];
      accumulateHeight += (height + chipInterval);
      const unitsPerRow = params.nRowUnits[i];
      const units = stateCluster.map((unit, j) => {
        const col = j % unitsPerRow;
        const row = Math.floor(j / unitsPerRow);
        return {
          topLeft: [col * (unitWidth + unitInterval), row * (unitHeight + unitInterval)],
          width: unitWidth,
          height: unitHeight,
          label: i,
          idx: unit,
        };
      });
      return {
        topLeft,
        width,
        height,
        units,
        edges: [],
        ref: 0,
      };
    });
    graph.chips = chips;
    return chips;
  }

  async buildClouds(graph = this.graph, coCluster = this.coCluster, params = this.params) {
    const cloudInterval = params.cloudInterval;
    const cloudWidth = params.cloudWidth;
    const words = coCluster.words;
    const aggInfo = coCluster.getAggregationInfo();
    const chordLength = params.cloudChordLength;
    const radius = params.cloudArcRadius;

    const wordClusters = coCluster.getRowClusters();
    const heights = params.cloudsHeights;
    let offset = -chordLength / 2;
    const clouds = wordClusters.map((wordCluster, i) => {
      const height = heights[i];
      const width = cloudWidth; // * wd_height[i] / max_height;
      const topLeftY = offset - (height / 2);
      const topLeftX = Math.sqrt((radius ** 2) - (offset ** 2));
      offset += ((height + heights[i + 1]) / 2) + cloudInterval;
      const data = wordCluster.map((d) => ({
        text: words[d],
        size: aggInfo.rowSingle2colCluster[d][i],
        highlighted: false,
      }));
      return {
        topLeft: [topLeftX, topLeftY],
        width,
        height,
        data,
        ref: 0,
        edges: [],
      };
    });
    graph.clouds = clouds;
    return clouds;
  }

  async buildEdges(graph = this.graph, coCluster = this.coCluster) {
    const { chips, clouds } = graph;
    const rowCluster2colCluster = coCluster.getAggregationInfo().rowCluster2colCluster;
    const colLabels = coCluster.colLabels;
    const rowLabels = coCluster.rowLabels;
    const edges = chips.map((chip, i) =>
      clouds.map((cloud, j) => {
        const edge = {
          ref: 0,
          strength: rowCluster2colCluster[rowLabels[j]][colLabels[i]],
          source: chip,
          target: cloud,
        };
        chip.edges.push(edge);
        cloud.edges.push(edge);
        return edge;
      }));
    graph.edges = edges;
    return edges;
  }

  renderState(data) {
    const chips = this.graph.chips;
    const chipInterval = this.params.chipInterval;
    if (data.length) {
      if (!this.hg.select('#color-legend').node()) {
        const state = chips[chips.length - 1];
        const g = this.hg.append('g')
          .attr('id', 'color-legend')
          .attr('transform',
            `translate(0,${state.topLeft[1] + state.height + 0.6 * chipInterval})` +
            `${this.compare ? 'scale(-1, 1)' : ''}`);
        const width = state.width;
        const height = 3.8 * 2;
        g.append('rect')
          .attr('transform', `translate(${-width / 2},${0})`)
          .transition()
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', width)
          .attr('height', height)
          .style('fill', 'url(#state-legend)');

        this.axisScale = d3.scaleLinear().range([0, width]).domain([-this.stateClip, this.stateClip]);
        this.colorLegendAxis = d3.axisBottom()
          .tickValues([parseInt(-this.stateClip, 10), 0, parseInt(this.stateClip, 10)])
          .tickFormat(d3.format(`.${d3.precisionFixed(1)}f`))
          .scale(this.axisScale);
        g.append('g')
          .classed('axis', true)
          .attr('transform', `translate(${[-width / 2, 0]})`)
          .call(this.colorLegendAxis);
      }

      // const extent = d3.extent(data);
      const scale = d3.scaleLinear()
        .domain([-this.stateClip, 0, this.stateClip])
        .range([1, 0, 1]);
      // console.log(`the extent of render state is ${extent}`);
      chips.forEach((chip) => {
        chip.units.forEach((u) => {
          d3.select(u.el)
            .property('colored', true)
            .property('strength', data[u.idx])
            .transition()
            .duration(500)
            .attr('fill', data[u.idx] > 0 ? positiveColor : negativeColor)
            .attr('fill-opacity', scale(Math.abs(data[u.idx])));
        });
      });
    } else {
      this.hg.select('#color-legend').remove();
      chips.forEach((chip) => {
        chip.units.forEach((u) => {
          d3.select(u.el)
            .property('colored', false)
            .transition()
            .duration(400)
            .attr('fill', this.unitNormalColor)
            .attr('fill-opacity', 0.3);
        });
      });
    }
  }

  renderWord(data = {}) {
    // console.log(data);
    this.graph.clouds.forEach((wordCluster) => {
      wordCluster.data.forEach((word) => {
        word.type = data[word.text];
      });
    });
    // console.log(this.graph.clouds);
    // console.log(this.graph.clouds);
    this.drawWord(this.graph);
  }

  redrawWordLink(graph = this.graph, coCluster = this.coCluster) {
    this.buildClouds(graph, coCluster, this.params).then(() => {
      this.drawWord(graph);
    })
    this.buildEdges(graph, coCluster).then(() => {
      this.drawLink(graph);
    })
  }

  drawState(graph = this.graph, g = this.hg) {
    console.log('Start drawState');
    const compare = this.compare;
    const store = this.store;
    const { chips } = graph;

    const chipMargin = this.params.chipMargin;
    const unitHeight = this.params.unitHeight;
    const unitWidth = this.params.unitWidth;
    const unitNormalColor = this.unitNormalColor;

    const stateClusters = g.selectAll('.state-cluster-group').data(chips);
    stateClusters.exit().transition().duration(500)
      .attr('opacity', 1e-6)
      .attr('transform', 'translate(0,0)')
      .remove();

    const newClusters = stateClusters.enter().append('g');
    newClusters.append('rect')
      .classed('hidden-cluster', true)
      .classed('active', false);
    newClusters.append('g')
      .classed('unit-group', true)
      .attr('transform', () => `translate(${chipMargin}, ${chipMargin})`);

    const allClusters = newClusters.merge(stateClusters)
      .classed('state-cluster-group', true)
      .on('mouseenter', (chip) => {
        updateRef(chip, 'plus');
        chip.edges.forEach((l) => {
          updateRef(l, 'plus');
          if (d3.select(l.el).attr('display') !== 'none') {
            updateRef(l.target, 'plus');
          }
        });
      })
      .on('mouseleave', (chip) => {
        updateRef(chip, 'minus');
        chip.edges.forEach((l) => {
          updateRef(l, 'minus');
          if (d3.select(l.el).attr('display') !== 'none') {
            updateRef(l.target, 'minus');
          }
        });
      })
      .on('click', (chip) => {
        chip.selected = !chip.selected;
        // this.redrawWordLink();
        updateRef(chip, chip.selected ? 'plus' : 'minus');
        chip.edges.forEach((l) => {
          updateRef(l, chip.selected ? 'plus' : 'minus');
          if (d3.select(l.el).attr('display') !== 'none') {
            updateRef(l.target, chip.selected ? 'plus' : 'minus');
          }
        });
        chip.selected = !chip.selected;
      });
    allClusters
      .transition()
      .duration(500)
      .attr('transform', (d, i) => `translate(${chips[i].topLeft})`);

    // update rectangle shapes
    allClusters.select('rect')
      .transition()
      .duration(500)
      .attr('width', (chip) => chip.width)
      .attr('height', (chip) => chip.height)
      .attr('x', 0)
      .attr('y', 0);

    // merge
    const hGroups = allClusters.select('.unit-group');
    hGroups.each(function (chip) {
      chip.el = this;
    });

    const units = hGroups
      .selectAll('rect')
      .data(d => d.units);

    units.exit().transition().duration(500)
      .attr('fill-opacity', 1e-6)
      .remove();

    const newUnits = units.enter()
      .append('rect')
      .attr('fill-opacity', 0);

    const allUnits = newUnits.merge(units)
      .on('mouseover', function (d) {
        if (d.selected) return;
        d3.select(this).classed('unit-active', true);
        // fisheye in
      })
      .on('mouseleave', function (d) {
        if (d.selected) return;
        d3.select(this).classed('unit-active', false);
        // fisheye out
      })
      .on('click', function (d) {
        if (!d.selected) {
          d.selected = true;
          d3.select(this).classed('unit-active', true);
          store.dispatch(SELECT_UNIT, {
            d,
            compare,
          });
        } else {
          d.selected = false;
          d3.select(this).classed('unit-active', false);
          store.dispatch(SELECT_UNIT, {
            d,
            compare,
          });
        }
        // }
      })
      .attr('fill', unitNormalColor);
    allUnits
      .transition()
      .duration(500)
      .attr('width', () => unitWidth)
      .attr('height', () => unitHeight)
      .attr('x', (d) => d.topLeft[0])
      .attr('y', (d) => d.topLeft[1])
      .attr('fill-opacity', 0.3);

    allUnits.each(function (d) {
      d.el = this;
    });
  }

  drawWord(graph = this.graph, g = this.wg) {
    const clouds = graph.clouds;
    // const chips = graph.chips;
    // console.log(clouds);
    // g.attr('transform', `translate(${this.cloudsX},${this.cloudArcCenterY})`)
    const cloudGroups = g.selectAll('g').data(clouds);
    const newClouds = cloudGroups.enter().append('g')
    .on('mouseenter', (cloud) => {
      cloud.edges.forEach((l) => {
        updateRef(l, 'plus');
        if (d3.select(l.el).attr('display') !== 'none') {
          updateRef(l.source, 'plus');
        }
      });
      updateRef(cloud, 'plus');
    })
    .on('mouseleave', (cloud) => {
      cloud.edges.forEach((l) => {
        updateRef(l, 'minus');
        if (d3.select(l.el).attr('display') !== 'none') {
          updateRef(l.source, 'minus');
        }
      });
      updateRef(cloud, 'minus');
    })
    .on('click', (cloud) => {
      // wclst.cloud.selected = ~wclst.cloud.selected;
      // wclst.cloud.bgHandle.property('selected', ~wclst.cloud.bgHandle.property('selected'))
      cloud.selected = !cloud.selected;
      cloud.edges.forEach((l) => {
        updateRef(l, cloud.selected ? 'plus' : 'minus');
        if (d3.select(l.el).attr('display') !== 'none') {
          updateRef(l.source, cloud.selected ? 'plus' : 'minus');
        }
      });
      updateRef(cloud, cloud.selected ? 'plus' : 'minus');
    });
    const allClouds = newClouds.merge(cloudGroups);
    allClouds.each(function (cloud) {
      cloud.g = this;
    });

    clouds.forEach((wclst) => {
      if (wclst.cloud) {
        const selected = wclst.selected;
        const active = wclst.cloud.bgHandle.classed('active');
        wclst.cloud
          .draw([wclst.width, wclst.height])
          .transform(`translate(${wclst.topLeft[0] + wclst.width / 2},${wclst.topLeft[1] + wclst.height / 2})`);
        wclst.cloud
          .property('selected', selected)
          .classed('wordcloud', true)
          .classed('active', active);
      } else {
        const myWordCloud = new WordCloud(d3.select(wclst.g), [wclst.width, wclst.height], {
          bgshape: 'rect',
          selectWordHandle: playload => this.store.dispatch(SELECT_WORD, playload),
        }, this.compare)
          .transform(`translate(${wclst.topLeft[0] + wclst.width / 2},${wclst.topLeft[1] + wclst.height / 2})`)
          .color(this.params.posColor);
        myWordCloud.update(wclst.data);
        myWordCloud.bgHandle.property('ref', 0)
          .property('selected', false)
          .classed('wordcloud', true);
        wclst.cloud = myWordCloud;
        wclst.el = myWordCloud.bgHandle.node();
      }
    });
  }

  drawLink(graph = this.graph, g = this.lg) {
    const edges = flatten(graph.edges);
    // console.log(this.linkFilter);
    const strengthExtent = d3.extent(edges.map((l) => l.strength));
    const maxStrength = Math.max(Math.abs(strengthExtent[0]), Math.abs(strengthExtent[1]));
    const strengthBound = this.linkFilter.map((t) => t * maxStrength);
    const strokeControlStrength = this.style.strokeControlStrength;

    function displayLink(l) {
      return (Math.abs(l.strength) < strengthBound[0] || Math.abs(l.strength) > strengthBound[1]);
    }

    function strokeWidth(t) {
      return Math.abs(t) * strokeControlStrength;
    };

    const links = g.selectAll('path').data(edges);

    links.exit().transition().duration(500)
      .attr('opacity', 0)
      .remove();

    const newLinks = links.enter()
      .append('path');

    const allLinks = newLinks.merge(links)
      .classed('positive', l => l.strength > 0)
      .classed('negative', l => l.strength < 0)
      .classed('link', true)
      .classed('active', false)
      .transition()
      .duration(500)
      .attr('d', l => Painter.createLink({
        x: l.source.topLeft[0] + l.source.width,
        y: l.source.topLeft[1] + (l.source.height / 2),
      }, {
        x: l.target.topLeft[0] + this.cloudsX,
        y: l.target.topLeft[1] + this.cloudsY + (l.target.height / 2),
      }))
      .attr('stroke-width', l => strokeWidth(l.strength/maxStrength))
      .attr('opacity', 0.1)
      .attr('display', l => (displayLink(l) ? 'none' : ''));

    allLinks.each(function (d) {
      d.el = this;
    });
  }

  eraseLink() {
    if (!this.graph.edges) return;
    this.graph.edges.forEach((ls) => {
      ls.forEach((l) => {
        d3.select(l.el)
          .transition()
          .duration(500)
          .style('opacity', 0)
          .remove();
      });
    });
  }

  eraseWord() {
    if (!this.graph.clouds) return;
    this.graph.clouds.forEach((w) => {
      if (w.cloud) {
        w.cloud.destroy();
        d3.select(w.g).remove();
      }
    });
  }

  eraseState() {
    if (!this.graph.chips) return;
    this.graph.chips.forEach((s) => {
      d3.select(s.el)
        .transition()
        .duration(500)
        .style('fill-opacity', 1e-6)
        .style('opacity', 1e-6)
        .remove();
    });
  }

  static createLink(source, target) {
    return `M${source.x},${source.y}` +
      `C${(source.x + target.x) / 2},${source.y}` +
      ` ${(source.x + target.x) / 2},${target.y}` +
      ` ${target.x},${target.y}`;
  }

  /**
   * Always call this when you have new coCluster data coming in and need to redraw the graph
   * @param {any} coCluster
   * @returns a Promise that will resolve when the drawing is done.
   * @memberof Painter
   */
  async draw(coCluster) {
    // clean up
    this.eraseWord();
    // this.graph = {};
    this.coCluster = coCluster;
    this.cloudsX = this.stateClusterWordCloudDX;
    this.cloudsY = this.cloudArcCenterY - this.chipsTopSpace;
    const p1 = this.buildChips(this.graph, coCluster)
      .then(() => this.drawState(this.graph));
    const p2 = this.buildClouds(this.graph, coCluster)
      .then(() => this.drawWord(this.graph));
    Promise.all([p1, p2]).then(() =>
      this.buildEdges(this.graph, coCluster))
      .then(() => this.drawLink(this.graph));
    this.translateX(0);
  }

  translateX(x) {
    console.log('Translate X');
    this.params.middleLineOffset += x;
    this.hg.transition().duration(500)
      .attr('transform', `translate(${this.chipsCenterX},${this.chipsTopSpace})`);
    this.wg.transition().duration(500)
      .attr('transform', `translate(${this.chipsCenterX + this.cloudsX},${this.cloudsY + this.chipsTopSpace})`);
    this.lg.transition().duration(500)
      .attr('transform', `translate(${this.chipsCenterX},${this.chipsTopSpace})`);
  }

  adjustdx(newdx) {
    this.cloudsX = newdx;
    this.wg.attr('transform', `translate(${this.chipsCenterX + this.cloudsX},${this.chipsTopSpace + this.cloudsY})`);
    this.redrawWordLink();
  }

  destroy() {
    if (!this.graph) {
      return;
    }
    this.deleteAllSentences();
    this.eraseState();
    this.eraseLink();
    this.eraseWord();
    this.topg.transition()
      // .attr('transform')
      .duration(500)
      .attr('transform', 'scale(0.01, 0.01)')
      .style('opacity', 1e-6)
      .remove();
    this.initGroups();
  }
}

