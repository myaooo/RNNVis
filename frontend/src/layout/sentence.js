import * as d3 from 'd3';
import { SentenceRecord, CoClusterProcessor } from '../preprocess'

const layoutParams = {
  nodeIntervalScale: 1.5,
  color: d3.scaleOrdinal(d3.schemeCategory20),
  radiusScale: 1.5,
  widthScale: 1.5,
  avgValueRange: [-0.5, 0.5],
  rulerScale: 0.3,
  markerWidthScale: 0.8,
  markerHeightScale: 0.4,
  wordSize: 12,
  labelSize: 10,
};

// example usage:
// see TestView.vue: draw3();

class SentenceLayout{
  constructor(selector, params = layoutParams){
    this.group = selector;
    this._size = [50, 600];
    this._rectSize = [20, 50];
    this._sentence;
    this._coCluster;
    this.params = params;
    // this.handles = [];
    this._dataList = [];
    this.type = 'bar2';
    this._mouseoverCallback = function(_) {console.log(_)};
    // each data in data list has 3 handles after drawing:
    // el: the group holding all elements of a word
    // els: 3 groups, each holds a pie chart
    // handles: 3 selector, each holds all paths in a pie chart
  }
  mouseoverCallback(func) {
    return arguments.length ? (this._mouseoverCallback = func, this) : this._mouseoverCallback;
  }
  size(size){
    return arguments.length ? (this._size = size, this) : this._size;
  }
  transform(transformStr) {
    this.group
      .transition()
      .duration(200)
      .attr('transform', transformStr);
    return this;
  }
  get radius() {
    if (!this._radius){
      const radius = this._size[0] / (this.params.widthScale*2);
      if (this._sentence && !this._radius){
        const radius2 = this._size[1] /
          (this.params.nodeIntervalScale * (this._sentence.length - 1) + this.params.radiusScale * 2 * this._sentence.length);
        // const radius2 = (this._size[1] - (this.sentence.length - 1) * this.params.nodeIntervalScale * radius) /
        //   (this.sentence.length * this.params.radiusScale * 2);
        // console.log(this.sentence.length)
        // console.log(radius2);
        this._radius = radius < radius2 ? radius : radius2;
      } else {
        return radius;
      }
    }
    return this._radius;
  }
  get nodeHeight() {
    return this.params.radiusScale * 2 * this.radius;
  }
  get nodeWidth() {
    return this.params.widthScale * 2 * this.radius;
  }
  get nodeInterval() {
    return this.params.nodeIntervalScale * this.radius;
  }
  get dataList() {
    return this._dataList;
  }
  // set the sentence data used
  sentence(sentence) {
    return arguments.length ? (this._sentence = sentence, this) : this._sentence;
  }
  // set the coCluster data used
  coCluster(coCluster) {
    return arguments.length ? (this._coCluster = coCluster, this) : this._coCluster;
  }
  // set the words that used
  words(words) {
    return arguments.length ? (this._words = words, this) : this._words;
  }
  // start to layout words
  draw(type=this.type, ) {
    this.type = type;
    // prepare
    this.prepareDraw(type);

    // draw
    if (type === 'pie') {
      this._dataList.forEach((data, i) => {
        const pos = this.getWordPos(i);
        const g = this.group.append('g');
        this.drawWord(g, data, i)
          .attr('transform', 'translate(' + pos + ')');
      });
    } else {
      this._dataList.forEach((data, i) => {
        const pos = this.getWordPos(i);
        if(i > 0){
          const gl = this.group.append('g');
          this.drawConnection(gl, data, i)
            .attr('transform', 'translate(' + [pos[0], pos[1] - this.nodeInterval] + ')')
        }
        const g = this.group.append('g');
        this.drawWord(g, data, i)
          .attr('transform', 'translate(' + pos + ')');
      });
    }
    // const rectGroup = this.group.append('g');
    // this.drawWordRect(rectGroup, this._dataList, this._rectSize, this.extentChangeCallback())
    //   .attr('transform', 'translate(-50, 50)');

    return this;
  }

  drawWordRect(g, data, rectSize, func) {
    data.forEach((d, i) => {
      g.append('text')
        .text(d.word)
        .style('text-anchor', 'middle')
        .attr('transform', 'rotate(90)translate(' + [rectSize[1] * i + rectSize[1] / 2, -rectSize[0]/4] + ')');

      g.append('rect')
      .attr('x', 0)
      .attr('y', i * rectSize[1])
      .attr('width', rectSize[0])
      .attr('height', rectSize[1])
      .attr('fill', 'lightgray')
      .attr('stroke-width', 2)
      .attr('stroke', 'blue')
      .attr('opacity', 0.2);
    });
    g.append('g').call(
      d3.brushY()
        .extent([[0, 0], [rectSize[0], rectSize[1] * data.length]])
        .on('end', function() {
          if (!d3.event.sourceEvent) return;
          if (!d3.event.selection) return;
          let extent = d3.event.selection;
          extent[0] = Math.round(extent[0] / rectSize[1]) * rectSize[1];
          extent[1] = Math.round(extent[1] / rectSize[1]) * rectSize[1];
          d3.select(this).transition().call(d3.event.target.move, extent);
          func([Math.round(extent[0] / rectSize[1]), Math.round(extent[1] / rectSize[1])])
          // console.log(d3.event.selection);
        })
    );

    return g;
  }

  prepareDraw(type = this.type) {
    if (this._dataList.length !== this._sentence.length || this._dataList[0].data[0].current.length != this._coCluster.labels.length)
      this._dataList = this.preprocess(this._sentence, this._coCluster, this._words);
    else
      this.clean();
    // adaptive height
    let maxValue = 0.1;
    this.dataList.forEach((data) => {
      data.data.forEach((clst) => {
        const clstMaxP =  clst.currents[0] - (clst.updateds[0] < 0 ? clst.updateds[0] : 0);
        const clstMaxN =  clst.currents[1] - (clst.updateds[1] < 0 ? clst.updateds[1] : 0);
        const clstMax = Math.max(clstMaxP/clst.size, clstMaxN/clst.size);
        maxValue = maxValue < clstMax ? clstMax : maxValue;
      })
    });
    maxValue = Math.ceil(maxValue * 11) / 10;
    this.params.avgValueRange = [-maxValue, maxValue];
    if (type === 'bar'){
      this.scaleHeight = d3.scaleLinear()
        .range([-this.nodeHeight/2, this.nodeHeight/2])
        .domain(this.params.avgValueRange);
      let stateSize = 0;
      for( let j = 0; j < this._dataList[0].data.length; j++) stateSize += this._dataList[0].data[j].size;
      this.scaleWidth = d3.scaleLinear()
      .range([0, this.nodeWidth])
      .domain([0, stateSize]);
    }
    else if (type === 'bar2')
      this.scaleHeight = d3.scaleLinear()
        .range([-this.nodeHeight/2, this.nodeHeight/2])
        .domain(this.params.avgValueRange);
    this.drawWord = type === 'pie' ? this.drawOneWordPie : (type === 'bar' ? this.drawOneWordBar : this.drawOneWordBar2);
    this.drawConnection = type === 'bar' ? this.drawOneConnection : (type === 'bar2' ? this.drawOneConnection2 : null);

  }
  // remove all the elements, all the preprocessed data are kept
  clean() {
    this._dataList.forEach(data => {
      if (data.el)
        data.el
          .transition()
          .duration(300)
          .style('opacity', 0)
          .remove();
      // data.el = null;
      // data.els = null;
      // data.handles = null;
    });
  }
  destroy() {
    this.clean();
    this._dataList = [];
  }
  // get the position [x, y] of a word regarding this.group
  getWordPos(i){
    // if (this.type === 'bar')
    //   return [this._size[0] / 2, this.radius * (this.params.radiusScale * (1 + 2 * i)) + i * this.nodeInterval];
    // else
      return [0, (this.nodeHeight + this.nodeInterval) * i]
  }
  // draw one word using bar chart
  // data: {word: 'he', data: Array}
  drawOneWordBar(el, data, i) {
    const height = this.nodeHeight;
    const width = this.nodeWidth;
    const color = this.params.color;
    console.log(data);
    const scaleHeight = this.scaleHeight;
    const scaleWidth = this.scaleWidth;

    const bg = el.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', width)
      .attr('height', height)
      .attr('stroke', 'gray')
      .attr('stroke-width', 0.5)
      .attr('fill', 'none');

    const gSelector = el.selectAll('g')
      .data(data.data);
    const g1 = gSelector.enter()
      .append('g');
    const cur = g1.append('rect')
      .attr('x', (d) => scaleWidth(d.accumulate))
      .attr('y', (d) => height - scaleHeight(d.prev / d.size))
      .attr('width', (d) => scaleWidth(d.size))
      .attr('height', (d) => scaleHeight(d.prev / d.size))
      .attr('fill', (d, j) => color(j))
    g1.style('fill-opacity', 0.4)
      .style('stroke', 'gray')
      // .style('stroke-opacity', 0.5)
      .style('stroke-width', 0.5)

    const g2 = gSelector.enter()
      .append('g');
    const updated = g2.append('rect')
      .attr('x', (d) => scaleWidth(d.accumulate))
      .attr('y', (d) => height - (d.updated * d.major < 0 ? scaleHeight(d.prev / d.size) : (scaleHeight((d.prev + d.updated * d.major) / d.size))))
      .attr('width', (d) => scaleWidth(d.size))
      .attr('height', (d) => scaleHeight(Math.abs(d.updated) / d.size))
      .attr('fill', (d, j) => d.updated * d.major < 0 ? 'white' : color(j));
    g2.style('fill-opacity', 0.8)
      .style('stroke', 'gray')
      .style('stroke-width', 0.5)

    data.el = el; // bind group
    return el;
  }

  drawOneWordBar2(el, data, t) {
    const self = this;
    const height = this.nodeHeight;
    const width = this.nodeWidth;
    const color = this.params.color;
    console.log(data);
    const scaleHeight = this.scaleHeight;
    const unitWidth = this.nodeWidth / data.data.length;

    // bounding box
    const bg = el.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', width)
      .attr('height', height)
      .attr('stroke', 'gray')
      .attr('stroke-width', 1)
      .attr('fill', 'none')
      // .on('mouseover', function() {console.log(`mourseover on ${t}`)})
      .on('mouseover', function() {self._mouseoverCallback(t)})

    const gSelector = el.selectAll('g')
      .data(data.data);
    const gCurrent = gSelector.enter()
      .append('g');
    const cur = gCurrent.append('rect')
      .attr('x', (d, i) => unitWidth*i)
      .attr('y', (d) => height/2 - scaleHeight(d.currents[0] / d.size))
      .attr('width', (d) => unitWidth)
      .attr('height', (d) => scaleHeight((d.currents[0]-d.currents[1]) / d.size))
      .attr('fill', (d, j) => color(j))
    gCurrent.style('fill-opacity', 0.4)
      .style('stroke', 'gray')
      // .style('stroke-opacity', 0.5)
      .style('stroke-width', 0.5);

    const gUpdated1 = gSelector.enter()
      .append('g');
    const updated1 = gUpdated1.append('rect')
      .attr('x', (d, i) => unitWidth * i)
      .attr('y', (d) => height/2 + scaleHeight(-d.currents[1] / d.size))
      .attr('width', (d) => unitWidth)
      .attr('height', (d) => scaleHeight(Math.abs(d.updateds[1]) / d.size))
      .attr('transform', (d) => d.updateds[1] < 0 ? ('translate(' + [0, -scaleHeight(Math.abs(d.updateds[1]) / d.size) ] + ')') : '')
      .attr('fill', (d, j) => d.updateds[1] > 0 ? 'none' : color(j))
      .style('stroke-opacity', (d, j) => d.updateds[1] > 0 ? 0.6 : 0.8);
    gUpdated1 //.style('fill-opacity', 0.8)
      .style('stroke-width', 0.5)
      .style('stroke', 'gray')
      .style('fill-opacity', 0.4);


    const gUpdated2 = gSelector.enter()
      .append('g');
    const updated2 = gUpdated2.append('rect')
      .attr('x', (d, i) => unitWidth * i)
      .attr('y', (d) => height/2 - scaleHeight(d.currents[0] / d.size))
      .attr('width', (d) => unitWidth)
      .attr('height', (d) => scaleHeight(Math.abs(d.updateds[0]) / d.size))
      .attr('transform', (d) => d.updateds[0] < 0 ? ('translate(' + [0, -scaleHeight(Math.abs(d.updateds[0]) / d.size) ] + ')') : '')
      .attr('fill', (d, j) => d.updateds[0] < 0 ? 'none' : color(j))
      .style('stroke-opacity', (d, j) => d.updateds[1] < 0 ? 0.6 : 0.8);
    gUpdated2 //.style('fill-opacity', 0.8)
      .style('stroke-width', 0.5)
      .style('stroke', 'gray')
      .style('fill-opacity', 0.5);


    el.append('path').attr('d', 'M0 ' + height/2 + ' H ' + width)
      .style('stroke', 'black').style('stroke-width', 0.5);

    // append labels
    const fontSize = this.params.wordSize;
    const labelSize = this.params.labelSize;
    el.selectAll('text')
      .data(this.params.avgValueRange).enter()
      .append('text')
      .attr('x', -2)
      .attr('y', (d, i) => i*(height-4)+5)
      .attr('text-anchor', 'end')
      .attr('font-size', labelSize)
      .text((d) => d);

    el.append('text')
      .attr('x', -2-fontSize)
      .attr('y', (height/2))
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(' + [90, -2-fontSize, (height/2)] + ')')
      .attr('font-size', fontSize)
      .text(data.word);

    cur.each(function(d) {
      if(d.els) d.els[0] = this;
      else d.els = [this];
    });
    updated1.each(function(d) { d.els[1] = this; });
    updated2.each(function(d) { d.els[2] = this; });
    data.els = [cur, updated1, updated2];
    data.el = el; // bind group
    return el;
  }

  drawOneConnection2(el, data, t) {
    const height = this.nodeInterval;
    // const width = this.nodeWidth;
    const color = this.params.color;
    // console.log(data);
    // const scaleHeight = this.scaleHeight;
    const unitWidth = this.nodeWidth / data.data.length;
    const rulerWidth = this.params.rulerScale * unitWidth;
    const markerWidth = this.params.markerWidthScale * unitWidth;
    const markerHeight = this.params.markerHeightScale * unitWidth;
    const gs = el.selectAll('g')
      .data(data.data).enter()
      .append('g');
    const rulers1 = gs.append('rect')
      .attr('x', (d, i) => unitWidth * (i+0.5) - rulerWidth/2).attr('y', 0)
      .attr('width', rulerWidth).attr('height', (d) => (height-markerHeight) * d.keptRate)
      .style('stroke', 'none').style('fill', (d, i) => color(i)).style('fill-opacity', 0.5);

    const rulers2 = gs.append('rect')
      .attr('x', (d, i) => unitWidth * (i+0.5) - rulerWidth/2).attr('y', (d) => (height-markerHeight) * d.keptRate)
      .attr('width', rulerWidth).attr('height', (d) => height - (height-markerHeight) * d.keptRate)
      .style('stroke', 'none').style('fill', 'gray').style('fill-opacity', 0.5);

    // const markers = gs.append('rect')
    //   .attr('x', (d, i) => unitWidth * (i+0.5) - markerWidth/2).attr('y', (d) => height * (d.keptRate))
    //   .attr('width', markerWidth).attr('height', markerHeight);

    const markers = gs.append('path')
      .attr('d', (d, i) => {
        return 'M' + (unitWidth * (i+0.5) - markerWidth/2) + ' ' + ((height-markerHeight) * d.keptRate)
          + ' ' + 'H' + ' ' + (unitWidth * (i+0.5) + markerWidth/2)
          + ' ' + 'L' + ' ' + (unitWidth * (i+0.5)) + ' ' + ((height-markerHeight) * d.keptRate + markerHeight);
      });

    return el;
  }

  drawOneConnection(el, data, i) {
    const height = this.nodeInterval;
    const width = this.nodeWidth;
    const color = this.params.color;
    console.log(data);
    // const scaleHeight = this.scaleHeight;
    const scaleWidth = this.scaleWidth;
    const calPoints = (clst) => {
      const arr = new Array(4);
      const mar = (1 - clst.keptRate) * clst.size / 2;
      arr[0] = [scaleWidth(clst.accumulate), 0];
      arr[1] = [scaleWidth(clst.accumulate + clst.size), 0];
      arr[2] = [scaleWidth(clst.accumulate + clst.size - mar), height];
      arr[3] = [scaleWidth(clst.accumulate + mar), height];
      return arr[0] + ' ' + arr[1] + ' ' + arr[2] + ' ' + arr[3];
    }
    el.selectAll('polygon')
      .data(data.data).enter()
      .append('polygon')
      .attr('points', (d) => calPoints(d))
      .attr('fill', (d, j) => color(j))
      .attr('fill-opacity', 0.6);
    return el;
  }
  // draw one word
  drawOneWordPie(el, data, i) {
    const radius = this.radius;
    const color = this.params.color;
    console.log(data);
    let arc1 = d3.arc()
      .innerRadius(1)
      .outerRadius((d) => {
        // console.log(d);
        return radius * d.data.kept;
      });

    let arc2 = d3.arc()
      .innerRadius((d) => radius * d.data.kept)
      .outerRadius(radius);

    let arc3 = d3.arc()
      .innerRadius((d) => { return radius * (d.data.updatedRate < 0 ? (1 + d.data.updatedRate*2) : 1); })
      .outerRadius((d) => { return radius * (d.data.updatedRate < 0 ? 1 : (1 + d.data.updatedRate*2)); });

    let arcs = [arc1, arc3, arc2];
    let pie = d3.pie()
      .sort(null)
      .value((d) => (d.prev ? d.prev : d.current));

    const gs = new Array(3);
    const handles = new Array(3);
    for (let j = 0; j < 3; j++){
      gs[j] = el.append('g');
      if (i === 0 && j === 0)
        continue;
      handles[j] = gs[j].selectAll(".arc")
        .append('g')
        .classed('arc', true)
        .data(pie(data.data)).enter()
        .append("path")
        .attr("d", arcs[j])
        .attr('stroke', 'gray')
        .attr('stroke-width', 0.3)
        .attr('fill', (d, k) => (j === 2 ? 'gray' : color(k)));
    };
    handles[1].attr('fill-opacity', (d, k) => (data.data[k].updatedRate < 0 ? 0.3 : 0.7))
    gs[0].attr('fill-opacity', 0.7);
    // gs[1].attr('fill-opacity', 0.6);
    gs[2].attr('fill-opacity', 0.0);

    data.els = gs; // 3 groups, each group corresponds to each ring of pie chart
    data.handles = handles; // 3 handles, each handle is a selector of paths of the pie chart
    data.el = el; // bind group
    return el;
  }
  preprocess(sentence, coCluster, words) {
    const len = sentence.length;
    const clusterNum = coCluster.labels.length;
    const stateNum = sentence[0].length;
    const clustersSize = coCluster.colClusters.map((clst) => {
      return clst.length;
    });
    const accClustersSize = new Float32Array(clustersSize.length);
    for (let i = 1; i < accClustersSize.length; i++)
      accClustersSize[i] += accClustersSize[i-1] + clustersSize[i-1];
    // const info
    // let infoCurrent
    const currentStates = sentence.map((word) => {
      return coCluster.colClusters.map((cluster) => {
        return cluster.map((idx) => {
          return word[idx];
        });
      });
    });

    const infoPositive = new Array(len);
    const infoNegative = new Array(len);
    const infoCurrent = new Array(len);
    for (let t = 0; t < len; t++) {
      infoPositive[t] = new Float32Array(clusterNum);
      infoNegative[t] = new Float32Array(clusterNum);
      infoCurrent[t] = new Float32Array(clusterNum);
      for (let i = 0; i < clusterNum; i++) {
        for (let j = 0; j < clustersSize[i]; j++) {
          if (currentStates[t][i][j] > 0) {
            infoPositive[t][i] += currentStates[t][i][j];
          } else {
            infoNegative[t][i] += currentStates[t][i][j];
          }
          infoCurrent[t][i] += Math.abs(currentStates[t][i][j]);
        }
      }
    }

    const infoPrevious = [new Float32Array(clusterNum), ...infoCurrent.slice(0, len-1)];
    const h_tij = [currentStates[0].map((clst) => new Float32Array(clst.length)), ...currentStates];
    // console.log(h_tij);
    const infoKept = new Array(len);
    const keptPositive = new Array(len);
    const keptNegative = new Array(len);
    for (let t = 0; t < len; t++) {
      infoKept[t] = new Float32Array(clusterNum);
      keptPositive[t] = new Float32Array(clusterNum);
      keptNegative[t] = new Float32Array(clusterNum);
      for (let i = 0; i < clusterNum; i++) {
        for (let j = 0; j < clustersSize[i]; j++){
          const prev = h_tij[t][i][j];
          const cur = h_tij[t+1][i][j];
          const ratio = cur / prev;
          infoKept[t][i] += Math.abs(prev) * (ratio < 0 ? 0 : 1 < ratio ? 1 : ratio);
          keptPositive[t][i] += prev > 0 ? (cur > 0 ? (cur > prev ? prev : cur) : 0) : 0;
          keptNegative[t][i] += prev < 0 ? (cur < 0 ? (cur < prev ? prev : cur) : 0) : 0;
        }
      }
    }

    const dataList = words.map((word, t) => {
      const data = Array.from(infoCurrent[t], (current, i) => {
        const prev = infoPrevious[t][i];
        // const kept = infoKept[t][i];
        const keptP = keptPositive[t][i];
        const keptN = keptNegative[t][i];
        const kept = keptP - keptN;
        const positive = infoPositive[t][i];
        const negative = infoNegative[t][i];
        const prevPositive = t > 0 ? infoPositive[t-1][i] : 0;
        const prevNegative = t > 0 ? infoNegative[t-1][i] : 0;
        const updatedPositive = positive - prevPositive;
        const updatedNegative = negative - prevNegative;
        const updated = updatedPositive + updatedNegative;
        return {
          currents: [positive, negative],
          current: current,
          prev: prev,
          updatedRate: prev === 0 ? 0 : updated / prev,
          keptRate: prev === 0 ? 0 : kept / prev,
          updated: updated,
          updateds: [updatedPositive, updatedNegative],
          kepts: [keptP, keptN],
          kept: kept,
          size: clustersSize[i],
          accumulate: accClustersSize[i],
          major: Math.sign(positive + negative),
        };
      });
      return {
        word: word,
        data: data,
      };
    });
    return dataList;
  }
  get strengthByCluster() {
    if (!this._sentence || !this._coCluster) return undefined;
    if (this._dataList.length !== this._sentence.length || this._dataList[0].data[0].current.length != this._coCluster.labels.length)
      this._dataList = this.preprocess(this._sentence, this._coCluster, this._words);
    if (!this._strengthByCluster) {
      this._strengthByCluster = this._dataList.map((word, i) => {
        return word.data.map((clst, j) => {
          return clst.updated;
        });
      });
    }
    return this._strengthByCluster;
  }
};

function sentence(selector){
  return new SentenceLayout(selector);
};

export {
  sentence,
}
