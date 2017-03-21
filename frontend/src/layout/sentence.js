import * as d3 from 'd3';
import { SentenceRecord, CoClusterProcessor } from '../preprocess'

const layoutParams = {
  nodeIntervalScale: 1.5,
  color: d3.scaleOrdinal(d3.schemeCategory20),
  radiusScale: 1.5,
  widthScale: 1.5,
  avgValueRange: [0, 1.6],
};

// example usage:
// see TestView.vue: draw3();

class SentenceLayout{
  constructor(selector, params = layoutParams){
    this.group = selector;
    this._size = [50, 600];
    this._sentence;
    this._coCluster;
    this.params = params;
    // this.handles = [];
    this.dataList = [];
    // each data in data list has 3 handles after drawing:
    // el: the group holding all elements of a word
    // els: 3 groups, each holds a pie chart
    // handles: 3 selector, each holds all paths in a pie chart
  }
  size(size){
    return arguments.length ? (this._size = size, this) : this._size;
  }
  get radius() {
    if (!this._radius){
      const radius = this._size[0] / (this.params.widthScale*2);
      if (this.sentence && !this._radius){
        const radius2 = (this._size[1] - (this.sentence.length - 1) * this.params.nodeIntervalScale * radius) /
          (this.sentence.length * this.params.radiusScale * 2);
        this._radius = radius < radius2 ? radius : radius2;
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
  get dataLength() {
    return this.dataList.length;
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
  draw(type='bar') {
    if (this.dataList.length !== this._sentence.length)
      this.dataList = this.preprocess(this._sentence, this._coCluster, this._words);
    else
      this.clean();
    // prepare
    this.scaleHeight = d3.scaleLinear()
      .range([0, this.nodeHeight])
      .domain(this.params.avgValueRange);
    let stateSize = 0;
    for( let j = 0; j < this.dataList[0].data.length; j++) stateSize += this.dataList[0].data[j].size;
    this.scaleWidth = d3.scaleLinear()
      .range([0, this.nodeWidth])
      .domain([0, stateSize]);

    // draw
    if (type === 'pie') {
      this.dataList.forEach((data, i) => {
        const pos = this.getWordPos(i);
        const g = this.group.append('g');
        this.drawOneWordPie(g, data, i)
          .attr('transform', 'translate(' + pos + ')');
      });
      return;
    }
    this.dataList.forEach((data, i) => {
      const pos = this.getWordPos(i);
      if(i > 0){
        const gl = this.group.append('g');
        this.drawOneConnection(gl, data, i)
          .attr('transform', 'translate(' + [pos[0], pos[1] - this.nodeInterval] + ')')
      }
      const g = this.group.append('g');
      this.drawOneWordBar(g, data, i)
        .attr('transform', 'translate(' + [pos[0], pos[1]] + ')');
    });
  }
  // remove all the elements, all the preprocessed data are kept
  clean() {
    this.dataList.forEach(data => {
      if (data.el)
        data.el.remove();
      data.el = null;
      data.els = null;
      data.handles = null;
    });
  }
  destroy() {
    this.clean();
    this.dataList = [];
  }
  // get the position [x, y] of a word regarding this.group
  getWordPos(i){
    return [this._size[0] / 2, this.radius * (this.params.radiusScale * (1 + 2 * i)) + i * this.nodeInterval];
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
      .attr('y', scaleHeight(this.params.avgValueRange[0]))
      .attr('width', width)
      .attr('height', scaleHeight(this.params.avgValueRange[1]))
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

    // const infoPositive = new Array(len);
    // const infoNegative = new Array(len);
    // const infoCurrent = new Array(len);
    // for (let t = 0; t < len; t++) {
    //   infoPositive[t] = new Float32Array(clusterNum);
    //   infoNegative[t] = new Float32Array(clusterNum);
    //   infoCurrent[t] = new Float32Array(clusterNum);
    //   for (let i = 0; i < clusterNum; i++) {
    //     for (let j = 0; j < clustersSize[i]; j++) {
    //       if (currentStates[t][i][j] > 0) {
    //         infoPositive[t][i] += currentStates[t][i][j];
    //       } else {
    //         infoNegative[t][i] += currentStates[t][i][j];
    //       }
    //       infoCurrent[t][i] += Math.abs(currentStates[t][i][j]);
    //     }
    //   }
    // }

    const infoCurrent = currentStates.map((word, t) => { // compute an array for each word
      return word.map((cluster, i) => { // compute a info for each cluster
        let absSum = 0;
        for(let j = 0; j < cluster.length; j++)
          absSum += Math.abs(cluster[j]);
        return absSum;
      })
    });

    const infoPrevious = [new Float32Array(clusterNum), ...infoCurrent.slice(0, len-1)];
    console.log(infoPrevious);
    const h_tij = [currentStates[0].map((clst) => new Float32Array(clst.length)), ...currentStates];
    // console.log(h_tij);
    const infoUpdated = new Array(len);
    const infoKept = new Array(len);
    const major = new Array(len);
    for (let t = 0; t < len; t++) {
      infoUpdated[t] = new Float32Array(clusterNum);
      infoKept[t] = new Float32Array(clusterNum);
      major[t] = new Float32Array(clusterNum);
      for (let i = 0; i < clusterNum; i++) {
        for (let j = 0; j < clustersSize[i]; j++){
          const prev = h_tij[t][i][j];
          const cur = h_tij[t+1][i][j];
          infoUpdated[t][i] += (cur-prev);
          // infoUpdated[t][i] += Math.sign(prev) * (cur-prev);
          const ratio = cur / prev;
          infoKept[t][i] += Math.abs(prev) * (ratio < 0 ? 0 : 1 < ratio ? 1 : ratio);
          major[t][i] += cur;
        }
      }
    }

    return words.map((word, t) => {
      const data = infoCurrent[t].map((current, i) => {
        const prev = infoPrevious[t][i];
        const updated = infoUpdated[t][i];
        const kept = infoKept[t][i];
        return {
          current: current,
          prev: prev,
          updatedRate: prev === 0 ? 0 : updated / prev,
          keptRate: prev === 0 ? 0 : kept / prev,
          updated: updated,
          kept: kept,
          size: clustersSize[i],
          accumulate: accClustersSize[i],
          major: Math.sign(major[t][i]),
        };
      });
      return {
        word: word,
        data: data,
      };
    });
  }
};

function sentence(selector){
  return new SentenceLayout(selector);
};

export {
  sentence,
}
