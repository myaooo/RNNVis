import * as d3 from 'd3';
import { SentenceRecord, CoClusterProcessor } from '../preprocess'

const layoutParams = {
  nodeInterval: 5,
  color: d3.scaleOrdinal(d3.schemeCategory20),
};

const radiusTimer = 1.5;

class SentenceLayout{
  constructor(selector, params = layoutParams){
    this.group = selector;
    this._size = [50, 600];
    this._sentence;
    this._coCluster;
    this.params = params;
  }
  size(size){
    return arguments.length ? (this._size = size, this) : this._size;
  }
  get radius() {
    const radius = this._size[0] / (radiusTimer*2);
    if (this.sentence){
      const radius2 = (this._size[1] - (this.sentence.length - 1) * this.params.nodeInterval) / this.sentence.length / (radiusTimer * 2);
      return radius < radius2 ? radius : radius2;
    }
    return radius;
  }
  sentence(sentence) {
    return arguments.length ? (this._sentence = sentence, this) : this;
  }
  coCluster(coCluster) {
    return arguments.length ? (this._coCluster = coCluster, this) : this;
  }
  words(words) {
    return arguments.length ? (this._words = words, this) : this;
  }
  layout() {
    const dataList = this.preprocess(this._sentence, this._coCluster, this._words);
    // const words = this._words;
    // console.log(dataList);
    dataList.forEach((data, i) => {
      const g = this.group.append('g');
      this.drawOneWord(g, data, i)
        .attr('transform', 'translate(' + [this._size[0] / 2, this.radius * (radiusTimer + radiusTimer * 2 * i) + i * this.params.nodeInterval] + ')');
    })
    // const groupsSelector = this.group.selectAll('g')
    //   .data(dataList);
    // const glyphGroups = groupsSelector.enter()
    //   .append('g');
    // glyphGroups
    //   .append()

  }
  drawOneWord(el, data, i) {
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
      .innerRadius((d) => { return radius * (d.data.updated < 0 ? (1 + d.data.updated*2) : 1); })
      .outerRadius((d) => { return radius * (d.data.updated < 0 ? 1 : (1 + d.data.updated*2)); });

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
    handles[1].attr('fill-opacity', (d, k) => (data.data[k].updated < 0 ? 0.3 : 0.7))
    gs[0].attr('fill-opacity', 0.7);
    // gs[1].attr('fill-opacity', 0.6);
    gs[2].attr('fill-opacity', 0.2);


    return el;
  }
  preprocess(sentence, coCluster, words) {
    const len = sentence.length;
    const clusterNum = coCluster.labels.length;
    const stateNum = sentence[0].length;
    // const info
    // let infoCurrent
    const currentStates = sentence.map((word) => {
      return coCluster.colClusters.map((cluster) => {
        return cluster.map((idx) => {
          return word[idx];
        });
      });
    });

    const infoCurrent = currentStates.map((word, t) => { // compute an array for each word
      return word.map((cluster, i) => { // compute a info for each cluster
        return cluster.reduce((a, b) => Math.abs(a) + Math.abs(b));
      })
    });
    const infoPrevious = [new Float32Array(clusterNum), ...infoCurrent.slice(0, len-1)];

    const h_tij = [currentStates[0].map((clst) => new Float32Array(clst.length)), ...currentStates];
    // console.log(h_tij);
    const infoUpdated = new Array(len);
    const infoKept = new Array(len);
    for (let t = 0; t < len; t++) {
      infoUpdated[t] = new Float32Array(clusterNum);
      infoKept[t] = new Float32Array(clusterNum);
      for (let i = 0; i < clusterNum; i++) {
        for (let j = 0; j < h_tij[t][i].length; j++){
          const prev = h_tij[t][i][j];
          const cur = h_tij[t+1][i][j];
          infoUpdated[t][i] += (cur-prev);
          // infoUpdated[t][i] += Math.sign(prev) * (cur-prev);
          const ratio = cur / prev;
          infoKept[t][i] += Math.abs(prev) * (ratio < 0 ? 0 : 1 < ratio ? 1 : ratio);
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
          updated: prev === 0 ? 0 : updated / prev,
          kept: prev === 0 ? 0 : kept / prev,
        }
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
