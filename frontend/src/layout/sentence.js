import * as d3 from 'd3';
import { SentenceRecord, CoClusterProcessor } from '../preprocess'

const layoutParams = {
  nodeInterval: 5
};

class SentenceLayout{
  constructor(selector, params = layoutParams){
    this.group = selector;
    this._size = [50, 600];
    this.sentence;
    this.params = params;
  }
  size(size){
    return arguments.length ? (this._size = size, this) : this._size;
  }
  get radius() {
    const radius = this._size[0] / 4;
    if (this.sentence){
      const radius2 = (this._size[1] - (this.sentence.length - 1) * this.params.nodeInterval) / this.sentence.length / 4;
      return radius < radius2 ? radius : radius2;
    }
    return radius;
  }
  layout(sentence, coCluster) {
    this.sentence = sentence;
    const len = sentence.length;
    // const info
  }
};

export default function sentence(selector){
  return new SentenceLayout(selector);
};
