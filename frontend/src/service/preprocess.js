export class SentenceRecord {
  constructor({
    tokens,
    records,
  }) {
    this.tokens = tokens[0]; // assume one sentence
    this.records = records[0];
    this.states = undefined;
    this.layerNum = undefined;
  }
  getStates() {
    if (this.records && !this.states) {
      this.states = Object.keys(this.records[0]);
    }
    return this.states;
  }
  getLayerNum() {
    if (this.records && !this.layerNum) {
      this.layerNum = this.records[0][this.states[0]].length;
    }
    return this.layerNum;
  }
  getRecords(stateName, layer = -1) {
    if (this.records) {
      const selectedLayer = layer === -1 ? this.layerNum - 1 : layer;
      return this.records.map((word) => word[stateName][selectedLayer]);
    }
    return undefined;
  }
}

export class StateProcessor {
  constructor(data) {
    this.data = data;
    this.stateNum = this.data.mean[0].length;
    this.statesData = undefined;
    this.wordsData = undefined;
    this.word2Id = undefined;
  }

  getStatesData() { // calculate statistics for each state unit
    if (this.data && !this.statesData) {
      this.statesData = this.data.mean[0].map((_, j) => ({
        words: this.data.words,
        // freqs: this.data.freqs,
        mean: this.data.mean.map((m) => m[j]),
        low1: this.data.low1.map((m) => m[j]),
        low2: this.data.low2.map((m) => m[j]),
        high1: this.data.high1.map((m) => m[j]),
        high2: this.data.high2.map((m) => m[j]),
        rank: this.data.sort_idx.map((indices) => indices.findIndex((idx) => (idx === j))),
      }));
    }
    return this.statesData;
  }
  getWord2Id() {
    if (this.data && !this.word2Id) {
      const word2Id = {};
      this.data.words.forEach((word, i) => {
        word2Id[word] = i;
      });
      this.word2Id = word2Id;
    }
    return this.word2Id;
  }
  getWordsData() {
    if (this.data && !this.wordsData) {
      this.wordsData = this.data.mean.map((mean, i) => {
        const data = {
          mean,
          range1: this.data.low1[i].map((low, j) => [low, this.data.high1[i][j]]),
          range2: this.data.low2[i].map((low, j) => [low, this.data.high2[i][j]]),
          word: this.data.words[i],
          sort_idx: this.data.sort_idx[i],
        };
        return data;
      });
    }
    return this.wordsData;
  }
  statOfWord(word) {
    if (this.data) {
      const id = this.getWord2Id()[word];
      return this.getWordsData()[id];
    }
    return undefined;
  }
}

function create2DArray(rowNum, colNum) {
  return Array.from({
    length: rowNum,
  }, () => new Float32Array(colNum)); // Array.from({ length: colNum }, (v, i) => 0);
}

function label2Index(labels) {
  const mapper = new Array(labels.length);
  labels.forEach((label, i) => {
    mapper[label] = i;
  });
  return mapper;
}

export class CoClusterProcessor {

  constructor({ data, params }) {
    this.aggregationInfo = undefined;
    this.colClusters = undefined;
    this.rowClusters = undefined;
    this.rawData = data;
    this.params = params;
    // preprocess
    this.correlation = data.data;
    this.rowData = data.row;
    this.colData = data.col;

    this.rowLabels = [...new Set([...(this.rowData)])].sort((a, b) => a - b);
    this.colLabels = [...new Set([...(this.colData)])].sort((a, b) => a - b);
    this.nRowCluster = this.rowLabels.length;
    this.nColCluster = this.colLabels.length;

    const colSizes = new Int32Array(this.colLabels.length);
    this.colData.forEach((label) => {
      colSizes[label] += 1;
    });
    this.colSizes = colSizes;

    const rowSizes = new Int32Array(this.rowLabels.length);
    this.rowData.forEach((label) => {
      rowSizes[label] += 1;
    });
    this.rowSizes = rowSizes;
  }
  process() {
    this.getColClusters();
    this.getRowClusters();
    this.getAggregationInfo();
  }
  // get labels() {
  //   if (this.hasData && !this._labels) {
  //     this._labels = [...new Set([...(this.colData), ...(this.rowData)])];
  //     this._labels.sort((a, b) => a - b);
  //   }
  //   return this._labels;
  // }

  // get uniqRowLabels() {
  //   if (this.hasData && !this._uniqRowLabels) {
  //     const uniqueLabels = new Set([...(this.rowData)]);
  //     const rowLabels = [];
  //     for (let i = 0; i < this.labels.length; i++) {
  //       if (uniqueLabels.has(this.labels[i])) {
  //         rowLabels.push(this.labels[i]);
  //       }
  //     }
  //     this._uniqRowLabels = rowLabels;
  //   }
  //   return this._uniqRowLabels;
  // }
  // get uniqColLabels() {
  //   if (this.hasData && !this._uniqColLabels) {
  //     const uniqueLabels = new Set([...(this.colData)]);
  //     const colLabels = [];
  //     for (let i = 0; i < this.labels.length; i++) {
  //       if (uniqueLabels.has(this.labels[i])) {
  //         colLabels.push(this.labels[i]);
  //       }
  //     }
  //     this._uniqColLabels = colLabels;
  //   }
  //   return this._uniqColLabels;
  // }
  get ids() {
    return this.rawDat ? this.rawData.ids : undefined;
  }

  get words() {
    return this.rawData ? this.rawData.words : undefined;
  }

  strengthFilter(mode = this.params.mode) {
    switch (mode) {
      case 'positive':
        return (strength) => (strength > 0 ? strength : 0);
      case 'negative':
        return (strength) => (strength < 0 ? Math.abs(strength) : 0);
      case 'abs':
        return Math.abs;
      case 'raw':
      default:
        return (strength) => (strength);
    }
  }

  getAggregationInfo() {
    if (this.rawData && !this.aggregationInfo) {
      const colLabels = this.colData;
      const rowLabels = this.rowData;
      const colLabelMap = label2Index(this.colLabels);
      const rowLabelMap = label2Index(this.rowLabels);
      const strengthFilter = this.strengthFilter();
      // const colClusters = this.colClusters;
      const rowCluster2colCluster = create2DArray(
        this.rowLabels.length, this.colLabels.length);
      const rowSingle2colCluster = create2DArray(
        rowLabels.length, this.colLabels.length);
      const rowCluster2colSingle = create2DArray(
        this.rowLabels.length, colLabels.length);
      const rowSingle2colSingle = create2DArray(
        rowLabels.length, colLabels.length);
      // calculate the correlation between clusters
      this.correlation.forEach((strengthList, r) => {
        strengthList.forEach((s, c) => {
          const strength = strengthFilter(s);
          const row = rowLabelMap[rowLabels[r]];
          const col = colLabelMap[colLabels[c]];
          rowCluster2colCluster[row][col] +=
            strength / (this.rowSizes[rowLabels[r]] * this.colSizes[colLabels[c]]);
          rowCluster2colSingle[row][c] += strength / this.rowSizes[rowLabels[r]];
          rowSingle2colCluster[r][col] += strength / this.colSizes[colLabels[c]];
          rowSingle2colSingle[r][c] += strength;
        });
      });

      this.aggregationInfo = {
        rowCluster2colCluster,
        rowSingle2colCluster,
        rowCluster2colSingle,
        rowSingle2colSingle,
      };
    }
    return this.aggregationInfo;
  }

  getRowClusters() {
    if (this.rawData && !this.rowClusters) {
      const rowClusters = Array.from({
        length: this.rowLabels.length,
      }, () => []);
      this.rawData.row.forEach((r, i) => {
        rowClusters[r].push(i);
      });
      // this._rowClusters = new Array(this.labels.length);
      // this.labels.forEach((l, i) => this._rowClusters[i] = rowClusters[l]);
      this.rowClusters = new Array(this.rowLabels.length);
      this.rowLabels.forEach((label, i) => {
        this.rowClusters[i] = rowClusters[label];
      });
    }
    return this.rowClusters;
  }
  getColClusters() {
    if (this.rawData && !this.colClusters) {
      const colClusters = Array.from({
        length: this.colLabels.length,
      }, () => []);
      this.colData.forEach((c, i) => {
        colClusters[c].push(i);
      });
      // this._colClusters = new Array(this.labels.length);
      // this.labels.forEach((l, i) => this._colClusters[i] = colClusters[l]);
      this.colClusters = new Array(this.colLabels.length);
      this.colLabels.forEach((label, i) => {
        this.colClusters[i] = colClusters[label];
      });
    }
    return this.colClusters;
  }

  static identifier({
    modelName,
    stateName,
    nCluster,
    params,
  }) {
    return `${modelName}${stateName}${nCluster}${params.layer + 1}`;
  }

}

export default {
  SentenceRecord,
  StateProcessor,
  CoClusterProcessor,
};

