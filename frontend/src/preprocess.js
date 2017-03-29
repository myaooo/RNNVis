import dataService from './services/dataService';

export class CoClusterProcessor {
  constructor(modelName, stateName, nCluster = 10, params = { top_k: 300, mode: 'raw', layer: -1 }, sortBy = 'col') {
    this.rawData;
    this._rowClusters;
    this._colClusters;
    this._colSizes;
    this._rolSizes;
    this._aggregation_info = null;
    this.modelName = modelName;
    this.stateName = stateName;
    this.nCluster = nCluster;
    this.params = params;
    this.sortBy = sortBy;
  }
  get correlation() {
    return this.hasData ? this.rawData.data : undefined;
  }
  get labels() {
    if (this.hasData && !this._labels) {
      this._labels = [...new Set([...(this.colLabels), ...(this.rowLabels)])];
      if (this.sortBy === 'col')
        this._labels.sort((a, b) => this.colSizes[a] - this.colSizes[b]);
      else if (this.sortBy === 'row')
        this._labels.sort((a, b) => this.rowSizes[a] - this.rowSizes[b]);
      // console.log(this._labels);
    }
    return this._labels;
  }
  get colSizes() {
    if (this.hasData && !this._colSizes) {
      const colSizes = new Int32Array(this.labels.length);
      this.colLabels.forEach((label, i) => { colSizes[label] += 1; });
      this._colSizes = colSizes;
    }
    return this._colSizes;
  }
  get rowSizes() {
    if (this.hasData && !this._rowSizes) {
      const rowSizes = new Int32Array(this.labels.length);
      this.rowLabels.forEach((label, i) => { rowSizes[label] += 1; });
      this._rowSizes = rowSizes;
    }
    return this._rowSizes;
  }
  get rowLabels() {
    return this.hasData ? this.rawData.row : undefined;
  }
  get colLabels() {
    return this.hasData ? this.rawData.col : undefined;
  }
  get ids() {
    return this.hasData ? this.rawData.ids : undefined;
  }

  get words() {
    return this.hasData ? this.rawData.words : undefined;
  }

  load() {
    return dataService.getCoCluster(this.modelName, this.stateName, this.nCluster, this.params, (response) => {
      if (response.status === 200) {
        this.rawData = response.data;
        console.log(this);
      } else {
        throw response;
      }
    });
  }
  get hasData() {
    return Boolean(this.rawData);
  }

  strength_filter(strength, mode = this.params.mode) {
    switch (mode) {
      case 'positive':
        return strength > 0 ? strength : 0;
        break;
      case 'negative':
        return strength < 0 ? Math.abs(strength) : 0;
        break;
      case 'abs':
        return Math.abs(strength);
        break;
      case 'raw':
        return strength;
        break;
    }
  }

  Create2DArray(rowNum, colNum) {
    return Array.from({ length: rowNum }, (v, i) => {
      return new Float32Array(colNum); //Array.from({ length: colNum }, (v, i) => 0);
    });
  }

  // sortData() {
  //   if (this.hasData) {

  //   }
  // }
  get aggregation_info() {
    if (this.hasData && !this._aggregation_info) {
      // const rowClusters = this.rowClusters;
      // const colClusters = this.colClusters;
      const row_cluster_2_col_cluster = this.Create2DArray(this.nCluster, this.nCluster);
      const row_single_2_col_cluster = this.Create2DArray(this.rawData.row.length, this.nCluster);
      const row_cluster_2_col_single = this.Create2DArray(this.nCluster, this.rawData.col.length);
      const row_single_2_col_single = this.Create2DArray(this.rawData.row.length, this.rawData.col.length);
      const cluster = [];
      // calculate the correlation between clusters
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_cluster_2_col_cluster[this.rawData.row[r_index]][this.rawData.col[c_index]] += strength / (this.rowSizes[this.rowLabels[r_index]] * this.colSizes[this.colLabels[c_index]]);
          row_cluster_2_col_single[this.rawData.row[r_index]][c_index] += strength / this.rowSizes[this.rowLabels[r_index]];
          row_single_2_col_cluster[r_index][this.rawData.col[c_index]] += strength / this.colSizes[this.colLabels[c_index]];
          row_single_2_col_single[r_index][c_index] += strength;
        });
      });
      
      this._aggregation_info = {
        row_cluster_2_col_cluster: row_cluster_2_col_cluster,
        row_single_2_col_cluster: row_single_2_col_cluster,
        row_cluster_2_col_single: row_cluster_2_col_single,
        row_single_2_col_single: row_single_2_col_single
      };
    }
    return this._aggregation_info;
  }

  get cluster2cluster() {
    if (this.hasData && !this._cluster2cluster) {
      const row_cluster_2_col_cluster = this.Create2DArray(this.nCluster, this.nCluster);
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_cluster_2_col_cluster[this.rawData.row[r_index]][this.rawData.col[c_index]] += strength;
        });
      });
      this._cluster2cluster = row_cluster_2_col_cluster;
    }
    return this._cluster2cluster;
  }

  get single2cluster() {
    if (this.hasData && !this._single2cluster) {
      const row_single_2_col_cluster = this.Create2DArray(this.rawData.row.length, this.nCluster);
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_single_2_col_cluster[r_index][this.rawData.col[c_index]] += strength;
        });
      });
      this._single2cluster = row_single_2_col_cluster;
    }
    return this._cluster2cluster;
  }

  get cluster2single() {
    if (this.hasData && !this._cluster2single) {
      const row_cluster_2_col_single = this.Create2DArray(this.nCluster, this.rawData.col.length);
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_single_2_col_single[r_index][c_index] += strength;
        });
      });
      this._cluster2single = row_cluster_2_col_single;
    }
    return this._cluster2single;
  }

  get single2single() {
    if (this.hasData && !this._single2single) {
      const row_single_2_col_single = this.Create2DArray(this.rawData.row.length, this.rawData.col.length);
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_single_2_col_cluster[r_index][this.rawData.col[c_index]] += strength;
        });
      });
      this._single2cluster = row_single_2_col_cluster;
    }
    return this._cluster2cluster;
  }

  get rowClusters() {
    if (this.hasData && !this._rowClusters) {
      const rowClusters = Array.from({ length: this.labels.length }, (v, i) => []);
      this.rawData.row.forEach((r, i) => {
        // if (rowClusters[r] === undefined) {
        //   rowClusters[r] = [];
        // }
        rowClusters[r].push(i);
      });
      this._rowClusters = new Array(this.labels.length);
      this.labels.forEach((l, i) => this._rowClusters[i] = rowClusters[l]);
    }
    return this._rowClusters;
  }
  get colClusters() {
    if (this.hasData && !this._colClusters) {
      const colClusters = Array.from({ length: this.labels.length }, (v, i) => []);
      this.rawData.col.forEach((c, i) => {
        if (colClusters[c] === undefined) {
          colClusters[c] = [];
        }
        colClusters[c].push(i);
      });
      this._colClusters = new Array(this.labels.length);
      this.labels.forEach((l, i) => this._colClusters[i] = colClusters[l]);
    }
    return this._colClusters;
  }

  static identifier(processor) {
    return `${processor.modelName}${processor.stateName}${processor.nCluster}${processor.params.layer+1}`;
  }

}

export class SentenceRecord {
  constructor(inputs, modelName) {
    this.inputs = inputs;
    this.tokens;
    this.records;
    this.modelName = modelName;
  }
  evaluate(modelName = this.modelName) {
    return dataService.getTextEvaluation(modelName, this.inputs, (response => {
      if (response.status === 200) {
        const data = response.data;
        this.tokens = data.tokens[0]; // assume one sentence
        this.records = data.records[0];
      } else {
        throw response;
      }
    }));
  }
  get states() {
    if (this.records && !this._states) {
      this._states = Object.keys(this.records[0]);
    }
    return this._states;
  }
  get layerNum() {
    if (this.records && !this._layerNum) {
      this._layerNum = this.records[0][this.states[0]].length;
    }
    return this._layerNum;
  }
  getRecords(stateName, layer = -1) {
    if (this.records) {
      layer = layer === -1 ? this.layerNum - 1 : layer;
      console.log(layer);
      return this.records.map((word) => {
        return word[stateName][layer];
      });
    }
    return undefined;
  }
}

export class StateStatistics {
  constructor(modelName, stateName, layer = -1, top_k = 600) {
    this.modelName = modelName;
    this.stateName = stateName;
    this.layer = layer;
    this.top_k = top_k;
    this.data;
  }
  load() {
    if (this.data) return Promise.resolve("Already loaded");
    return dataService.getStateStatistics(this.modelName, this.stateName, this.layer, this.top_k, (response) => {
      this.data = response.data;
    });
  }
  get stateNum() {
    return this.data ? this.data.mean[0].length : undefined;
  }
  get statesData() { // calculate statistics for each state unit
    if (this.data && !this._statesData) {
      this._statesData = this.data.mean[0].map((_, j) => {
        const data = {
          words: this.data.words,
          // freqs: this.data.freqs,
          mean: this.data.mean.map((m) => m[j]),
          low1: this.data.low1.map((m) => m[j]),
          low2: this.data.low2.map((m) => m[j]),
          high1: this.data.high1.map((m) => m[j]),
          high2: this.data.high2.map((m) => m[j]),
          rank: this.data.sort_idx.map((indices) => {
            return indices.findIndex((idx) => (idx === j));
          }),
        };
        return data;
      });
    }
    return this._statesData;
  }
  get word2Id() {
    if (this.data && !this._word2Id) {
      const word2Id = {}
      this.data.words.forEach((word, i) => {
        word2Id[word] = i;
      })
      this._word2Id = word2Id;
    }
    return this._word2Id;
  }
  get wordsData() {
    if (this.data && !this._wordsData) {
      this._wordsData = this.data.mean.map((mean, i) => {
        const data = {
          mean: mean,
          range1: this.data.low1[i].map((low, j) => [low, this.data.high1[i][j]]),
          range2: this.data.low2[i].map((low, j) => [low, this.data.high2[i][j]]),
          word: this.data.words[i],
          sort_idx: this.data.sort_idx[i],
        };
        return data;
      });
    }
    return this._wordsData;
  }
  statOfWord(word) {
    if (this.data) {
      const id = this.word2Id[word];
      return this.wordsData[id];
    }
    return undefined;
  }
}

function memorize(fn) {
  var cache = {};
  return function () {
    var key = arguments.length + Array.prototype.join.call(arguments, ",");
    if (key in cache) {
      return cache[key];
    } else {
      return cache[key] = f.apply(this, arguments);
    }
  }
}

export default {
  CoClusterProcessor,
  SentenceRecord,
  StateStatistics,
  memorize,
};
