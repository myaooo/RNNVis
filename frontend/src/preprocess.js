import dataService from './services/dataService';

class CoClusterProcessor {
  constructor(modelName, stateName, nCluster = 10, params = { top_k: 300, mode: 'raw' }) {
    this.rawData;
    this.modelName = modelName;
    this.stateName = stateName;
    this.nCluster = nCluster;
    this.params = params;
    // this.s
  }
  get correlation() {
    return this.hasData ? this.rawData.data : undefined;
  }
  get labels() {
    if (this.hasData) {
      this._labels = this.rowLabels.filter((v, i, arr) => { return arr.indexOf(v) === i; });
      this._labels.sort();
      return this._labels;
    }
    return this._labels;
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
      } else {
        throw response;
      }
    });
  }
  get hasData() {
    return Boolean(this.rawData);
  }
  get rowClusters() {
    if (this.hasData) {
      // delete this.rowClusters;
      const range = Array.from({ length: this.rowLabels.length }, (v, i) => i);
      return this._rowClusters = this.labels.map((label) => {
        return range.filter((i) => this.rowLabels[i] == label);
      });
    }
    return this._rowClusters;
  }
  get colClusters() {
    if (this.hasData) {
      // delete this.colClusters;
      const range = Array.from({ length: this.colLabels.length }, (v, i) => i);
      return this._colClusters = this.labels.map((label) => {
        return range.filter((i) => this.colLabels[i] == label);
      });
    }
    return this._colClusters;
  }
  static identifier(processor) {
    return `${processor.modelName}_${processor.stateName}_${processor.nCluster}`;
  }

}

class SentenceRecord{
  constructor(inputs) {
    this.inputs = inputs;
    this.tokens;
    this.records;
  }
  evaluate(modelName) {
    return dataService.getTextEvaluation(modelName, this.inputs, (response => {
      if(response.status === 200){
        const data = response.data;
        this.tokens = data.tokens;
        this.records = data.records;
      }
    }));
  }
  get states() {
    if (this.records) {
      this._states = Object.keys(this.records[0][0]);
    }
  }
  get layerNum() {
    return this.records[0][0][this.states[0]].length;
  }
  getRecords(stateName, layer = -1){
    if (this.records) {
      layer = layer === -1 ? this.layerNum - 1 : layer;
      return this.records.forEach((u) => {
        return u.forEach((v) => {
          return v[stateName][layer];
        });
      });
    }
    return undefined;
  }
}

export {
  CoClusterProcessor,
  SentenceRecord,
}
