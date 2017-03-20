import dataService from './services/dataService';

export class CoClusterProcessor {
  constructor(modelName, stateName, nCluster = 10, params = { top_k: 300, mode: 'raw' }) {
    this.rawData;
    this._rowClusters;
    this._colClusters;
    this._aggregation_info = null;
    this.modelName = modelName;
    this.stateName = stateName;
    this.nCluster = nCluster;
    this.params = params;
  }
  get correlation() {
    return this.hasData ? this.rawData.data : undefined;
  }
  get labels() {
    if (this.hasData) {
      this._labels = [...new Set(this.rowLabels)];
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

  strength_filter(strength, mode=this.params.mode) {
    let strength_item = 0;
    switch(mode) {
      case 'positive':
        strength_item = strength > 0 ? strength : 0;
        break;
      case 'negative':
        strength_item = strength < 0 ? Math.abs(strength) : 0;
        break;
      case 'abs':
        strength_item = Math.abs(strength);
        break;
      case 'raw':
        strength_item = strength;
        break;
    }
    return strength_item;
  }

  Create2DArray(rowNum, colNum) {
    return Array.from({ length: rowNum }, (v, i) => {
      return Array.from({ length: colNum }, (v, i) => 0);
    });
  }
  get aggregation_info() {
    if (this.hasData) {
      if (!this._aggregation_info) {
        return this._aggregation_info;
      }
      let rowClusters = this.rowClusters;
      let colClusters = this.colClusters;
      let row_cluster_2_col_cluster = this.Create2DArray(this.nCluster, this.nCluster);
      let row_single_2_col_cluster = this.Create2DArray(this.rawData.row.length, this.nCluster);
      let row_cluster_2_col_single = this.Create2DArray(this.nCluster, this.rawData.col.length);
      let row_single_2_col_single = this.Create2DArray(this.rawData.row.length, this.rawData.col.length);
      let cluster = [];
      // calculate the correlation between clusters
      this.correlation.forEach((strength_list, r_index) => {
        strength_list.forEach((s, c_index) => {
          let strength = this.strength_filter(s);
          row_cluster_2_col_cluster[this.rawData.row[r_index]][this.rawData.col[c_index]] += strength;
          row_cluster_2_col_single[this.rawData.row[r_index]][c_index] += strength;
          row_single_2_col_cluster[r_index][this.rawData.col[c_index]] += strength;
          row_single_2_col_single[r_index][c_index] += strength;
        });
      });

      return {row_cluster_2_col_cluster: row_cluster_2_col_cluster,
              row_single_2_col_cluster: row_single_2_col_cluster,
              row_cluster_2_col_single: row_cluster_2_col_single,
              row_single_2_col_single: row_single_2_col_single};
    }
    return null;
  }

  get rowClusters() {
    if (this.hasData) {
      this._rowClusters = [];
      this.rawData.row.forEach((r, i) => {
        if (this._rowClusters[r] === undefined) {
          this._rowClusters[r] = [];
        }
        this._rowClusters[r].push(i);
      });
      return this._rowClusters;
    }
    return this._rowClusters;
  }
  get colClusters() {
    if (this.hasData) {
      this._colClusters = [];
      this.rawData.col.forEach((c, i) => {
        if (this._colClusters[c] === undefined) {
          this._colClusters[c] = [];
        }
        this._colClusters[c].push(i);
      });
      return this._colClusters;
    }
    return this._colClusters;
  }

  static identifier(processor) {
    return `${processor.modelName}_${processor.stateName}_${processor.nCluster}`;
  }

}

export class SentenceRecord{
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

export default {
  CoClusterProcessor,
  SentenceRecord,
};
