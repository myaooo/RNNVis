<template>
  <div class="grid-content">
    <el-form label-position="left" label-width="30%">
      <!--Model Selector-->
      <el-form-item :label="compare ? 'Model 2' : 'Model'">
        <el-select v-model="selectedModel" placeholder="Select A Model" size="small" style="width: 80%;">
          <el-option v-for="(model, idx) in availableModels" :key="model" :value="model" :label="model"></el-option>
        </el-select>
        <el-button v-if="compare" @click="toggle" icon="delete" size="small"></el-button>
      </el-form-item>

      <!--Model Configs as tags-->
      <div v-if="config" class="el-form-item hyper-params">
        <el-tag v-for="key in Object.keys(config)" :key="key" :type="colorType">{{key}}: {{config[key]}}</el-tag>
      </div>
      <hr v-if="config" class="local-hr">

      <!--Radio control for selecting state type-->
      <div class="visual-config">
      <el-form-item label="Hidden State" v-if="states.length">
        <el-radio-group v-model="selectedState" size="small">
          <el-radio-button v-for="state in states" :key="state" :label="state">{{stateName(state)}}</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="Layer" v-if="layerNum">
        <el-input-number size="small" v-model="selectedLayer" :max="layerNum-1" style="width: 100px; margin-top: 5px"></el-input-number>
      </el-form-item>
      <el-form-item label="POS Tag" v-if="states.length">
        <el-switch v-model="posSwitch" on-text="" off-text="">
        </el-switch>
        <span class="align" v-if="style" style='margin-left: 30px'>Align</span>
        <el-switch v-model="mode" on-text="" off-text="" @change="updateStyle">
        </el-switch>
      </el-form-item>

      <!--Sentence Editor-->
      <el-form-item label="Sentences" v-if="selectedModel">
        <el-tooltip placement="top" :open-delay="500">
        <div slot="content">click to input a sentence<br/>for the model to evaluate on</div>
        <el-input
          class="input-new-tag"
          v-if="inputVisible"
          v-model="inputValue"
          ref="saveTagInput"
          size="small"
          @keyup.enter.native="handleInputConfirm"
          @blur="handleInputConfirm"
        >
        </el-input>
        <el-button v-else class="button-new-tag" size="small" @click="showInput">+ New Sentence</el-button>
        </el-tooltip>
      </el-form-item>
      </div>
      <div class="sentence-container" v-if="sentences.length">
        <el-tag v-for="sentence in sentences" :key="sentence" :closable="true" @close="closeSentence(sentence)" :type="colorType">
          <a>{{sentence}}</a>
        </el-tag>
      </div>

      <hr v-if="selectedState" class="local-hr">
      <div class="style-config">
      <!--Controls for the style-->
      <el-form-item label="Cluster Num" v-if="selectedState">
        <el-tooltip placement="top" :open-delay="500">
        <div slot="content">slide to adjust cluster number</div>
        <el-slider v-model="nCluster" :min="2" :max="20" @change="updateCoClusterData" style="width: 80%"></el-slider>
        </el-tooltip>
      </el-form-item>

      <el-form-item label="Stroke Width" v-if="style" style="margin-top: -7px; padding-bottom: -10px">
        <el-slider v-model="style.strokeControlStrength" :min="0" :max="20" :step="1" style="width: 80%" @change="updateStyle"></el-slider>
      </el-form-item>

      <el-form-item label="Link Filter" v-if="style" style="margin-top: -7px">
        <el-slider v-model="style.linkFilterThreshold" range :min="0" :max="1" :step="0.0001" @change="updateStyle" style="width: 80%"></el-slider>
      </el-form-item>

      <el-form-item label="State Clip" v-if="style" style="margin-top: -7px">
        <el-slider v-model="style.stateClip" :min="1" :max="5" :step="1" @change="updateStyle" style="width: 80%"></el-slider>
      </el-form-item>
      </div>
      <!--Color Picker for temporal use-->
      <!--<el-form-item label="colorPicker" v-if="selectedState">
        <el-color-picker label="positive" v-model="color[1]" @change="colorChange"></el-color-picker>
        <el-color-picker label="negative" v-model="color[0]" @change="colorChange"></el-color-picker>
      </el-form-item>-->
    </el-form>
  </div>
</template>
<style>
  .el-form-item {
    margin-bottom: 5px;
    margin-top: -5px;
    font-size: 12px;
  }

  .align {
    color: rgb(72, 87, 106);
  }

  .el-form-item__content{
    line-height: 30px !important;
  }

  label {
    font-size: 12px !important;
  }

  .el-tag {
    margin-right: 5px;
    display: inline-block;
    /*width: 90%;*/
    white-space: normal;
    height: auto;
    line-height: 18px;
  }

  .local-hr {
    width: 95%;
    font-size: 1px;
    color: gray;
    opacity: 0.5;
    line-height: 1px;
    margin-top: 8px;
    margin-bottom: 4px;
  }

  .sentence-container {
    width: 90%;
    margin-left: 10px;
  }
</style>
<script>
  import {
    mapActions,
    mapMutations,
    mapState
  } from 'vuex';
  import {
    SELECT_MODEL,
    SELECT_STATE,
    SELECT_LAYER,
    UPDATE_STYLE,
    EVALUATE_SENTENCE,
    CLOSE_SENTENCE,
    SELECT_COLOR,
    GET_COCLUSTER_DATA,
    RENDER_GRAPH,
  } from '../store';

  const stateDict = {
    'state_c': 'cell state',
    'state_h': 'hidden state',
    'state': 'hidden state',
    'state_h_en': 'encoder state',
    'state_h_de': 'decoder state',
  };

  export default {
    name: 'ModelConfig',
    data() {
      return {
        selectedModel: null,
        selectedState: null,
        selectedLayer: null,
        posSwitch: false,
        config: null,
        style: null,
        model: null,
        nCluster: 10,
        sentences: [],
        inputVisible: false,
        inputValue: '',
        mode: true,
        color: ['#09adff', '#ff5b09'],
        loading: false,
      };
    },
    props: {
      toggle: {
        type: Function,
        default: () => { return; },
      },
      compare: {
        type: Boolean,
        default: false,
      },
    },
    computed: {
      colorType: function() {
        return this.compare ? 'gray' : '';
      },
      // maxWidth: function() {
      //   // if (this.selectedModel.substring(0, 4) === 'YELP' || this.selectedModel.substring(0, 4) === 'IMDB') return 100;
      //   return 20;
      // },
      states: function() {
        return this.model ? this.model.stateList : [];
      },
      layerNum: function() {
        return this.model ? this.model.layerNum : 0;
      },
      modelName: function() {
        return this.model ? this.model.name : undefined;
      },
      ...mapState({
        availableModels: 'modelList',
        isLoading: 'isLoading',
      }),
    },
    watch: {
      selectedModel: function (selectedModel) {
        if (!selectedModel) return;
        this.selectModel({ modelName: selectedModel, })
          .then(model => {
            this.model = model;
            this.style = JSON.parse(JSON.stringify(model.style));
            this.selectedState = model.stateList[0];
            this.config = {
              Cell: model.cellType,
              LayerNum: model.layerNum,
              LayerSize: model.layerSize(-1),
            };
            this.selectedLayer = model.layerNum - 1;
            this.sentences = [];
            this.updateCoClusterData();
          });
      },
      mode: 'updateStyle',
      posSwitch: 'updateStyle',
      selectedState: 'updateCoClusterData',
      selectedLayer: 'updateCoClusterData',
      // nCluster: 'updateCoClusterData',
    },
    methods: {
      stateName(state) {
        if (state in stateDict) {
          return stateDict[state];
        }
        return 'Unkown';
      },
      closeSentence(sentence) {
        const idx = this.sentences.indexOf(sentence);
        if (idx !== -1){
          bus.$emit(CLOSE_SENTENCE, sentence, this.compare);
          this.sentences.splice(idx, 1);
        }
      },
      showInput() {
        this.inputVisible = true;
        this.$nextTick(_ => {
          this.$refs.saveTagInput.$refs.input.focus();
        });
      },
      handleInputConfirm() {
        let inputValue = this.inputValue;
        if (inputValue) {
          this.updateStyle({
            modelName: this.selectedModel.name
          })
          bus.$emit(EVALUATE_SENTENCE, inputValue, this.compare);
          this.sentences.push(inputValue);
        }
        this.inputVisible = false;
        this.inputValue = '';
      },
      updateCoClusterData() {
        if (this.isLoading) return Promise.resolve();
        return this.getCoCluster({
          modelName: this.selectedModel,
          stateName: this.selectedState,
          layer: this.selectedLayer,
          nCluster: this.nCluster,
          params: {topK: 500, mode: 'raw'},
        }).then(data => {
          if (data) {
            this.renderGraph({
              modelName : this.selectedModel,
            });
          }
          return data;
        });
      },
      updateStyle() {
        this.style.mode = this.mode ? 'width' : 'height';
        this.style.renderPos = this.renderPOS;
        this.changeStyle({
          modelName: this.model.name,
          style: this.style,
        });
      },
      ...mapActions({
        selectModel: SELECT_MODEL,
        getCoCluster: GET_COCLUSTER_DATA,
      }),
      ...mapMutations({
        selectState: SELECT_STATE,
        selectLayer: SELECT_LAYER,
        changeStyle: UPDATE_STYLE,
        renderGraph: RENDER_GRAPH,
      })
    },
  };

</script>
