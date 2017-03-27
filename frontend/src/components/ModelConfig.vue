<template>
  <div class="grid-content">
    <el-form label-position="left" label-width="30%">
      <!--Model Selector-->
      <el-form-item :label="compare ? 'Model 2' : 'Model'">
        <el-select v-model="selectedModel" placeholder="Select A Model" size="small" style="width: 80%;">
          <el-option v-for="(model, idx) in availableModels" :value="model" :label="model"></el-option>
        </el-select>
        <el-button v-if="compare" @click="toggle" icon="delete" size="small"></el-button>
      </el-form-item>

      <!--Model Configs as tags-->
      <div v-if="config" class="el-form-item">
        <el-tag v-for="key in Object.keys(config)" :type="colorType">{{key}}: {{config[key]}}</el-tag>
      </div>
      <hr v-if="config" class="local-hr">

      <!--Radio control for selecting state type-->
      <el-form-item label="Hidden State" v-if="states.length">
        <el-radio-group v-model="selectedState" size="small">
          <el-radio-button v-for="state in states" :label="state">{{stateName(state)}}</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="POS Tag" v-if="states.length">
        <el-switch v-model="posSwitch" on-text="" off-text="">
        </el-switch>
      </el-form-item>
      <el-form-item label="Layer" v-if="layerNum">
        <el-input-number size="small" v-model="selectedLayer" :max="layerNum-1" style="width: 100px; margin-top: 10px"></el-input-number>
      </el-form-item>

      <!--Sentence Editor-->
      <el-form-item label="Sentences" v-if="selectedModel">
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
      </el-form-item>
      <div class="sentence-container" v-if="sentences.length">
        <el-tag v-for="sentence in sentences" :closable="true" @close="closeSentence(sentence)" :type="colorType">
          <a>{{sentence}}</a>
        </el-tag>
      </div>
      <hr v-if="selectedState" class="local-hr">

      <!--Controls for the layout-->
      <el-form-item label="Cluster Num" v-if="selectedState">
        <el-slider v-model="layout.clusterNum" :min="2" :max="20" style="width: 80%" @change="layoutChange"></el-slider>
      </el-form-item>
      <el-form-item label="Stroke Width" v-if="selectedState">
        <el-slider v-model="layout.strokeControlStrength" :min="0" :max="0.02" :step="0.0001" style="width: 80%" @change="layoutChange"></el-slider>
      </el-form-item>
      <el-form-item label="Link Filter" v-if="selectedState">
        <el-slider v-model="layout.linkFilterThreshold" range show-stops :min="0" :max="1" :step="0.05" @change="layoutChange"></el-slider>
      </el-form-item>
    </el-form>
  </div>
</template>
<style>
  .el-form-item {
    margin-bottom: 5px;
    margin-top: -5px;
    font-size: 12px;
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
  import { bus, SELECT_MODEL, SELECT_STATE, SELECT_LAYER, CHANGE_LAYOUT, EVALUATE_SENTENCE, CLOSE_SENTENCE } from '../event-bus';

  export default {
    name: 'ModelConfig',
    data() {
      return {
        shared: bus.state,
        states: [],
        selectedState: null,
        selectedModel: null,
        selectedLayer: null,
        posSwitch: false,
        config: null,
        layout: {
           clusterNum: 10, 
           strokeControlStrength: 0.01, 
           linkFilterThreshold: [0.2, 1],
        },
        sentences: [],
        inputVisible: false,
        inputValue: '',
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
      availableModels: function () { // model list
        return this.shared.availableModels;
      },
      colorType: function() {
        return this.compare ? 'gray' : '';
      },
      layerNum: function() {
        if (this.config) return this.config.LayerNum;
        return 0;
      }
    },
    watch: {
      selectedModel: function (selectedModel) {
        if (!selectedModel) return;
        bus.loadModelConfig(selectedModel) // make sure the bus has got the config data
          .then(() => {
            const states = bus.availableStates(selectedModel);
            if (states) {
              this.states = states;
              this.selectedState = this.states[0]; // reset
              const config = bus.state.modelConfigs[selectedModel];
              this.config = {
                Cell: config.model.cell_type,
                LayerNum: config.model.cells.length,
                LayerSize: config.model.cells[0].num_units,
              };
              this.posSwitch = false;
              this.selectedLayer = this.config.LayerNum - 1;
              this.sentences = [];
              bus.$emit(SELECT_MODEL, this.selectedModel, this.compare);
              bus.$emit(CHANGE_LAYOUT, this.layout, this.compare);
            }
          }).catch((v) => console.log(v));
      },
      selectedState: function (newState) {
        if (newState === 'state' || newState === 'state_c' || newState === 'state_h') {
          bus.$emit(SELECT_STATE, this.selectedState, this.compare);
        }
      },
      selectedLayer: function (newLayer) {
        bus.$emit(SELECT_LAYER, newLayer, this.compare);
      },
      posSwitch: function(pos) {
        if (this.compare) this.shared.renderPos2 = pos;
        else this.shared.renderPos = pos;
      }
      // layout: function (newLayout) {
      //   console.log('layout changed.')
      //   bus.$emit(CHANGE_LAYOUT, newLayout, this.compare);
      // }
    },
    methods: {
      stateName(state) {
        switch (state) {
          case 'state_c': return 'c_state';
          case 'state_h': return 'h_state';
          case 'state': return 'h_state';
          default: return 'Unknown';
        }
      },
      layoutChange() {
        console.log("Layout changed")
        // copy to a new one to force change
        const layout = Object.assign({}, this.layout)
        bus.$emit(CHANGE_LAYOUT, layout, this.compare);
      },
      // strokeControlTypeChange() {
      //   switch(this.layout.strokeControlType) {
      //     case "Linear":
      //       this.strokeControlStrengthMin = 0;
      //       this.strokeControlStrengthMax = 0.02;
      //       this.strokeControlStrengthStep = 0.0001;
      //       break;
      //     case "Logarithm":
      //       this.strokeControlStrengthMin = 2;
      //       this.strokeControlStrengthMax = 10;
      //       this.strokeControlStrengthStep = 0.1;
      //       break;
      //     case "Exponential":
      //       this.strokeControlStrengthMin = 1;
      //       this.strokeControlStrengthMax = 1.002;
      //       this.strokeControlStrengthStep = 0.0001;
      //       break;
      //     default:
      //       console.log("The control type of " + controlType + " currently is not supported");
      //       return;
      //   }
      //   const layout = Object.assign({}, this.layout)
      //   bus.$emit(CHANGE_LAYOUT, layout, this.compare);
      // },
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
          bus.$emit(EVALUATE_SENTENCE, inputValue, this.compare);
          this.sentences.push(inputValue);
        }
        this.inputVisible = false;
        this.inputValue = '';
      },
    },
    mounted() {
      // this.activeSentences = this.sentences.map((d, i) => {
      //   return true;
      // });
    }
  };

</script>
