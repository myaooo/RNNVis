<template>
  <div class="text-view">
    <h4 class="normal">Text</h4>
    <hr>
    <div class="state-select" v-if="displayControl">
      <el-radio-group v-model="selectedState" size="small">
        <el-radio-button v-for="state in states" :label="state"></el-radio-button>
      </el-radio-group>
      <div class="input">
        <el-col :span="4" class="label">
          <span>Layer:</span>
        </el-col>
        <el-col :span="8">
          <el-input-number size="small" v-model="selectedLayer" :max="layerNum-1" style="width: 80%"></el-input-number>
        </el-col>
        <!--<el-col :span="2"> <span> </span></el-col>-->
        <el-col :span="4" class="label">
          <span>Dim:</span>
        </el-col>
        <el-col :span="8">
          <el-input-number size="small" v-model="selectedDim" :max="layerSize" style="width: 100%"></el-input-number>
        </el-col>
      </div>
    </div>
    <el-input
      type="textarea"
      :rows="2"
      placeholder="Please input"
      v-if="isInput"
      v-model="inputs"
      class="text-input">
    </el-input>
    <el-button @click="onSubmit"> Evaluate
    </el-button>
    <hr v-if="records">
    <div class="text-box" v-if="records">
      <div v-for="sentence in sentences">
        <span v-for="word in sentence" v-bind:style="{backgroundColor: color(word.value)}"> {{ word.text }} </span>
      </div>
    </div>
  </div>
</template>
<style scoped>
.text-view {
  padding: 5px;
  text-align: left;
}
.text-box {
  border-style: solid;
  border-width: 1px;
  border-color: lightgrey;
  padding: 6px;
  margin-top: 2px;
  margin-bottom: 10px;
}
.text-input {
  margin-top: 2px;
  margin-bottom: 10px;
}
.state-select {
  margin-bottom: 10px;
}
.label {
  margin: auto;
  font-size: 10pt;
  text-align: left;
  /*line-height: 20px;*/
  /*padding: 8px 0 8px;*/
}
.input {
  margin-top: 6px;
  margin-bottom: 8px;
}
</style>
<script>
  import dataService from '../services/dataService';
  import { bus, SELECT_MODEL } from '../event-bus';

  let activeColorScheme = ["88, 126, 182", "201, 90, 95"];
  export default {
    name: 'TextView',
    data() {
      // const texts = dataService.getTextData('1', '2');
      const texts = [
        [['i', 0.2], ['love', 0.4], ['you', 0.5], ['omg', 0.2], ['<eos>', 0.1]],
        [['i', 0.4], ['like', 0.2], ['you', 0.3], ['<eos>', 0.1], ['omg', 0.2]],
      ];
      // const sentences = Array.from(texts,
      //   function(words, i) {
      //     return Array.from( words,
      //       word => { return {text: word[0], value: word[1]}; }
      //     )
      //   }
      // );
      // console.log(sentences);
      return {
        // sentences: sentences,
        inputs: null,
        shared: bus.state,
        isInput: true,
        states: [],
        selectedState: null,
        selectedLayer: -1,
        selectedDim: null,
        cache: {},
        oldInputs: null,
        tokens: null,
        records: null,
        layerNum: 0,
        // layerSize: 0,
      };
    },
    computed: {
      displayControl: function() {
        return this.states.length !== 0;
      },
      // hasRecords: function() {
      //   return this.records;
      // },
      layerSize: function() {
        const selectedModel = this.shared.selectedModel;
        return bus.layerSize(selectedModel, this.selectedLayer);
      },
      sentences: function() {
        if(this.records){
          return this.tokens.map((sentence, i) => {
            return sentence.map((token, j) => {
              return {text: token, value: this.records[i][j][this.selectedDim]};
            })
          });
        }
        return [];
      }
    },
    watch: {
      // inputs: function(newValue, oldValue){
      //   this.oldInputs = oldValue;
      // }
    },
    methods: {
      color(value) {
        if (value < 0)
          return 'rgba(' + activeColorScheme[0] + ',' + (-value) + ')';
        return 'rgba(' + activeColorScheme[1] + ',' + value + ')';
      },
      onSubmit() {
        if (this.selectedState === null) return false;
        const selectedModel = this.shared.selectedModel;
        if(this.inputs === this.oldInputs) return true;
        // const splitted = this.inputs.split('.')
        dataService.getTextEvaluation(selectedModel, this.selectedState, this.selectedLayer, this.inputs,
          (response) => {
            const data = response.data;
            this.tokens = data.tokens;
            this.records = data.records;
            this.oldInputs = this.inputs;
          });
      }
    },
    mounted() {
      bus.$on(SELECT_MODEL, (modelName) => {
        const states = bus.availableStates(modelName);
        if (states) {
          this.states = states;
          this.layerNum = bus.layerNum(modelName);
          this.selectedLayer = this.layerNum - 1;
          this.selectedState = this.states[0];
        }
        else {
          this.states = [];
        }
      })
    }
  }
  function colorGrad(color1, color2, ratio) {
    return color1.map((c, i) => c * ratio + color2[i] * (1-ratio));
  }

</script>
