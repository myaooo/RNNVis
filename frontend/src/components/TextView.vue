<template>
  <div>
    <div v-for="sentence in sentences">
      <span v-for="word in sentence" v-bind:style="{backgroundColor: color(word.value)}"> {{ word.text }} </span>
    </div>
  </div>
</template>
<style>

</style>
<script>
  import dataService from '../services/dataService.js';

  let activeColorScheme = ["88, 126, 182", "201, 90, 95"];
  export default{
    name: 'TextView',
    data() {
      var texts = dataService.getTextData('1', '2');
      var sentences = Array.from(texts,
        function(words, i) {
          return Array.from( words,
            word => { return {text: word[0], value: word[1]}; }
          )
        }
      );
      // console.log(sentences);

      return {
        sentences
      };
    },
    methods: {
      color(value) {
        if (value < 0)
          return 'rgba(' + activeColorScheme[0] + ',' + (-value) + ')';
        return 'rgba(' + activeColorScheme[1] + ',' + value + ')';
      }
    }
  }
  function colorGrad(color1, color2, ratio) {
    return color1.map((c, i) => c * ratio + color2[i] * (1-ratio));
  }

</script>
