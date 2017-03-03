<template>
  <div>
    <h3>Models</h3>
    <div class="grid-content bg-purple-light">
      <el-select v-model="selected" placeholder="Select">
        <el-option v-for="model in models" :value="model.value" :label="model.text"></el-option>
      </el-select>

    </div>
  </div>
</template>
<style>

</style>
<script>
  import dataService from '../services/dataService.js'


  // let activeColorScheme = ["88, 126, 182", "201, 90, 95"];
  export default{
    name: 'ModelView',
    data() {
      return {
        selected: null,
        models: []
      };
    },
    mounted(){
      this.getModels();
    },
    methods: {
      getModels() {
        dataService.getModels(response => {
          if (response.status === 200) {
            const data = response.data;
            // console.log(data);
            var models = data.models.map( (d, i) => {
              return { text: d, value: i };
            });
            this.models = models
          }
        });
      }
    }
  }

</script>
