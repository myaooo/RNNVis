<template>
  <div class="project">
    <el-tabs v-model="selectedState">
    <el-tab-pane v-for="state in states" :label="state" :name="state">
      <svg :id="paneId(model, state)" :width="width" :height="height"> </svg>
    </el-tab-pane>

    </el-tabs>
  </div>
</template>
<style>

</style>
<script>
  import * as d3 from 'd3';
  import dataService from '../services/dataService.js';
  import {bus, SELECT_MODEL} from 'event-bus.js';

  export default {
    name: 'CircleView',
    data() {
      return {}
    },
    props: {},
    methods: {
    },

  }

</script>
