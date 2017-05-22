<template>
  <div>
    <h4 class="normal">Main</h4>
    <hr>
    <cluster-view v-if="compare" :width="width/2-15" :height="height" :compare='true'> </cluster-view>
    <cluster-view :width="compare ? width/2-15 : width" :height="height" :compare='false'> </cluster-view>

    <!--<test-view> </test-view>-->
      <!--<project-view> </project-view>-->

  </div>

  <!--<el-tabs v-model="activeTab" @tab-click="handleClick" type="border-card" height=800>

    <el-tab-pane label="Cluster" name="cluster">
      <cluster-view class="content" :height="height"> </cluster-view>
    </el-tab-pane>
    <el-tab-pane label="Test" name="test">
      <test-view> </test-view>
    </el-tab-pane>
  </el-tabs>-->

  <!--<el-tab-pane label="Project" name="project">
      <project-view> </project-view>
    </el-tab-pane>
    <el-tab-pane label="Arc" name="arc">
      <arc-view> </arc-view>
    </el-tab-pane>-->

</template>
<style>
  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
<script>
  // import ModelView from 'components/ModelView';
  // import ProjectView from 'components/ProjectView';
  // import TestView from 'components/TestView';
  // import ArcView from 'components/ArcView';
  import ClusterView from 'components/ClusterView';
  import { bus } from '../event-bus';

  export default {
    name: "MainView",
    data() {
      return {
        activeTab: 'project',
        shared: bus.state,
        width: 800,
      };
    },
    props: {
      height: {
        type: Number,
        default: 700,
      }
    },
    methods: {
      handleClick(tab, event) {
        // console.log(tab, event);
        return;
      }
    },
    computed: {
      // width: function () {
      //   if (this.$el) {
      //     console.log(this.$el.clientWidth);
      //     return this.$el.clientWidth;
      //   }
      //   return 800;
      // },
      compare: function () {
        return this.shared.compare;
      }
    },
    components: {
      // ProjectView,
      // TestView,
      // ArcView,
      ClusterView,
    },
    mounted() {
      this.width = this.$el.clientWidth;
    }
  };
</script>
