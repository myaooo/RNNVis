<template>
  <div>
    <h4 class="normal">Interpreter</h4>
    <hr>
    <cluster-view class="content" :height="height"> </cluster-view>
  </div>

  <!--<el-tabs v-model="activeTab" @tab-click="handleClick" type="border-card" height=800>
    <el-tab-pane label="Project" name="project">
      <project-view> </project-view>
    </el-tab-pane>
    <el-tab-pane label="Arc" name="arc">
      <arc-view> </arc-view>
    </el-tab-pane>
    <el-tab-pane label="Cluster" name="cluster">
      <cluster-view> </cluster-view>
    </el-tab-pane>
    <el-tab-pane label="Test" name="test">
      <test-view> </test-view>
    </el-tab-pane>
  </el-tabs>-->

</template>
<style>
  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
<script>
  // import ModelView from 'components/ModelView';
  import ProjectView from 'components/ProjectView';
  import TestView from 'components/TestView';
  import ArcView from 'components/ArcView';
  import ClusterView from 'components/ClusterView';

  export default {
    name: "MainView",
    data() {
      return {
        activeTab: 'project'
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
    components: {
      ProjectView,
      TestView,
      ArcView,
      ClusterView,
    }
  };
</script>
