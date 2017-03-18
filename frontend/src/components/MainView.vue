<template>
  <el-tabs v-model="activeTab" @tab-click="handleClick" type="border-card" height=800>
    <el-tab-pane label="Project" name="project">
      <project-view> </project-view>
    </el-tab-pane>
    <el-tab-pane label="Arc" name="arc">
      <arc-view> </arc-view>
    </el-tab-pane>
    <el-tab-pane label="Test" name="test">
      <test-view> </test-view>
    </el-tab-pane>
    <!--<el-tab-pane label="Role" name="third">Role</el-tab-pane>
    <el-tab-pane label="Task" name="fourth">Task</el-tab-pane>-->
  </el-tabs>
</template>
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
<style>
  .model-config {
    text-align: left;
    padding: 14px;
  }

  .config-tree {
    margin-top: 10px;
  }
</style>
