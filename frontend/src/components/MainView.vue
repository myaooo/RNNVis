<template>
  <el-tabs v-model="activeTab" @tab-click="handleClick" type="border-card">
    <el-tab-pane label="Project" name="project">
      <project-view> </project-view>
    </el-tab-pane>
    <el-tab-pane label="Tree" name="tree">Config</el-tab-pane>
    <el-tab-pane label="Arc" name="arc">
      <arc-view> </arc-view>
    </el-tab-pane>
    <!--<el-tab-pane label="Role" name="third">Role</el-tab-pane>
    <el-tab-pane label="Task" name="fourth">Task</el-tab-pane>-->
  </el-tabs>
</template>
<script>
  // import ModelView from 'components/ModelView';
  import ProjectView from 'components/ProjectView';
  import TreeView from 'components/TreeView';
  import ArcView from 'components/ArcView';

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
      TreeView,
      ArcView,
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
