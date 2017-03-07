<template>
  <div id="app">
    <el-row :gutter="15">
      <el-col :span="4" class="col-bg" :gutter="15">
        <model-view></model-view>
      </el-col>
      <el-col :span="14" class="col-bg" :gutter="15">
        <!--<router-view></router-view>-->
        <main-view> </main-view>
      </el-col>
      <el-col :span='6' class="col-bg border" :gutter="15">
        <!--<el-col :span='2' class="col-bg"> </el-col>-->
        <!--<el-col :span='20' class="col-bg border">-->
          <text-view> </text-view>
        <!--</el-col>-->
        <!--<el-col :span='2' class="col-bg"> </el-col>-->
      </el-col>
    </el-row>
  </div>
</template>

<script>
import ModelView from 'components/ModelView';
import MainView from 'components/MainView';
import TextView from 'components/TextView';

export default {
  name: 'app',
  components: { ModelView, MainView, TextView },
};
</script>

<style>
#app {
  font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
  /*font-family: 'Avenir', Helvetica, Arial, sans-serif;*/
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  /*color: #2c3e50;*/
  margin-top: 60px;
}

.el-row {
  margin-bottom: 10px;
  &:last-child {
    margin-bottom: 0;
  }
}
.el-col {
  border-radius: 4px;
}
.bg-purple-dark {
  background: #99a9bf;
}
.bg-purple {
  background: #d3dce6;
}
.bg-purple-light {
  background: #e5e9f2;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.col-bg {
  padding: 5px 0;
  background-color: #f9fafc;
}
.row-bg {
  padding: 10px 0;
  background-color: #f9fafc;
}
.border {
  border-style: solid;
  border-width: 1px;
  border-color: #99A9BF;
}
.normal {
  font-weight: normal;
}
h4 {
  line-height: 20px;
  margin-top: 5px;
  margin-bottom: 5px;
  text-align: left;
}
</style>
