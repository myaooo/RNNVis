<template>
  <div id="app">
    <el-row :gutter="10">
      <el-col :span="6" class="col-bg">
        <model-view></model-view>
      </el-col>
      <el-col :span="18" class="col-bg">
        <router-view></router-view>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import ModelView from 'components/ModelView'

export default {
  name: 'app',
  components: { ModelView }
}
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
  margin-bottom: 20px;
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
  padding: 10px 0;
  background-color: #f9fafc;
}
.row-bg {
  padding: 10px 0;
  background-color: #f9fafc;
}
</style>
