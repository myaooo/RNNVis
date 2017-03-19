<template>
  <div id="app">
    <!--<el-row :gutter="10" class="header-menu">-->
      <el-menu theme="dark" :default-active="''" class="header-menu" mode="horizontal" @select="() => {console.log('Hi!');}">
        <el-menu-item class="logo" index="1">RNNVis</el-menu-item>
        <el-menu-item index="2">Info</el-menu-item>
      </el-menu>
    <!--</el-row>-->
    <el-row :gutter="10">
      <el-col :span="6" class="col-bg" :gutter="15">
        <model-view></model-view>
      </el-col>
      <el-col :span="18" class="col-bg" :gutter="15">
        <!--<router-view></router-view>-->
        <el-row>
          <main-view :height="height * 0.55"> </main-view>
        </el-row>
        <el-row>
          <el-col :span="12" class="col-bg" :gutter="15">
            <info-board :type="'state'" :id="'state-info'" :height="height * 0.2"> </info-board>
          </el-col>
          <el-col :span="12" class="col-bg" :gutter="15">
            <info-board :type="'word'" :id="'word-info'" :height="height * 0.2"> </info-board>
          </el-col>
        </el-row>
      </el-col>
      <!--<el-col :span='6' class="col-bg border" :gutter="15">-->

          <!--<text-view> </text-view>-->

      <!--</el-col>-->
    </el-row>
  </div>
</template>

<script>
import ModelView from 'components/ModelView';
import MainView from 'components/MainView';
import TextView from 'components/TextView';
import InfoBoard from 'components/InfoBoard';

export default {
  name: 'app',
  components: { ModelView, MainView, TextView, InfoBoard },
  computed: {
    height: function() {
      return document.documentElement.clientHeight;
    },
  },
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
  margin-top: 10px;
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
  padding-left: 10px;
  padding-top: 10px;
  padding-bottom: 10px;
  margin-bottom: -10px;
  text-align: left;
  background-color: rgba(128, 128, 128, 0.1);
}

.header-menu {
  height: 60px;
  /*margin-bottom: -10px;*/
}

.logo {
  font-size: 18px;
}

</style>
