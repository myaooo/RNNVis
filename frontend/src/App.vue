<template>
  <div id="app">
    <!--<el-row :gutter="10" class="header-menu">-->
      <el-menu theme="dark" :default-active="index" class="header-menu" mode="horizontal" @select="selectMenu">
        <el-menu-item class="logo" index="logo">RNNVis</el-menu-item>
        <el-menu-item index="guide">Help</el-menu-item>
      </el-menu>
    <!--</el-row>-->
    <el-row :gutter="10">
      <el-col :span="6" class="col-bg" :gutter="15">
        <model-view class="modelView"></model-view>
        <br>
        <info-board v-if='!compare' :compare='false' :type='"word"' :id='"info-word"' :height='infoHeight*1.0'></info-board>
        <info-board v-if='!compare' :compare='false' :type='"state"' :id='"info-state"' :height='infoHeight*0.8'></info-board>
      </el-col>
      <el-col :span="18" class="col-bg" :gutter="15">
        <!--<router-view></router-view>-->
        <el-row>
          <main-view :height="mainHeight"> </main-view>
        </el-row>
        <el-row :gutter="10">
          <info-view v-if='compare' :height="infoHeight"> </info-view>
        </el-row>
      </el-col>

    </el-row>
  </div>
</template>

<script>
import ModelView from 'components/ModelView';
import MainView from 'components/MainView';
import TextView from 'components/TextView';
import InfoView from 'components/InfoView';
import InfoBoard from 'components/InfoBoard';
import { bus } from './event-bus';
import { introJs } from 'intro.js'

export default {
  name: 'app',
  components: { ModelView, MainView, TextView, InfoView, InfoBoard },
  data() {
    return {
      height: 800,
      width: 1000,
      shared: bus.state,
      index: "logo",
    };
  },
  computed: {
    mainHeight: function () {
      return this.compare ? this.height * 0.6 : this.height*0.9;
    },
    infoHeight: function () {
      return this.height * 0.2;
    },
    compare: function () {
      return this.shared.compare;
    }
  },
  mounted() {
    this.height = window.innerHeight;
    this.width = window.innerWidth;
    window.addEventListener("resize", () => {
      this.height = window.innerHeight;
      this.width = window.innerWidth;
    });
  },
  methods: {
    selectMenu(index) {
      this.index = index;
      if (index == "guide") {
        let intro = introJs().addSteps([
          {
            intro: "Flowing the tips to learn how to use RNNVis :)",
          },
          {
            element: document.querySelectorAll('.modelView')[0],
            intro: "Configurations on your analysis of the model(s)",
            position: "right",
          },
          {
            element: document.querySelectorAll('.modelView')[0],
            intro: "Configurations on your analysis of the model(s)",
            position: "right",
          },
          {
            element: document.querySelectorAll('.topGroup')[0],
            intro: "Hidden units of a hidden state vector are coclustered with words from vocabulary.",
            position: 'bottom',
          },
          {
            element: document.querySelectorAll('.chipGroup')[0],
            intro: "Different hidden unit clusters are arranged as memory chips",
            position: 'right',
          },
          {
            element: document.querySelectorAll('.wordGroup')[0],
            intro: "Word clusters are displayed as word clouds served as sementic indicator",
            position: 'left',
          },
          {
            element: document.querySelectorAll('.linkGroup')[0],
            intro: "bi-connections between hidden units clusters and word clusters represent their relations. Wider links denote stronger relations. Color denotes positive (red) and negative (blue) relations",
            position: 'bottom',
          }
        ]);
        intro.start();
      }
    }
  }
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
  background: #abc;
}
.bg-purple {
  background: white;
}
.bg-purple-light {
  background: #fff;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.col-bg {
  padding: 5px 0;
  background-color: white;
}
.row-bg {
  padding: 10px 0;
  background-color: white;
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
  font-size: 14px;
  color: #555;
  line-height: 18px;
  margin-top: 0px;
  padding-left: 10px;
  padding-top: 5px;
  padding-bottom: 5px;
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
