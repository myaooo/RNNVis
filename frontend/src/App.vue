<template>
  <div id="app">
    <el-menu theme="dark" :default-active="index" class="header-menu" mode="horizontal" @select="selectMenu">
      <el-menu-item class="logo" index="logo">RNNVis</el-menu-item>
      <el-menu-item index="guide">Help</el-menu-item>
    </el-menu>
    <el-row :gutter="10">
      <el-col :span="6" class="col-bg" :gutter="15">
        <model-view class="modelView"></model-view>
        <br>
        <info-board v-if='!compare' :compare='false' :type='"word"' :id='"info-word"' :height='infoHeight*1.0'></info-board>
        <info-board v-if='!compare' :compare='false' :type='"state"' :id='"info-state"' :height='infoHeight*0.8'></info-board>
      </el-col>
      <el-col :span="18" class="col-bg" :gutter="15">
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
import { mapState } from 'vuex';
import { introJs } from 'intro.js';
import ModelView from 'components/ModelView';
import MainView from 'components/MainView';
import TextView from 'components/TextView';
import InfoView from 'components/InfoView';
import InfoBoard from 'components/InfoBoard';
// import { state } from './state';

function startGuidance() {
  const intro = introJs().addSteps([
    {
      intro: 'Flowing the tips to learn how to use RNNVis :)',
    },
    {
      element: document.querySelectorAll('.modelView')[0],
      intro: 'The control panel shows the configurations of your analysis of the model(s)',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.hyper-params')[0],
      intro: 'The items show the architecture and hyper parameters of the selected model.',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.visual-config')[0],
      intro: 'You can configurate the visualization of the model here. ' +
        'You can onfigurate the target hidden state, layer, and other layout options.',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.layout-config')[0],
      intro: 'You can configurate the style of the visualization here, including: ' +
        'the number of the clusters, the width of the link, the link to be filtered, ' +
        'and the value range of the heatmap.',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.compare-button')[0],
      intro: 'To compare two models side-by-side, click this button to add a second model.',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.main-view')[0],
      intro: 'This is the main view showing the co-cluster visualization.',
      position: 'left',
    },
    {
      element: document.querySelectorAll('.topGroup')[0],
      intro: 'Hidden units of a hidden state vector are coclustered with words in the vocabulary.',
      position: 'bottom',
    },
    {
      element: document.querySelectorAll('.chipGroup')[0],
      intro: 'Each hidden unit is a small squared memory cell. Hidden unit clusters are arranged as memory chips',
      position: 'right',
    },
    {
      element: document.querySelectorAll('.wordGroup')[0],
      intro: 'Word clusters are displayed as word clouds served as sementic indicator. ' +
        'Click a few words to see the model\'s response in the memory chips!',
      position: 'left',
    },
    {
      element: document.querySelectorAll('.linkGroup')[0],
      intro: 'bi-connections between hidden units clusters and word clusters represent their relations. ' +
        'Wider links denote stronger relations. Color denotes positive (red) and negative (blue) relations',
      position: 'bottom',
    },
    {
      intro: 'Enjoy the analysis!',
    },
  ]);
  intro.start();
}

export default {
  name: 'app',
  components: { ModelView, MainView, TextView, InfoView, InfoBoard },
  data() {
    return {
      height: 800,
      width: 1000,
      index: 'logo',
    };
  },
  computed: {
    mainHeight() {
      return this.compare ? this.height * 0.6 : this.height * 0.9;
    },
    infoHeight() {
      return this.height * 0.2;
    },
    ...mapState({
      compare: 'compare',
    }),
  },
  mounted() {
    this.height = window.innerHeight;
    this.width = window.innerWidth;
    window.addEventListener('resize', () => {
      this.height = window.innerHeight;
      this.width = window.innerWidth;
    });
  },
  methods: {
    selectMenu(index) {
      this.index = index;
      if (index === 'guide') {
        startGuidance();
      }
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
}

.logo {
  font-size: 18px;
}

</style>
