<template>
  <div class="main-view">
    <h4 class="normal">Main</h4>
    <hr>
    <cluster-view v-if="selectedModel2" :width="width/2-15" :height="height" :compare='true'> </cluster-view>
    <cluster-view v-if="selectedModel" :width="selectedModel2 ? width/2-15 : width" :height="height" :compare='false'> </cluster-view>

  </div>

</template>
<style>
  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
<script>
  import { mapState } from 'vuex';
  import ClusterView from 'components/ClusterView';
  import { bus } from '../store';

  export default {
    name: "MainView",
    data() {
      return {
        activeTab: 'project',
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
        return;
      }
    },
    computed: {
      ...mapState({
        selectedModel: 'selectedModel',
        selectedModel2: 'selectedModel2',
      }),
    },
    components: {
      ClusterView,
    },
    mounted() {
      this.width = this.$el.clientWidth;
    }
  };
</script>
