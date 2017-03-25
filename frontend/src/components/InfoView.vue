<template>
  <div>
    <el-col :span="12" class="col-bg" :gutter="15">
      <info-board :type="type1" :id="'info-board-1'" :height="height" :compare='compare'> </info-board>
    </el-col>
    <el-col :span="12" class="col-bg" :gutter="15">
      <info-board :type="type2" :id="'info-board-2'" :height="height" :compare='false'> </info-board>
    </el-col>
  </div>
</template>
<style scoped>
  .model-view {
    text-align: left;
    /*padding: 5px;*/
  }

  .content {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
<script>
  import { bus, SELECT_UNIT, SELECT_WORD } from '../event-bus.js';
  import InfoBoard from './InfoBoard';

  export default{
    name: 'InfoView',
    components: { InfoBoard },
    data() {
      return {
        shared: bus.state,
        type1: 'state',
        type2: 'word',
      };
    },
    props: {
      height: {
        type: Number,
        default: 300,
      }
    },
    computed: {
      compare: function() {
        return this.shared.compare;
      },
    },
    methods: {

    },
    mounted() {
      bus.$on(SELECT_UNIT, (unit, compare) => {
        if (!this.compare) return;
        if (compare) {
          this.type1 = 'state';
        } else {
          this.type2 = 'state';
        }
      });
      bus.$on(SELECT_WORD, (word, compare) => {
        if (!this.compare) return;
        if (compare) {
          this.type1 = 'word';
        } else {
          this.type2 = 'word';
        }
      });
    }
  }

</script>
