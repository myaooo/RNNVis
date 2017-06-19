<template>
  <div id="guidance">
    <h4 class="normal">{{ header }}</h4>
    <hr>
    <svg :id='svgId' :width='width' :height='height'> </svg>
    <svg :id='svgId+"2"' :width='width' :height='30'> </svg>
  </div>
</template>

<style>
#guidance {
    position: fixed;
    display: none;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0,0,0,0.5);
    z-index: 2;
    cursor: pointer;
}
</style>

<script>
export default {

}
</script>
