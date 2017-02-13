import Vue from 'vue'
import Router from 'vue-router'
import TreeView from 'components/TreeView'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'TreeView',
      component: TreeView
    }
  ]
})
