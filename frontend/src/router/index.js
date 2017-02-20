import Vue from 'vue'
import Router from 'vue-router'
import TreeView from 'components/TreeView'
import ProjectView from 'components/ProjectView'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'TreeView',
      component: TreeView
    },
    {
      path: '/project',
      name: 'ProjectView',
      component: ProjectView
    }
  ]
})
