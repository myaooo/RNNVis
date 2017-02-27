import Vue from 'vue'
import Router from 'vue-router'
import TreeView from 'components/TreeView'
import ProjectView from 'components/ProjectView'
import TextView from 'components/TextView'

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
    },
    {
      path: '/text',
      name: 'TextView',
      component: TextView
    }
  ]
})
