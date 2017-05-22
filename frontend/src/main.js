// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue';
import ElementUI from 'element-ui';
import 'theme/index.css';
import App from './App';

Vue.use(ElementUI);

/* eslint-disable no-new */
new Vue({
  el: '#app',
  // router,
  // template: '<App/>',
  // components: {App},
  render: h => h(App)
});
