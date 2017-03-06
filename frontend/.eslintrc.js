// http://eslint.org/docs/user-guide/configuring

module.exports = {
  root: true,
  parser: 'babel-eslint',
  parserOptions: {
    ecmaVersion: 6,
    sourceType: 'module'
  },
  env: {
    browser: true,
  },
  // https://github.com/feross/standard/blob/master/RULES.md#javascript-standard-style
  extends: 'standard',
  // required to lint *.vue files
  plugins: [
    'html'
  ],
  // "settings": {
  //   "import/resolver": {
  //     "webpack": {
  //       "config": "build/webpack.base.conf.js"
  //     }
  //   }
  // },
  // add your custom rules here
  'rules': {
    // 'semi': 2,
    // allow paren-less arrow functions
    'arrow-parens': 0,
    // allow async-await
    'generator-star-spacing': 0,
    // allow debugger during development
    'no-debugger': process.env.NODE_ENV === 'production' ? 2 : 0,
    // "import/no-extraneous-dependencies": [2, { devDependencies: true }],
    // "import/extensions": [
    //   "error",
    //   "always",
    //   {
    //     "js": "never",
    //     "vue": "never"
    //   }
    // ],
  }
}
