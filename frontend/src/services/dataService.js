/**
 * Created by mingyao on 2/17/17.
 */

let getProjectionData = function (model, field) {
//  empty api for future implementation
  return require('../assets/' + model + '-' + field + '-tsne.json')
}

let getStrengthData = function (model, field) {
  return require('../assets/' + model + '-' + field + '-strength.json')
}

let getTextData = function (model, field) {
  return [
    [['i', 0.2], ['love', 0.4], ['you', 0.5], ['omg', 0.2], ['<eos>', 0.1]],
    [['i', 0.4], ['like', 0.2], ['you', 0.3], ['<eos>', 0.1], ['omg', 0.2]]
  ]
}

export default {
  getProjectionData,
  getStrengthData,
  getTextData
}
