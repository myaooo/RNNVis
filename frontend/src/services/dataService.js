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

export default {
  getProjectionData,
  getStrengthData
}
