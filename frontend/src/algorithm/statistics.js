// pre-process helpers

export const distances = {
  euclidean: function(v1, v2) {
    var total = 0;
    for (var i = 0; i < v1.length; i++) {
        total += Math.pow(v2[i] - v1[i], 2);
    }
    return Math.sqrt(total);
  },
  manhattan: function(v1, v2) {
    var total = 0;
    for (var i = 0; i < v1.length ; i++) {
      total += Math.abs(v2[i] - v1[i]);
    }
    return total;
  },
  max: function(v1, v2) {
    var max = 0;
    for (var i = 0; i < v1.length; i++) {
      max = Math.max(max , Math.abs(v2[i] - v1[i]));
    }
    return max;
  }
};

export function mean(points){
  var meanPoint = new Float32Array(points[0].length);
  for(let i = 0; i < meanPoint.length; i++){
    let sum = 0;
    for (let j = 0; j < points.length; j++) {
      sum += points[j][i];
    }
    meanPoint[i] = sum / points.length;
  }
  return meanPoint;
}
