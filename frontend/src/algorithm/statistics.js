// pre-process helpers

const FloatArray = Float32Array;

export const distance = {
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

export function mean(arrays) {
  const meanPoint = new FloatArray(arrays[0].length);
  for(let i = 0; i < meanPoint.length; i++){
    let sum = 0;
    for (let j = 0; j < arrays.length; j++) {
      sum += arrays[j][i];
    }
    meanPoint[i] = sum / arrays.length;
  }
  return meanPoint;
}

export function sequenceDiff(arrays) {
  const diffArrays = new Array(arrays[0].length - 1);
  for (let i = 0; i < diffArrays.length; i++) {
    diffArrays[i] = minus(arrays[i+1], arrays[i]);
  }
  return diffArrays;
}

export function minus(arrayA, arrayB) {
  const diffArray = FloatArray.from(arrayA);
  for (let i = 0; i < diffArrays.length; i++) {
    diffArray[i] -= arrayB[i];
  }
  return diffArray
}

export default {
  mean,
  distance
}
