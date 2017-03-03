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

function mean(points){
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

export class kMeans{
  constructor(centroids, distance="euclidean") {
    this.centroids = centroids || [];
    this.distance = distance;
    if (typeof this.distance == "string") {
        this.distance = distances[this.distance];
    }
  }

  randomCentroids(points, k){
    let centroids = points.slice(0); // copy
    centroids.sort(function() {
        return (Math.round(Math.random()) - 0.5);
    });
    return centroids.slice(0, k);
  }

  classify(point) {
    let min = Infinity,
      index = 0;

    for (let i = 0; i < this.centroids.length; i++) {
      let dist = this.distance(point, this.centroids[i]);
      if (dist < min) {
        min = dist;
        index = i;
      }
    }

    return index;
  }

  cluster(points, k, maxIter, snapshotFn) {
    k = k || Math.max(2, Math.ceil(Math.sqrt(points.length / 2)));
    maxIter = maxIter || 100
    this.centroids = this.randomCentroids(points, k);

    var assignment = new Int32Array(points.length);
    var clusters = new Array(k);

    var iterations = 0;
    var movement = true;
    while (movement) {
      // update point-to-centroid assignments
      for (let i = 0; i < points.length; i++) {
         assignment[i] = this.classify(points[i], distance);
      }

      // update location of each centroid
      movement = false;
      for (let j = 0; j < k; j++) { // iterate over k clusters
        let assigned = points.filter((p, i) => {assignment[i] == j})

        if (!assigned.length) {
          continue;
        }

        let centroid = this.centroids[j];
        let newCentroid = new mean(assigned);

        for (let g = 0; g < centroid.length; g++) {
          if (newCentroid[g] != centroid[g]) {
            movement = true;
          }
        }

        this.centroids[j] = newCentroid;
        clusters[j] = assigned;
      }
      if (snapshotFn){
        snapshotFn(clusters, iterations++);
      }
    }

    return clusters;
  }

  toJSON() {
    return JSON.stringify(this.centroids);
  }

  fromJSON() {
    this.centroids = JSON.parse(json);
    return this;
  }
}

export function kmeans(vectors, k, maxIter=100, distance="euclidean", snapshotFn=null) {
   return (new KMeans(distance)).cluster(vectors, k, maxIter, snapshotFn);
}
