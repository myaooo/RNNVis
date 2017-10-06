// import d3 from 'd3';

// function normalize(points, range = [-50, 50]) {
//   const extents = calExtent(points);
//   const factor_x = 100 / (extents[0][1] - extents[0][0]);
//   const factor_y = 100 / (extents[1][1] - extents[1][0]);
//   const factor = Math.max(factor_x, factor_y);
//   for (let i = 0; i < points.length; i++){
//     points[i].coords[0] *= factor;
//     points[i].coords[1] *= factor;
//   }
// }

// function calExtent(points) {
//   var xExtent = d3.extent(points, function (d) { return d.coords[0]; }),
//     yExtent = d3.extent(points, function (d) { return d.coords[1]; });
//   return [xExtent, yExtent];
// }

export function memorize(fn) {
  const cache = {};
  return function (...args) {
    const key = args.length + Array.prototype.join.call(args, ',');
    if (!(key in cache)) {
      cache[key] = fn.apply(this, args);
    }
    return cache[key];
  };
}

export function memorizePromise(fn) {
  const cache = {};
  return function (...args) {
    const key = args.length + Array.prototype.join.call(args, ',');
    if (!(key in cache)) {
      return fn(...args).then(value => {
        cache[key] = value;
        return Promise.resolve(value);
      });
    }
    return Promise.resolve(cache[key]);
  };
}

export function isString(obj) {
  return typeof obj === 'string' || obj instanceof String;
}

const has = Object.prototype.hasOwnProperty;

export default {
  memorize,
  memorizePromise,
  has,
  isString,
};
