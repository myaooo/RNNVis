import d3 from 'd3'

function normalize(points, range=[-50, 50]) {
  const extents = calExtent(points);
  const factor_x = 100 / (extents[0][1] - extents[0][0]);
  const factor_y = 100 / (extents[1][1] - extents[1][0]);
  const factor = Math.max(factor_x, factor_y)
  for (let i = 0; i < points.length; i++){
    points[i].coords[0] *= factor;
    points[i].coords[1] *= factor;
  }
}

function calExtent(points) {
  var xExtent = d3.extent(points, function (d) { return d.coords[0] }),
    yExtent = d3.extent(points, function (d) { return d.coords[1] });
  return [xExtent, yExtent]
}
