import * as d3 from 'd3';

function rectCollide() {
  var nodes, sizes, masses
  var size = functor([0, 0])
  var strength = 1
  var iterations = 1

  function force() {
    var node, size, mass, xi, yi
    var i = -1
    while (++i < iterations) { iterate() }

    function iterate() {
      var j = -1
      var tree = d3.quadtree(nodes, xCenter, yCenter).visitAfter(prepare)

      while (++j < nodes.length) {
        node = nodes[j]
        size = sizes[j]
        mass = masses[j]
        xi = xCenter(node)
        yi = yCenter(node)

        tree.visit(apply)
      }
    }

    function apply(quad, x0, y0, x1, y1) {
      var data = quad.data
      var xSize = (size[0] + quad.size[0]) / 2
      var ySize = (size[1] + quad.size[1]) / 2
      if (data) {
        if (data.index <= node.index) { return }

        var x = xi - xCenter(data)
        var y = yi - yCenter(data)
        var xd = Math.abs(x) - xSize
        var yd = Math.abs(y) - ySize

        if (xd < 0 && yd < 0) {
          var l = Math.sqrt(x * x + y * y)
          var m = masses[data.index] / (mass + masses[data.index])

          if (Math.abs(xd) < Math.abs(yd)) {
            node.vx -= (x *= xd / l * strength) * m
            data.vx += x * (1 - m)
          } else {
            node.vy -= (y *= yd / l * strength) * m
            data.vy += y * (1 - m)
          }
        }
      }

      return x0 > xi + xSize || y0 > yi + ySize ||
        x1 < xi - xSize || y1 < yi - ySize
    }

    function prepare(quad) {
      if (quad.data) {
        quad.size = sizes[quad.data.index]
      } else {
        quad.size = [0, 0]
        var i = -1
        while (++i < 4) {
          if (quad[i] && quad[i].size) {
            quad.size[0] = Math.max(quad.size[0], quad[i].size[0])
            quad.size[1] = Math.max(quad.size[1], quad[i].size[1])
          }
        }
      }
    }
  }

  function xCenter(d) { return d.x + d.vx + sizes[d.index][0] / 2 }
  function yCenter(d) { return d.y + d.vy + sizes[d.index][1] / 2 }

  force.initialize = function (_) {
    sizes = (nodes = _).map(size)
    masses = sizes.map(function (d) { return d[0] * d[1] })
  }

  force.size = function (_) {
    return (arguments.length
      ? (size = typeof _ === 'function' ? _ : functor(_), force)
      : size)
  }

  force.strength = function (_) {
    return (arguments.length ? (strength = +_, force) : strength)
  }

  force.iterations = function (_) {
    return (arguments.length ? (iterations = +_, force) : iterations)
  }

  return force
}

function boundedBox() {
  var nodes, sizes
  var bounds
  var size = functor([0, 0])

  function force() {
    var node, size
    var xi, x0, x1, yi, y0, y1
    var i = -1
    while (++i < nodes.length) {
      node = nodes[i]
      size = sizes[i]
      xi = node.x + node.vx
      x0 = bounds[0][0] - xi
      x1 = bounds[1][0] - (xi + size[0])
      yi = node.y + node.vy
      y0 = bounds[0][1] - yi
      y1 = bounds[1][1] - (yi + size[1])
      if (x0 > 0 || x1 < 0) {
        node.x += node.vx
        node.vx = -node.vx
        if (node.vx < x0) { node.x += x0 - node.vx }
        if (node.vx > x1) { node.x += x1 - node.vx }
      }
      if (y0 > 0 || y1 < 0) {
        node.y += node.vy
        node.vy = -node.vy
        if (node.vy < y0) { node.vy += y0 - node.vy }
        if (node.vy > y1) { node.vy += y1 - node.vy }
      }
    }
  }

  force.initialize = function (_) {
    sizes = (nodes = _).map(size)
  }

  force.bounds = function (_) {
    return (arguments.length ? (bounds = _, force) : bounds)
  }

  force.size = function (_) {
    return (arguments.length
      ? (size = typeof _ === 'function' ? _ : functor(_), force)
      : size)
  }

  return force
}

function functor(_) {
  if (typeof _ === 'function') {
    return _;
  } else {
    return () => _;
  }
}

class ForceCloud {
  constructor() {
    this.data;
    this._words;
    this._font = () => 'Impact';
    this._text = (d) => d.text;
    this._fontSize = () => 10;
    this._fontWeight = () => 200;
    this._fontStyle = () => 'normal';
    this._padding = () => 1;
    this._size = [200, 200];
    this._radius = [100, 100];
    this._timeInterval = Infinity;
    this.event = d3.dispatch('end', 'ticked');
    this.timer = null;
    this.simulationTime = 1000;
    this.simulation;
    const canvas = document.createElement('canvas');
    this.c = canvas.getContext('2d');
    this.c.fillStyle = this.c.strokeStyle = 'red';
    this.c.textAlign = 'center';
    this._ticked = (data) => { return; };
  }
  words(words) {
    this._words = words;
    return this;
  }
  font(_) {
    return arguments.length ? (this._font = functor(_), this) : this._font;
  }
  size(shape) {
    return arguments.length ? (this._size = shape, this) : this._size;
  }
  fontSize(_) {
    return arguments.length ? (this._fontSize = functor(_), this) : this._fontSize;
  }
  fontWeight(_) {
    return arguments.length ? (this._fontWeight = functor(_), this) : this._fontWeight;
  }
  fontStyle(_) {
    return arguments.length ? (this._fontStyle = functor(_), this) : this._fontStyle;
  };
  text(_) {
    return arguments.length ? (this._text = functor(_), this) : this._text;
  }
  padding(_) {
    return arguments.length ? (this._padding = functor(_), this) : this._padding;
  }
  rotate(_) { // do nothing
    if (_ || _ !== 0) console.log('do not support rotation!');
    return this;
  }
  on() {
    var value = this.event.on.apply(this.event, arguments);
    return value === this.event ? this : value;
  }
  stop() {
    if (this.timer) {
      // clearInterval(this.timer);
      this.timer = null;
    }
    return this;
  }
  ticked(_) {
    return arguments.length ? (this._ticked = _, this) : this._ticked;
  }
  start() {
    const self = this;
    this.timer = this.simulationTime;

    const ratio = 1;
    this.data = this._words.map((d, i, arr) => {
      let angle = Math.random() * 360;
      const e = {
        style: this._fontStyle(d, i, arr),
        weight: this._fontWeight(d, i, arr),
        size: this._fontSize(d, i, arr),
        font: this._font(d, i, arr),
        text: this._text(d, i, arr),
        padding: this._padding(d, i, arr),
        x: Math.random() * 2 * this._radius[0] - this._radius[0],
        y: Math.random() * 2 * this._radius[1] - this._radius[1],
        vx: 0.1 * Math.cos(angle / 180 * Math.PI),
        vy: 0.1 * Math.sin(angle / 180 * Math.PI),
      }
      e.width = this.getTextWidth(d, ratio) * 1.5 + this._padding(d, i, arr);
      e.height = e.size * 1.1 + this._padding(d, i, arr);
      e.area = e.width * e.height;
      console.log(`${e.text}, size: ${e.size}, width: ${e.width}`);
      return e;
    })
    // console.log(this.data);
    console.log([d3.extent(self.data, (d) => d.x), d3.extent(self.data, (d) => d.y)]);
    console.log(this._radius);
    var collisionForce = rectCollide()
      .size((d) => [d.width, d.height])
      .strength(10)
      .iterations(5);

    var boxForce = boundedBox()
      .bounds([[-this._radius[0]*0.9, -this._radius[1]*0.9], [this._radius[0]*0.9, this._radius[1]*0.9]])
      .size((d) => [d.width, d.height]);

    const strengthScale = d3.scalePow()
      .domain(d3.extent(this.data, (d) => d.area))
      .range([0.01, 0.5]);
    // console.log(strengthScale(150));
    // console.log(strengthScale(1000));
    // console.log(d3.extent(this.data, (d) => d.area));

    var centerForceX = d3.forceX()
      .x((d) => -d.width/2)
      .strength((d) => strengthScale(d.area));
    var centerForceY = d3.forceY()
      .y((d) => d.height/2)
      .strength((d) => strengthScale(d.area));

    let counter = 0;
    this.simulation = d3.forceSimulation()
      .velocityDecay(0.9)
      .alpha(0.3)
      .alphaDecay(0.09)
      .alphaTarget(0)
      .force('box', boxForce)
      .force('collision', collisionForce)
      .force('x', centerForceX)
      .force('y', centerForceY)
      .on('tick', () => {
        counter++;
        // console.log(counter);
        // if (counter% 10 === 0)
        // self.event.call('ticked', self, self.data);
        // console.log([d3.extent(self.data, (d) => d.x), d3.extent(self.data, (d) => d.y)]);
        self.data.forEach((d) => {
          const x = d.x + d.width / 2;
          const y = d.y + d.height / 2;
          const dist2 = (x / self._radius[0] * 1.2) ** 2 + (y / self._radius[1] * 1.2) ** 2;
          if (dist2 > 1){
            const scale = Math.sqrt(dist2);
            // d.vx = - scale * x;
            // d.vy = - scale * y;
            d.x /= scale;
            d.y /= scale;
          }
        });
        this._ticked(this.data);
      })
      .nodes(this.data);

    setTimeout(() => {
      this.simulation.stop();
      this.stop();
      this.event.call('end', this, this.data);
    }, this.simulationTime);
    // console.log(this.data);
  }

  getTextWidth(d, ratio) {
    this.c.save();
    this.c.font = d.style + ' ' + d.weight + ' ' + ~~((d.size + 1) / ratio) + 'px ' + d.font;
    var w = this.c.measureText(d.text + 'm').width * ratio;
    this.c.restore();
    return w;
  }

  radius(radius) {
    return arguments.length ? (this._radius = radius, this) : this._radius;
  }

  timeInterval = function (_) {
    return arguments.length ? (timeInterval = _ == null ? Infinity : _, this) : this._timeInterval;
  };

}

function cloud() {
  return new ForceCloud();
}

export default cloud;
