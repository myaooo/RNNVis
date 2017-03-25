import * as d3 from 'd3';
// import dimple from '../lib/dimple';

class Artist {
  constructor(chart) {
    this.chart = chart;
    // this.data = [];
    // this.areas = [];
    this.handles = []
    this.drawHooks = [];
  }
  get group() {
    return this.chart.group;
  }
  get scale() {
    return this.chart.scale;
  }
  draw() {
    this.drawHooks.forEach((drawHook) => drawHook());
  }
  clean() {
    this.handles.forEach((handle) => {
      handle
        .transition()
        .duration(200)
        .style('opacity', 0)
        .remove();
    });
    this.handles = [];
    this.drawHooks = [];
  }
}

const defaultColorScheme = d3.scaleOrdinal(d3.schemeCategory10);

function drawBox(el, xs, mean, range1, range2) {
  const box = el.append('g');
  // const x = width / 2;
  // console.log(range1);
  const mid = (xs[0] + xs[1]) / 2;
  box.append('rect')
    .attr('x', xs[0])
    .attr('y', Math.min(range1[0], range1[1]))
    .attr('width', xs[1] - xs[0])
    .attr('height', Math.abs(range1[1] - range1[0]));
  box.append('line')
    .attr('x1', xs[0])
    .attr('y1', mean)
    .attr('x2', xs[1])
    .attr('y2', mean);
  for (let i = 0; i < 2; i += 1) {
    box.append('line')
      .attr('x1', mid)
      .attr('y1', range1[i])
      .attr('x2', mid)
      .attr('y2', range2[i]);
    box.append('line')
      .attr('x1', xs[0])
      .attr('y1', range2[i])
      .attr('x2', xs[1])
      .attr('y2', range2[i]);
  }
  return box;
}

// Chart class, for usage example, see TestView.vue

export class Chart {
  // svg: a selector from d3.select, could be a <g>
  constructor(svg, width = 100, height = 100) {
    this.bggroup = svg.append('g');
    this.bg = this.bggroup.append('rect')
      .attr('opacity', 0);
    this.group = this.bggroup.append('g');
    this.width = width;
    this.height = height;
    this.marginAll = [0, 0, 0, 0];
    this.scale = { x: null, y: null };
    this.extents = [[Infinity, -Infinity], [Infinity, -Infinity]];
    this.axis = { x: null, y: null };
    this.axisHandles = { x: null, y: null };
    this.charts = [];
    this.artists = {}
    this.drawHooks = { xAxis: null, yAxis: null};
    this.offset = [];
    this.rotateFlag = false;
  }
  resize(width, height) {
    this.width = width;
    this.height = height;
    return this;
  }
  // set the relative translation regarding its mother element
  translate(x, y) {
    this.offset = [x, y];
    this.bggroup.attr('transform', `translate(${this.offset[0]},${this.offset[1]})`);
    // this.transform();
    return this;
  }
  // perform the transform rendering, will be called by `draw()`
  transform() {
    if (this.rotateFlag === true) {
      const x = this.width - this.marginLeft - this.marginRight;
      const y = this.height - this.marginTop - this.marginBottom;
      const rx = y / x;
      const ry = x / y;
      Object.keys(this.artists).forEach((artist) => {
        this.artists[artist].handles.forEach((handle) => {
          const scale = ''; //`scale(${rx}, ${ry}) `;
          const rotate = 'rotate(-90, 0, 0)';
          const translate = `translate(${-(this.width - this.marginLeft - this.marginRight)}, 0)`;
          handle.attr('transform', rotate + translate + scale);
        });
      });
    }
  }
  // rotate the chart, and perform axis rotation as well
  rotate() {
    this.rotateFlag = !this.rotateFlag;
    if (this.rotateFlag) {
      this.drawTmp = this.draw;
      const tmpSize = [this.width, this.height];
      const tmpMargin = this.marginAll;
      this.draw = () => {
        // placeholder for axis to prevent draw Axis
        this.axisHandles.x = this.axisHandles.y = 1;
        // reverse size
        this.width = tmpSize[1];
        this.height = tmpSize[0];
        this.marginAll = [tmpMargin[1], tmpMargin[2], tmpMargin[3], tmpMargin[0]];

        this.updateScale();
        this.drawTmp();
        // remove placeholder
        this.axisHandles.x = this.axisHandles.y = null;
        // reverse size back
        this.width = tmpSize[0];
        this.height = tmpSize[1];

        this.marginAll = tmpMargin;
        this.extents.reverse();
        this.updateScale();
        this.drawAxis();
        this.extents.reverse();

      };
    } else {
      this.draw = this.drawTmp;
    }
    return this;
  }
  // set the background color and opacity
  background(color, alpha) {
    this.bg
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('fill', color)
      .attr('opacity', alpha);
    return this;
  }
  // set the margin of the chart
  margin(top, right = top, bottom = top, left = right) {
    this.marginAll = [top, right, bottom, left];
    this.group.attr('transform', `translate(${this.marginLeft}, ${this.marginTop})`);
    return this;
  }
  // a few getter helpers
  get marginTop() {
    return this.marginAll[0];
  }
  get marginRight() {
    return this.marginAll[1];
  }
  get marginBottom() {
    return this.marginAll[2];
  }
  get marginLeft() {
    return this.marginAll[3];
  }
  // create a sub chart in this chart
  subChart(width = 100, height = 100) {
    const chart = new Chart(this.group, width, height)
      .translate(this.marginLeft, this.marginTop);
    this.charts.push(chart);
    return chart;
  }
  // update a extent for one dimension, i=0 -> x, i=1 -> y
  updateExtentI(extent, i) {
    const range = extent[1] - extent[0];
    // const timer = 10;
    // console.log(extent);
    // const maxRange = Math.max(range, this.extents[i][1] - this.extents[i][0]);
    const timer = range === 0 ? 1 : (10 < range ? 1 : Math.round(10 / range));
    this.extents[i][0] = Math.min(Math.floor(timer*(extent[0] - range * 0.02))/timer, this.extents[i][0]);
    this.extents[i][1] = Math.max(Math.ceil(timer*(extent[1] + range * 0.02))/timer, this.extents[i][1]);
    // console.log(`${i} > ${this.extents[i]}`)
    return this;
  }
  // update both extent
  updateExtent(xExtent, yExtent) {
    return this.updateExtentI(xExtent, 0).updateExtentI(yExtent, 1);
  }
  // update the scale function with given data
  updateScale(data, xfn = (d) => d[0], yfn = (d) => d[1]) {
    return this.updateScaleX(data, xfn).updateScaleY(data, yfn);
  }
  // update scale.x
  updateScaleX(data, xfn) {
    if (data) this.updateExtentI(d3.extent(data, xfn), 0);
    this.scale.x = d3.scaleLinear()
      .domain(this.extents[0])
      .rangeRound([0, this.width - this.marginRight - this.marginLeft]);
    return this;
  }
  // update scale.y
  updateScaleY(data, yfn) {
    if (data) this.updateExtentI(d3.extent(data, yfn), 1);
    this.scale.y = d3.scaleLinear()
      .domain(this.extents[1])
      .rangeRound([this.height - this.marginBottom - this.marginTop, 0]);
    return this;
  }
  // a function that draw axis automatically
  drawAxis() {
    if (!this.axisHandles.x && this.drawHooks.xAxis) {
      this.drawHooks.xAxis();
    }
    if (!this.axisHandles.y && this.drawHooks.yAxis) {
      this.drawHooks.yAxis();
    }
  }
  // set the x Axis draw function
  xAxis(label = 'x', pos = 'bottom') {
    let translateStr;
    if (pos === 'bottom') {
      translateStr = () => [0, this.scale.y(this.extents[1][0])];
      this.axis.x = d3.axisBottom();
    } else if (pos === 'top') {
      translateStr = () => [0, this.scale.y(this.extents[1][1])];
      this.axis.x = d3.axisTop();
    } else {
      // eslint-disable-next-line
      console.log(`Unknown axis position: ${pos}`);
      return this;
    }
    this.drawHooks.xAxis = () => {
      this.axis.x.scale(this.scale.x);
      this.axisHandles.x = this.group.append('g');
      this.axisHandles.x
        .attr('transform', 'translate(' + translateStr() + ')')
        .call(this.axis.x);
      if (label){
        const labelSize = 13;
        this.axisHandles.x.append('text')
          .attr('transform', 'translate(' + [this.scale.x(this.extents[0][1]), 0] + ')')
          .attr('dx', labelSize/2).attr('dy', labelSize/2)
          .attr('text-anchor', 'start')
          .attr('font-size', labelSize).style('fill', '#000')
          .text(label);
      }
    };
    return this;
  }
  yAxis(label='y', pos = 'left') {
    let translateStr;
    if (pos === 'left') {
      translateStr = () => [this.scale.x(this.extents[0][0]), 0];
      this.axis.y = d3.axisLeft();
    } else if (pos === 'right') {
      translateStr = () => [this.scale.x(this.extents[0][1]), 0];
      this.axis.y = d3.axisRight();
    } else {
      // eslint-disable-next-line
      console.log(`Unknown axis position: ${pos}`);
      return this;
    }
    this.drawHooks.yAxis = () => {
      this.axis.y.scale(this.scale.y);
      this.axisHandles.y = this.group.append('g')
        .attr('transform', 'translate(' + translateStr() + ')')
        .call(this.axis.y);
      if (label){
        const labelSize = 13;
        this.axisHandles.y.append('text')
          .attr('transform', 'translate(' + [0, this.scale.y(this.extents[1][1])] + ')')
          .attr('dx', 0).attr('dy', -labelSize/2)
          .attr('text-anchor', 'middle')
          .attr('font-size', labelSize).style('fill', '#000')
          .text(label);
      }
    };
    return this;
  }
  /*
  * major api of drawing a line,
  * this method will return a handle of <path> selector for users to further set styles
  */
  line(data, xfn = (d) => d[0], yfn = (d) => d[1]) {
    // console.log(data);
    this.updateScale(data, xfn, yfn);
    let lineArtist;
    if (this.artists.lines) {
      lineArtist = this.artists.lines;
    } else {
      const self = this;
      lineArtist = new LineArtist(self);
      this.artists.lines = lineArtist;
    }
    const handle = lineArtist.plot(data, xfn, yfn);
    return handle;
  }
  /*
  * major api of filling an area surrounded by 2 lines,
  * this method will return a handle of <path> selector for users to further set styles
  */
  area(data, xfn, y0fn, y1fn, color) {
    this.updateScaleX(data, xfn)
      .updateScaleY(data, y0fn)
      .updateScaleY(data, y1fn);
    let artist;
    if (this.artists.areas) {
      artist = this.artists.areas;
    } else {
      const self = this;
      artist = new AreaArtist(self);
      this.artists.areas = artist;
    }
    const handle = artist.plot(data, xfn, y0fn, y1fn, color);
    return handle;
  }
  /*
  * major api of drawing box plot,
  * this method will return a handle of <g> selector for users to further set styles
  */
  box(data, width, xfn, y0fn, r1fn, r2fn, params={}) {
    this.updateScaleX(data, xfn)
      .updateScaleY(data, (d, i) => r2fn(d, i)[0])
      .updateScaleY(data, (d, i) => r2fn(d, i)[1]);
    let artist;
    if (this.artists.boxes) {
      artist = this.artists.boxes;
    } else {
      const self = this;
      artist = new BoxArtist(self);
      this.artists.boxes = artist;
    }
    const handle = artist.plot(data, width, xfn, y0fn, r1fn, r2fn, params={});
    return handle;
  }
  // after all settings, call this methods for rendering
  draw() {
    this.transform();
    // this.drawHooks.shapes.forEach((hook) => hook());
    Object.keys(this.artists).forEach((name) => this.artists[name].draw());
    this.charts.forEach((c) => c.draw());
    this.drawAxis();
  }
  clean() {
    this.charts.forEach((c) => c.clean());
    Object.keys(this.artists).forEach((name) => this.artists[name].clean());
    Object.keys(this.axisHandles).forEach((name) => {
      if (this.axisHandles[name]) {
        this.axisHandles[name].remove();
        this.axisHandles[name] = null;
      }
    })
    this.marginAll = [0, 0, 0, 0];
    this.rotateFlag = false;
    this.scale = { x: null, y: null };
    this.extents = [[Infinity, -Infinity], [Infinity, -Infinity]];
    // this.axis = { x: null, y: null };
    this.drawHooks = { xAxis: null, yAxis: null};
    this.charts = [];
    return this;
  }
}

export class LineArtist extends Artist {
  constructor(chart) {
    super(chart);
    this.data = [];
    this.lines = [];
  }
  plot(data, xfn = (d) => d[0], yfn = (d) => d[1], color = defaultColorScheme(this.handles.length)) {
    const handle = this.group.append('path');
    const line = d3.line()
      .x((d, i) => this.scale.x(xfn(d, i)))
      .y((d, i) => this.scale.y(yfn(d, i)));
    const drawHook = () => {
      handle.datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        // .attr('stroke', color)
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round');
      // handle.attr('stroke', color);
    };
    this.data.push(data);
    this.handles.push(handle);
    this.lines.push(line);
    this.drawHooks.push(drawHook);
    return handle;
  }
}

export class AreaArtist extends Artist {
  constructor(chart) {
    super(chart);
    this.data = [];
    this.areas = [];
  }
  plot(data, xfn, y0fn, y1fn, color = defaultColorScheme(this.handles.length)) {
    const handle = this.group.append('path');
    const area = d3.area()
        .x((d, i) => this.scale.x(xfn(d, i)))
        .y0((d, i) => this.scale.y(y0fn(d, i)))
        .y1((d, i) => this.scale.y(y1fn(d, i)));
    const drawHook = () => {
      handle.datum(data);
      handle
        // .attr('fill', color)
        .attr('d', area);
    };
    this.data.push(data);
    this.handles.push(handle);
    this.areas.push(area);
    this.drawHooks.push(drawHook);
    return handle;
  }
}

export class BoxArtist extends Artist{
  constructor(chart) {
    super(chart);
    this.data = [];
    // this. = [];
  }
  plot(data, width, xfn, y0fn, r1fn, r2fn, params){
    const handle = this.group.append('g');
    const drawHook = () => {
      data.forEach((d, i) => {
        const xShift = this.scale.x(xfn(d, i));
        BoxArtist.drawBox(
          handle,
          [xShift - (width / 2), xShift + (width / 2)],
          this.scale.y(y0fn(d, i)),
          r1fn(d, i).map((v) => this.scale.y(v)),
          r2fn(d, i).map((v) => this.scale.y(v)),
        );
      });
    };
    this.handles.push(handle);
    this.drawHooks.push(drawHook);
    this.data.push(data);
    return handle;
  }
  static drawBox(el, xs, mean, range1, range2) {
    const box = el.append('g');
    // const x = width / 2;
    // console.log(range1);
    const mid = (xs[0] + xs[1]) / 2;
    box.append('rect')
      .attr('x', xs[0])
      .attr('y', Math.min(range1[0], range1[1]))
      .attr('width', xs[1] - xs[0])
      .attr('height', Math.abs(range1[1] - range1[0]));
    box.append('line')
      .attr('x1', xs[0])
      .attr('y1', mean)
      .attr('x2', xs[1])
      .attr('y2', mean)
      .attr('stroke-width', 0.5);
    for (let i = 0; i < 2; i += 1) {
      box.append('line')
        .attr('x1', mid)
        .attr('y1', range1[i])
        .attr('x2', mid)
        .attr('y2', range2[i])
        .attr('stroke-width', 0.5);
      box.append('line')
        .attr('x1', xs[0])
        .attr('y1', range2[i])
        .attr('x2', xs[1])
        .attr('y2', range2[i])
        .attr('stroke-width', 0.5);
    }
    return box;
  }
}

export default {
  // LineArtist,
  Chart,
};
