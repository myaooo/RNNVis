import * as d3 from 'd3';
import cloud from 'd3-cloud';
// console.log(d3);
// var cloud = require('./d3.cloud.js');
// console.log('haha');

export class WordCloud{
  constructor(selector, radiusX = 100, radiusY = radiusX) {
    this.selector = selector;
    this.bggroup = this.selector.append('g');
    this.bg = this.bggroup.append('ellipse')
      .attr('fill-opacity', 0.0)
      .attr('cx', 0)
      .attr('cy', 0)
      .attr('rx', radiusX)
      .attr('ry', radiusY);
    this.group = this.bggroup.append('g');
    this.radiusX = radiusX;
    this.radiusY = radiusY;
    this._data = [];
    this.cloud = null;  // the handle to all the texts
    this.font = 'Impact';
    this.offset = null;
    this.rotateDegree = 0;
    this.margin_ = 0;
    this.scale = null;
    this.colorScheme = d3.scaleOrdinal(d3.schemeCategory10);
    this.bounding();
  }
  get width() {
    return (this.radiusX - this.margin_) * 2;
  }
  get height() {
    return (this.radiusX - this.margin_) * 2;
  }
  // set the relative translation regarding its mother element
  translate(x, y) {
    this.offset = [x, y];
    this.transform();
    return this;
  }
  rotate(degree) {
    this.rotateDegree = degree;
    this.transform();
    return this;
  }
  // set the background color and opacity
  background(color, alpha = 1.0) {
    this.bg
      .attr('fill', color)
      .attr('fill-opacity', alpha);
    return this;
  }
  bounding(parameters = [['stroke', this.colorScheme(0)], ['stroke-dasharray', '5,5'], ['stroke-width', '2px']]) {
    parameters.forEach((parameter) => {
      this.bg.attr(parameter[0], parameter[1]);
    });
    return this;
  }
  scale(scaleX, scaleY) {
    this.scale = {x: scaleX, y: scaleY};
    this.transform();
    return this;
  }
  margin(margin) {
    this.margin_ = margin;
    // this.group.attr('transform', `scale()`)
    return this;
  }
  // perform the transform rendering, will be called by `draw()`
  transform() {
    const translate = this.offset ? `translate(${this.offset[0]}, ${this.offset[1]})` : '';
    const rotate = this.rotateDegree ? `rotate(${this.rotateDegree}, 0, 0)` : '';
    const scale = this.scaleRatio ? `scale(${this.scale.x}, ${this.scale.y})` : '';
    const transform = scale + rotate + translate;
    if (transform)
      this.bggroup.attr('transform', transform);
    return this;
  }
  fontFamily(font) {
    this.font = font;
    return this;
  }
  color(colorScheme) {
    this.colorScheme = colorScheme;
  }
  draw(data, bounds) {
    // console.log(this.cloud);
    const self = this;
    this.cloud = this.group.selectAll('g text')
      .data(data, function (d) { return d.text; }); // matching key
    // console.log(data);
    //Entering words
    const texts = this.cloud.enter()
      .append('text')
      .style('font-family', this.font)
      .style('fill', (d, i) => { return self.colorScheme(d.type); })
      .attr('text-anchor', 'middle')
      // .attr('font-size', 1)
      .text(function (d) { return d.text; });


    texts
      .style('font-size', function (d) { return d.size + 'px'; })
      .attr('transform', function (d) {
        return 'translate(' + [d.x, d.y] + ')rotate(' + d.rotate + ')';
      })
      .style('fill-opacity', 1);;

    //Entering and existing words
    // this.cloud
    //   // .transition()
    //   // .duration(600)
    //   .style('font-size', function (d) { return d.size + 'px'; })
    //   .attr('transform', function (d) {
    //     return 'translate(' + [d.x, d.y] + ')rotate(' + d.rotate + ')';
    //   })
    //   .style('fill-opacity', 1);

    //Exiting words
    this.cloud.exit()
      // .transition()
      // .duration(200)
      // .style('fill-opacity', 1e-6)
      // .attr('font-size', 1)
      .remove();

    // autoscale
    setTimeout(() => self.autoscale(bounds), 100);
    // this.autoscale();
  }
  autoscale(bounds) {
    // console.log(bounds);
    // console.log(`centerx: ${centerX}, centerY: ${centerY}`);
    const scaleX = 0.8 * this.width / Math.abs(bounds[0].x - bounds[1].x);
    const scaleY = 0.8 * this.height / Math.abs(bounds[0].y - bounds[1].y);
    const scale = Math.min(scaleX, scaleY);
    const centerX = (bounds[1].x + bounds[0].x - this.width) / 2 * scale;
    const centerY = (bounds[1].y + bounds[0].y - this.height) / 2 * scale;
    this.group.attr('transform', `scale(${scale}) translate(${-centerX}, ${-centerY})`);
  }
  update(words) {
    const self = this;
    // console.log(this.width);
    cloud().size([this.width, this.height])
      .words(words)
      .padding(1)
      .rotate(0)
      .font(this.font)
      .fontSize(function (d) { return d.size; })
      .on('end', (words, bounds) => self.draw(words, bounds))
      .start();
    // return this
  }
}

export default {
  WordCloud,
};
