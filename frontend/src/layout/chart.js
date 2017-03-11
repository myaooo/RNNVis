import * as d3 from 'd3';
// import dimple from '../lib/dimple';

// class Artist {
//   constructor(chart) {
//     this.chart = chart;
//   }
//   draw() {
//     throw new Error('Calling abstract interface of Artist! Call a subclass instance instead!');
//   }
// }

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

export class Chart {
  constructor(svg, width = 100, height = 100) {
    // svg: a handler from d3.select
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
    this.charts = [];
    this.shapes = [];
    this.drawHooks = { xAxis: null, yAxis: null, shapes: [] };
    this.offset = [];
    this.rotateFlag = false;
  }
  width(value) {
    this.width = value;
    return this;
  }
  height(height) {
    this.height = height;
    return this;
  }
  translate(x, y) {
    this.offset = [x, y];
    this.bggroup.attr('transform', `translate(${this.offset[0]},${this.offset[1]})`);
    // this.transform();
    return this;
  }
  transform() {
    if (this.offset.length) {
      this.group.attr('transform', `translate(${this.marginLeft}, ${this.marginTop})`);
    }
    if (this.rotate === true) {
      const x = this.width - this.marginLeft - this.marginRight;
      const y = this.height - this.marginTop - this.marginBottom;
      const rx = y / x;
      const ry = x / y;
      this.shapes.forEach((handle) => {
        const scale = `scale(${rx}, ${ry}) `;
        const rotate = 'rotate(-90) ';
        const translate = `translate(${-(this.height - this.marginBottom - this.marginTop)}, 0)`;
        handle.attr('transform', rotate + translate + scale);
      });
    }
  }
  rotate() {
    this.rotateFlag = !this.rotateFlag;
    if (this.rotateFlag) {
      this.drawTmp = this.draw;
      this.draw = () => {
        this.axis.x = this.axis.y = 1;
        this.drawTmp();
        this.axis.x = this.axis.y = null;
        this.extents.reverse();
        this.updateScale();
        this.drawAxis();
        this.extents.reverse();
        this.updateScale();
      };
    } else {
      this.draw = this.drawTmp;
    }
    return this;
  }
  background(color, alpha) {
    this.bg
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('fill', color)
      .attr('opacity', alpha);
    return this;
  }
  margin(top, right = top, bottom = top, left = right) {
    this.marginAll = [top, right, bottom, left];
    return this;
  }
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
  subChart(width = 100, height = 100) {
    const chart = new Chart(this.group, width, height)
      .translate(this.marginLeft, this.marginTop);
    this.charts.push(chart);
    return chart;
  }
  updateExtentI(extent, i) {
    const range = extent[1] - extent[0];
    this.extents[i][0] = Math.min(extent[0] - (range * 0.05), this.extents[i][0]);
    this.extents[i][1] = Math.max(extent[1] + (range * 0.05), this.extents[i][1]);
    return this;
  }
  updateExtent(xExtent, yExtent) {
    return this.updateExtentI(xExtent, 0).updateExtentI(yExtent, 1);
  }
  updateScale(data, xfn = (d) => d[0], yfn = (d) => d[1]) {
    return this.updateScaleX(data, xfn).updateScaleY(data, yfn);
  }
  updateScaleX(data, xfn) {
    if (data) this.updateExtentI(d3.extent(data, xfn), 0);
    this.scale.x = d3.scaleLinear()
      .domain(this.extents[0])
      .rangeRound([0, this.width - this.marginRight - this.marginLeft]);
    return this;
  }
  updateScaleY(data, yfn) {
    if (data) this.updateExtentI(d3.extent(data, yfn), 1);
    this.scale.y = d3.scaleLinear()
      .domain(this.extents[1])
      .rangeRound([this.height - this.marginBottom - this.marginTop, 0]);
    return this;
  }
  drawAxis() {
    if (!this.axis.x && this.drawHooks.xAxis) {
      this.drawHooks.xAxis();
    }
    if (!this.axis.y && this.drawHooks.yAxis) {
      this.drawHooks.yAxis();
    }
  }
  xAxis(pos = 'bottom') {
    if (pos === 'bottom') {
      this.drawHooks.xAxis = () => {
        this.axis.x = this.group.append('g')
          .attr('transform', `translate(0, ${this.scale.y(this.extents[1][0])})`)
          .call(d3.axisBottom(this.scale.x));
      };
    } else if (pos === 'top') {
      this.drawHooks.xAxis = () => {
        this.axis.x = this.group.append('g')
          .attr('transform', `translate(0, ${this.scale.y(this.extents[1][1])})`)
          .call(d3.axisTop(this.scale.x));
      };
    } else {
      // eslint-disable-next-line
      console.log(`Unknown axis position: ${pos}`);
    }
    return this;
  }
  yAxis(pos = 'left') {
    if (pos === 'left') {
      this.drawHooks.yAxis = () => {
        this.axis.y = this.group.append('g')
          .attr('transform', `translate(${this.scale.x(this.extents[0][0])}, 0)`)
          .call(d3.axisLeft(this.scale.y));
      };
    } else if (pos === 'right') {
      this.drawHooks.yAxis = () => {
        this.axis.y = this.group.append('g')
          .attr('transform', `translate(${this.scale.x(this.extents[0][1])}, 0)`)
          .call(d3.axisRight(this.scale.y));
      };
    } else {
      // eslint-disable-next-line
      console.log(`Unknown axis position: ${pos}`);
    }
    return this;
  }
  line(data, xfn = (d) => d[0], yfn = (d) => d[1]) {
    // console.log(data);
    this.updateScale(data, xfn, yfn);
    const handle = this.group.append('path');
    const drawHook = () => {
      const line = d3.line()
        .x((d, i) => this.scale.x(xfn(d, i)))
        .y((d, i) => this.scale.y(yfn(d, i)));

      handle.datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', defaultColorScheme(this.shapes.length))
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round');
    };
    this.shapes.push(handle);
    this.drawHooks.shapes.push(drawHook);
    // this.drawAxis();
    return handle;
  }
  area(data, xfn, y0fn, y1fn) {
    this.updateScaleX(data, xfn)
      .updateScaleY(data, y0fn)
      .updateScaleY(data, y1fn);
    const areaGroup = this.group.append('g');
    const handle = areaGroup.append('path');
    const drawHook = () => {
      const area = d3.area()
        .x((d, i) => this.scale.x(xfn(d, i)))
        .y0((d, i) => this.scale.y(y0fn(d, i)))
        .y1((d, i) => this.scale.y(y1fn(d, i)));
      // const lineAboveId = `${this.shapes.length}-line-above`;
      // const lineBelowId = `${this.shapes.length}-line-below`;
      handle.datum(data)
        .attr('fill', 'steelblue')
        .attr('d', area);
    };
    // const lineAbove = areaGroup.append('path')
    //   .datum(data)
    //   .attr('id', `${this.shapes.length}-line-above`)
    this.shapes.push(handle);
    this.drawHooks.shapes.push(drawHook);
    return handle;
  }
  box(data, width, xfn, y0fn, r1fn, r2fn) {
    this.updateScaleX(data, xfn)
      .updateScaleY(data, (d, i) => r2fn(d, i)[0])
      .updateScaleY(data, (d, i) => r2fn(d, i)[1]);
    const handle = this.group.append('g');
    const drawHook = () => {
      data.forEach((d, i) => {
        const xShift = this.scale.x(xfn(d, i));
        drawBox(
          handle,
          [xShift - (width / 2), xShift + (width / 2)],
          this.scale.y(y0fn(d, i)),
          r1fn(d, i).map((v) => this.scale.y(v)),
          r2fn(d, i).map((v) => this.scale.y(v)),
        );
        // .attr('transform', `translate(${xShift}, 0)`);
      });
    };
    this.shapes.push(handle);
    this.drawHooks.shapes.push(drawHook);
    return handle;
  }
  draw() {
    this.transform();
    this.drawHooks.shapes.forEach((hook) => hook());
    this.charts.forEach((c) => c.draw());
    this.drawAxis();
  }
}

// export class LineArtist extends Artist {
//   constructor(chart) {
//     super(chart);
//     this.group = null;
//     this.lines = [];
//     this.lineHandles = [];
//   }
//   plot(data, xfn, yfn) {
//     // const data = x.map((d, i) => { return [d, y[i]]; });
//     this.lines.push(() => {
//       const line = d3.line();
//       if (xfn) line.x(xfn);
//       if (yfn) line.y(yfn);
//       const handle = this.group.append('path')
//         .datum(data)
//         .attr('d', line);
//       this.lineHandles.push(handle);
//     });
//     // return this.lineHandles[this.lineHandles.length - 1];
//   }
//   draw() {
//     this.group = this.chart.group.append('g')
//       .attr('transform', `translate(${this.margin[3]},${this.margin[0]})`);
//     this.lines.forEach((line) => { line(); });
//   }

// }

export default {
  // LineArtist,
  Chart,
};
