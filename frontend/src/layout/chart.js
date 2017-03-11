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

function drawBox(el, width, mean, range1, range2) {
  const box = el.append('g');
  // const x = width / 2;
  // console.log(range1);
  box.append('rect')
    .attr('x', -width / 2)
    .attr('y', Math.min(range1[0], range1[1]))
    .attr('width', width)
    .attr('height', Math.abs(range1[1] - range1[0]));
  box.append('line')
    .attr('x1', -width / 2)
    .attr('y1', mean)
    .attr('x2', width / 2)
    .attr('y2', mean);
  for (let i = 0; i < 2; i += 1) {
    box.append('line')
      .attr('x1', 0)
      .attr('y1', range1[i])
      .attr('x2', 0)
      .attr('y2', range2[i]);
    box.append('line')
      .attr('x1', -width / 2)
      .attr('y1', range2[i])
      .attr('x2', width / 2)
      .attr('y2', range2[i]);
  }
  return box;
}

export class Chart {
  constructor(svg, width = 100, height = 100) {
    // svg: a handler from d3.select
    this.group = svg.append('g');
    this.width = width;
    this.height = height;
    this.marginAll = [0, 0, 0, 0];
    this.scale = { x: null, y: null };
    this.extents = [[Infinity, -Infinity], [Infinity, -Infinity]];
    this.axis = { x: null, y: null };
    this.charts = [];
    this.shapes = [];
    this.drawHooks = { xAxis: null, yAxis: null, shapes: [] };
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
    this.group.attr('transform', `translate(${x},${y})`);
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
  background(color, alpha) {
    this.bg = this.group.append('rect')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('fill', color)
      .attr('opacity', alpha);
    return this;
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
    this.updateExtentI(d3.extent(data, xfn), 0);
    this.scale.x = d3.scaleLinear()
      .domain(this.extents[0])
      .rangeRound([this.marginLeft, this.width - this.marginRight]);
    return this;
  }
  updateScaleY(data, yfn) {
    this.updateExtentI(d3.extent(data, yfn), 1);
    this.scale.y = d3.scaleLinear()
      .domain(this.extents[1])
      .rangeRound([this.height - this.marginBottom, this.marginTop]);
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
          .attr('transform', `translate(0, ${this.height - this.marginBottom})`)
          .call(d3.axisBottom(this.scale.x));
      };
    } else if (pos === 'top') {
      this.drawHooks.xAxis = () => {
        this.axis.x = this.group.append('g')
          .attr('transform', `translate(${this.marginLeft}, ${this.marginTop})`)
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
          .attr('transform', `translate(${this.marginLeft}, 0)`)
          .call(d3.axisLeft(this.scale.y));
      };
    } else if (pos === 'right') {
      this.drawHooks.yAxis = () => {
        this.axis.y = this.group.append('g')
          .attr('transform', `translate(${this.width - this.marginRight}, 0)`)
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
          width,
          this.scale.y(y0fn(d, i)),
          r1fn(d, i).map((v) => this.scale.y(v)),
          r2fn(d, i).map((v) => this.scale.y(v)),
        )
        .attr('transform', `translate(${xShift}, 0)`);
      });
    };
    this.shapes.push(handle);
    this.drawHooks.shapes.push(drawHook);
    return handle;
  }
  draw() {
    this.drawAxis();
    this.drawHooks.shapes.forEach((hook) => hook());
    this.charts.forEach((c) => c.draw());
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
