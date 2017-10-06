import * as d3 from 'd3';
import cloud from 'd3-cloud';

const bgLayout = {
  stroke: 'black',
  'stroke-width': 1,
  fill: 'white',
  'fill-opacity': 0.0,
  'stroke-opacity': 0.2,
};

// const wordLayout = {
//   font: 'Arial',
//   // 'fontSize': [12, 24],
//   fontSize: [9, 17],
//   // 'fontSize': [7, 11],
//   fontWeight: [300, 400, 500],
//   padding: 0,
//   opacity: 0.7,
//   baseColor: '#1f77b4',
// };

export class WordCloud {
  constructor(selector, size = [200, 200], {
    bgshape = 'rect',
    selectWordHandle = () => {},
    style = {
      font: 'Arial',
      fontSize: [12, 24],
      // fontSize: [9, 17],
      // 'fontSize': [7, 11],
      fontWeight: [300, 400, 500],
      padding: 0,
      opacity: 0.7,
      baseColor: '#1f77b4',
    }
  }, compare = false) {
    this.selector = selector;
    this.bggroup = this.selector.append('g');
    this.bg = this.bggroup.append('g');
    this.bgHandle = null;
    this.bgshape = bgshape;
    this.bgLayout = bgLayout;
    this.wordLayout = style;
    this.group = this.bggroup.append('g');
    this.data = null;
    this.cloud = null; // the handle to all the texts
    // this.font = style.font;
    this.margin_ = 0;
    this.colorScheme = d3.scaleOrdinal(d3.schemeCategory10);
    this.word2data;
    this.compare = compare;
    this.boundingSize = size;
    this.selectWordHandle = selectWordHandle;
    // this.selected = [];
    // this.bounding();
    // register event listener
  }
  get width() {
    return this.boundingSize[0] - (this.margin_ * 2);
  }
  get height() {
    return this.boundingSize[1] - (this.margin_ * 2);
  }

  size(size) {
    return arguments.length ? (this.boundingSize = size, this) : this.boundingSize;
  }
  wordLayoutParams(layoutParams) {
    return arguments.length ? (this.wordLayout = layoutParams, this) : this.wordLayout;
  }
  bgLayoutParams(layoutParams) {
    return arguments.length ? (this.bgLayout = layoutParams, this) : this.bgLayout;
  }
  transform(transformStr) {
    if (this.compare) transformStr += ' scale(-1, 1)';
    this.bggroup
      .transition()
      .duration(200)
      .attr('transform', transformStr);
    return this;
  }
  // set the background color and opacity
  background(color, alpha = 1.0) {
    this.bgLayout['fill'] = color;
    this.bgLayout['fill-opacity'] = alpha;
    return this;
  }
  drawBackground() {
    // console.log("Redrawing backgrounds")
    if (!this.bgHandle) {
      this.bgHandle = this.bg.append(this.bgshape);
    }
    if (this.bgshape === 'rect') {
      this.bgHandle
        .attr('x', -this.boundingSize[0]/2)
        .attr('y', -this.boundingSize[1]/2)
        .attr('rx', 4)
        .attr('ry', 4)
        .attr('width', this.boundingSize[0])
        .attr('height', this.boundingSize[1])
    } else if (this.bgshape === 'ellipse') {
      this.bgHandle
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('rx', this.boundingSize[0]/2)
        .attr('ry', this.boundingSize[1]/2);
    }
    Object.keys(this.bgLayout).forEach((param) => {
      this.bgHandle.attr(param, this.bgLayout[param]);
    });
    return this;
  }
  margin(margin) {
    return arguments.length ? (this.margin_ = margin, this) : this.margin_;
  }
  fontFamily(font) {
    return arguments.length ? (this.wordLayout.font = font, this) : this.wordLayout.font;
  }
  color(colorScheme) {
    return arguments.length ? (this.colorScheme = colorScheme, this) : this.colorScheme;
  }
  draw(size = this.boundingSize, data = this.data) {
    this.drawBackground();
    if (!this.data)
      this.data = data;
    // if (this.compare) this.group
    // console.log(data);
    const radiusX = size[0];
    const radiusY = size[1];
    const wordLayout = this.wordLayout;
    const selectWordHandle = this.selectWordHandle;
    const compare = this.compare;
    // this.group.attr('transform', 'translate(' + [-radiusX, -radiusY] + ')');
    const filterData = data.filter((d) => {
      return -radiusX < d.x - d.width / 4 && -radiusY < d.y - d.size / 2 && d.x + d.width / 4 < radiusX && d.y - d.size / 2 < radiusY;
    });
    const self = this;
    this.cloud = this.group.selectAll('text')
      .data(filterData, (d) => d.text); // matching key

    // Entering words
    const text = this.cloud.enter()
      .append('text')
      .style('font-family', wordLayout.font)
      .attr('text-anchor', 'middle')
      .style('fill', (d) => (typeof d.type === 'number' ? self.colorScheme(d.type) : wordLayout.baseColor));
    text
      .text((d) => d.text);

    this.cloud
      .style('fill', (d) => (typeof d.type === 'number' ? self.colorScheme(d.type) : wordLayout.baseColor));

    // .attr('font-size', 1);

    text
      .attr('transform', (d) => `translate(${d.x},${d.y})`)
      // .attr('font-size', 1)
      .attr('font-size', (d) => `${d.size}px`)
      .attr('font-weight', (d) => d.weight)
      .style('fill-opacity', 0)
      .transition()
      .duration(300)
      .style('fill-opacity', wordLayout.opacity);

    text
      .on('mouseover', function () {
        d3.select(this).style('fill-opacity', 1.0);
      })
      .on('mouseout', function (d) {
        if (d.select) return;
        d3.select(this).style('fill-opacity', wordLayout.opacity);
      })
      .on('click', function (d) {
        if (!d.select) {
          d.select = true;
          d.opacity = wordLayout.opacity;
          d.baseColor = typeof d.type === 'number' ? self.colorScheme(d.type) : wordLayout.baseColor;
          d3.select(this).style('fill-opacity', 1.0).style('font-weight', d.weight + 500);
          selectWordHandle({ word: d.text, compare });
        } else {
          d.select = false;
          selectWordHandle({ word: d.text, compare });
          d3.select(this).style('fill-opacity', wordLayout.opacity).style('font-weight', d.weight);
        }
      });

    // registering el in to datum
    text.each(function (d) {
      d.el = this;
    });

    this.word2data = {};
    this.data.forEach((d) => {
      this.word2data[d.text] = d;
    });
    // console.log(data);

    // Exiting words
    this.cloud.exit()
      .transition()
      .duration(200)
      .style('fill-opacity', 1e-6)
      .attr('font-size', 1)
      .remove();

    return this;

    // this._txt = null;
    // autoscale
    // setTimeout(() => self.autoscale(bounds), 100);
  }
  update(words) {
    words.sort((a, b) => a.size - b.size);
    const fontExtent = d3.extent(words, (d) => d.size);
    const scale = d3.scaleLinear()
      .range(this.wordLayout.fontSize)
      .domain(fontExtent);
    const weightScale = d3.scaleQuantize()
      .range(this.wordLayout.fontWeight)
      .domain(fontExtent);
    words.forEach((word) => {
      word.weight = Math.round(weightScale(word.size));
      word.size = scale(word.size);
    });
    // d3.cloud()
    // when layout, first give a larger region
    const boundingSize = [this.boundingSize[0] * 1.3, this.boundingSize[1] * 1.05];
    cloud()
      .size(boundingSize)
      .words(words)
      .padding(this.wordLayout.padding)
      .rotate(0)
      .font(this.wordLayout.font)
      .text(d => d.text)
      .fontSize(d => d.size)
      .fontWeight(d => d.weight)
      .on('end', (words_) => {
        // if(words_.length < words.length * 0.8){
        //   self.boundingSize = [self.boundingSize[0]*1.3, self.boundingSize[1]*1.05];
        //   console.log('word cloud size updated to ' + self.boundingSize);
        //   self.update(words);
        // } else {
        this.draw([this.width, this.height], words_);
        // }
      })
      .random(() => 0.5)
      .spiral('rectangular')
      .start();
    // return this
  }
  autoscale(bounds) {

    // console.log(`centerx: ${centerX}, centerY: ${centerY}`);
    const scaleX = 0.9 * this.width / Math.abs(bounds[0].x - bounds[1].x);
    const scaleY = 0.9 * this.height / Math.abs(bounds[0].y - bounds[1].y);
    const scale = Math.min(scaleX, scaleY);
    const centerX = (bounds[1].x + bounds[0].x - this.width) / 2 * scale;
    const centerY = (bounds[1].y + bounds[0].y - this.height) / 2 * scale;
    this.group.attr('transform', `scale(${scale}) translate(${-centerX}, ${-centerY})`);
  }
  destroy() {
    this.selector
      .transition()
      .duration(300)
      .attr('opacity', 0)
      .remove();
  }
}

export default {
  WordCloud,
};

