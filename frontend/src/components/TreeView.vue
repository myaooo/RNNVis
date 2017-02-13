<template>
  <div class="tree">
  </div>
</template>
<script>
  import * as d3 from 'd3';
  export default {
    name: 'TreeView',
    data() {
      return {
      }
    },
    mounted() {
      var treeData = require('./test.json');
      const maxDepth = 6;
      const duration = 500;

      function treeNode(x, parent = null) {
        const self = {
          name: x.word,
          weight: Math.pow(x.cond_prob, 2),
          prob: x.cond_prob,
          value: x.cond_prob,
          children: null,
          _children: null,
          node: null,
          link: null
        };

        self.children = x.children_id
            .map(id => treeData.nodes[id])
            .map(y => treeNode(y, self));

        if (self.children !== null) {
          let m = 0;
          for (const y of self.children) {
            m += y.weight;
          }
          for (const y of self.children) {
            y.weight /= m;
          }
          self.children.sort((a, b) => b.weight - a.weight);
        }
        return self;
      }

      function traverse(x, depth = 0) {
        if (x.children !== null) {
          for (const y of x.children) {
            y.prob = x.prob * y.weight;
            traverse(y, depth + 1);
          }
        }

        if (depth > maxDepth) {
          const t = x.children;
          x.children = x._children;
          x._children = t;
        }
      }

      var data = treeNode(treeData.nodes[treeData.root_id]);
      traverse(data);

      // set the dimensions and margins of the diagram
      var margin = {
          top: 20,
          right: 120,
          bottom: 20,
          left: 120
        },
        width = 1280 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

      // declares a tree layout and assigns the size
      var tree = d3.tree()
        .size([height, width]);

      // append the svg object to the body of the page
      // appends a 'group' element to 'svg'
      // moves the 'group' element to the top left margin
      var svg = d3.select('.tree').append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom),
        g = svg.append('g')
        .attr('transform',
          'translate(' + margin.left + ',' + margin.top + ')');

      update(300, 0);
      // adds the links between the nodes
      function update(x0, y0, removed = []) {
        console.log(removed);
        var nodes = d3.hierarchy(data, function (d) {
          return d.children;
        });

        nodes = tree(nodes);
        removed.forEach(function(d){
          const nodeExit = d.node.transition()
            .duration(duration)
            .attr("transform", "translate(" + y0 + "," + x0 + ")")
            .remove();

          nodeExit.select("circle")
            .attr("r", 1e-6);

          nodeExit.select("text")
            .style("fill-opacity", 1e-6);

          d.node = null;

          const linkExit = d.link.transition()
            .duration(duration)
            .attr('d', 'M' + y0 + ',' + x0 +
                'C' + y0 + ',' + x0 +
                ' ' + y0 + ',' + x0 +
                ' ' + y0 + ',' + x0)
            .remove();
          
          d.link = null;
        });

        g.selectAll('.link')
          .data(nodes.descendants().slice(1))
          .exit().transition()
          .duration(duration)
          .attr('d', function (d) {
            return 'M' + y0 + ',' + x0 +
              'C' + y0 + ',' + x0 +
              ' ' + y0 + ',' + x0 +
              ' ' + y0 + ',' + x0;
          })
          .remove();

        nodes.descendants().slice(1).forEach(function(d){
          let link;
          if (d.data.link === null) {
            link = d.data.link = g.append('path')
              .attr('class', 'link')
              .style('stroke', 'lightsteelblue')
              .attr('d', 'M' + y0 + ',' + x0 +
                  'C' + y0 + ',' + x0 +
                  ' ' + y0 + ',' + x0 +
                  ' ' + y0 + ',' + x0)
              .style('stroke-width', Math.pow(d.data.prob, 0.5) * 50)
              .style('fill', 'none')
              .style('opacity', Math.pow(d.data.prob, 0.5) * 0.8);
          } else {
            link = d.data.link;
          }
          link
            .transition()
            .duration(duration)
            .attr('d', 'M' + d.y + ',' + d.x +
                'C' + (d.y + d.parent.y) / 2 + ',' + d.x +
                ' ' + (d.y + d.parent.y) / 2 + ',' + d.parent.x +
                ' ' + d.parent.y + ',' + d.parent.x);
        });

        nodes.descendants().forEach(function(d) {
          let node;
          if (d.data.node === null) {
            node = d.data.node = g.append('g')
              .attr('class', 'node')
              .attr('transform', 'translate(' + y0 + ',' + x0 + ')')
              .on('click', function() {
                click(d);
              });

            node.append('circle')
              .attr('cx', 0)
              .attr('cy', 0)
              .attr('r', 0)
              .style('stroke-width', 0)
              .style('opacity', 1e-6)
              .style('fill', 'steelblue');

            node.append('text')
              .attr('dy', '.35em')
              .attr('x', 4)
              .style('text-anchor', d.children ? 'end' : 'start')
              .style('fill-opacity', 1e-6)
              .text(d.data.name);
          } else {
            node = d.data.node;
          }
          
          node
            .transition()
            .duration(duration)
            .attr('transform', 'translate(' + d.y + ',' + d.x + ')');

          node.select('circle')
            .attr('r', 5 + 30 * Math.pow(d.data.prob, 0.3))
            .style('opacity', Math.pow(d.data.prob, 0.5) * 0.7);

          node.select('text')
            .text(d.data.name)
            .style('fill-opacity', 1);
        });

        nodes.descendants().forEach(d => {
          d.data.x = d.x;
          d.data.y = d.y;
        });
      }

      function close(d, removed = []) {
        if (d.children) {
          d._children = d.children;
          d.children = null;
          d._children.forEach(e => {
            removed.push(e);
            close(e, removed);
          });
        }
        return removed;
      }

      function click(d) {
        console.log(d.data);
        let removed = [];
        if (d.data.children) {
          removed = close(d.data);
        } else {
          d.data.children = d.data._children;
          d.data._children = null;
        }
        update(d.data.x, d.data.y, removed);
      }
    }
  }

</script>
<!-- Add 'scoped' attribute to limit CSS to this component only -->
<style scoped>


</style>
