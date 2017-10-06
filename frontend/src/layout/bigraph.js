class Node {
  constructor(selector) {
    this.selector = selector;
    this.edges = [];
  }
  render() {

  }
}

class Edge {
  constructor(node1, node2, weight) {
    this.nodes = [node1, node2];
    this.weight = weight;
  }
  render() {

  }
}

class BiGraph {
  constructor(nodes1, nodes2, weights, {
    EdgeType = Edge,
    NodeType = Node,
  }) {
    this.EdgeType = EdgeType;
    this.NodeType = NodeType;
    this.nodes1 = nodes1;
    this.nodes2 = nodes2;
    nodes1.forEach((node1, i) => {
      nodes2.forEach((node2, j) => {
        const edge = new EdgeType(node1, node2, weights[i][j]);
        node1.edges.push(edge);
        node2.push(edge);
      });
    });
  }
  render() {
    this.nodes1.forEa
  }
}

export default BiGgraph;
