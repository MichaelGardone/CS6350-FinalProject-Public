from pyvis.network import Network

def drawGraph(graph, filename):
  net = Network(directed=True)
  net.from_nx(graph)
  net.show(filename)