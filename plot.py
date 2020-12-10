import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mayavi import mlab

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph_colormap='winter'
    bgcolor = (1, 1, 1)
    node_size=0.03
    edge_color=(0.8, 0.8, 0.8) 
    edge_size=0.002
    text_size=0.008
    text_color=(0, 0, 0)
    H=nx.Graph()

    # add edges
    # for node, edges in graph.items():
    #     for edge, val in edges.items():
    #         if val == 1:
    #             H.add_edge(node, edge)

    G=nx.convert_node_labels_to_integers(graph)

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])

    # scalar colors
    scalars=np.array(G.nodes())+5
    mlab.figure(1, bgcolor=bgcolor)
    mlab.clf()

    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=node_size,
                        scale_mode='none',
                        colormap=graph_colormap,
                        resolution=20)

    for i, (x, y, z) in enumerate(xyz):
        label = mlab.text(x, y, str(i), z=z,
                          width=text_size, name=str(i), color=text_color)
        label.property.shadow = True

    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=edge_color)

    mlab.show() # interactive window
    # nx.draw(gr, node_size=100, with_labels=False)
    # plt.show()
a=[0]*100
b=[0]*100
adj=np.zeros((100,100),dtype=int)
f=open("trace_0.5.txt","r")
c=0
l=f.readlines()
# for i in range(10):
for line in l[1:]:
    l1=line.strip()
    # l2=list(l1)
    l2=l1.split("\t")
    # print(l2)
    # and float(l2[1])<=(i+1)*10
    if int(l2[0])==2:
        # print(l2)
        adj[int(l2[2])-1,int(l2[3])-1]=1
        # adj[int(l2[3])-1,int(l2[2])-1]=1
    # elif float(l2[1])>(i+1)*10:
    #     show_graph_with_labels(adj)
    #     break
        c+=1
    if int(l2[0])==0:
        b[int(l2[2])-1]+=1
print(c)
for i in range(100):
    a[i]=np.sum(adj[i])

# print(a,b)

# plt.scatter(a,b)
# plt.xlabel("number of links")
# plt.ylabel("number of tweets+retweets")
# plt.show()


# show_graph_with_labels(adj)
