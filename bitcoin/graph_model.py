import graph_tool
#import graph_tool.all as gt
import numpy as np
g = graph_tool.Graph(directed=True)
v = g.add_vertex(n=444075)
vertices = []

for i in v:
    vertices.append(i)

import pandas as pd
import scipy.sparse
train_data = '/home/junyic/Work/Courses/4th_year/DataSci/project3/txTripletsCounts.txt'

btc_train_pd = pd.read_csv(train_data, header=None, sep = ' ')
btc_train_np = np.array(btc_train_pd, dtype=np.int32)

prob = np.ones_like(btc_train_np[:,2])

btc_train_sm = scipy.sparse.coo_matrix((prob, (btc_train_np[:,0], btc_train_np[:,1])), dtype=np.int32, shape=(444075, 444075))

for i in range(0, btc_train_np.shape[0]):
    i_indx = btc_train_np[i,0]
    j_indx = btc_train_np[i,1]
    g.add_edge(vertices[i_indx], vertices[j_indx])

import graph_tool.centrality
prank = graph_tool.centrality.pagerank(g)

prank_np = np.zeros(444075)
for i in range(0, 444075):
    prank_np[i] = prank[vertices[i]]

##
component_num = 2
valid_index = np.logical_and(num_outbounds>0, V[component_num,:]>0)
r = pearsonr(np.log(num_outbounds[valid_index]), np.log(V[component_num,valid_index]))
rand_indx = np.random.randint(0, 444075, 2000)
plt.scatter(np.log(num_outbounds[rand_indx]), np.log(V[1,rand_indx]))
plt.show()
    
## starts from one nodes and draw a tree with it to be root
"""np.
outbounds = {}
for i in range(0, btc_train_np.shape[0]):
    current_k = btc_train_np[i,0]
    if outbounds.has_key(current_k):
        outbounds[current_k].append(btc_train_np[i,1])
    else:
        outbounds[current_k] = [btc_train_np[i,1]]
        
gtree = graph_tool.Graph(directed=True)
v_root = gtree.add_vertex()

root_indx = 3
indx_to_vertex = {root_indx: v_root}

max_levels = 20
def add_vertices(root, current_level):
    if current_level > max_level:
        return
"""


    
## pick the 5000 top ranked nodes and draw the graph
prank_np_indx = np.zeros((444075, 2))
prank_np_indx[:,0] = prank_np
prank_np_indx[:,1] = np.arange(0, 444075)

prank_np_indx = prank_np_indx[(-prank_np_indx[:,0]).argsort()]

num_top_nodes = 1000
gtop = graph_tool.Graph(directed=True)
vtop = gtop.add_vertex(n=num_top_nodes)
top_nodes_indx = np.array(prank_np_indx[:num_top_nodes, 1], dtype=int)
pr = prank_np_indx[:num_top_nodes,0]

text_to_disp = {}
for i in range(0, 10):
    text_to_disp[vertices[top_nodes_indx[i]]] = str(top_nodes_indx[i])
               
##
import matplotlib.cm
vtops = [vertices[i] for i in top_nodes_indx]
v_to_delete = list(set(vertices) - set(vtops))
for i, v in enumerate(reversed(sorted(v_to_delete))):
    g.remove_vertex(v, fast=True)
    print i

gt.graph_draw(g, vertex_fill_color=prank,
               vertex_size=gt.prop_to_size(prank, mi=2, ma=15),
               vorder=prank, vcmap=matplotlib.cm.gist_heat,
               output="pr_top1000b.pdf", gamma=0.5)
               
##  tree graph
pos_tree = gt.radial_tree_layout(g, g.vertex(3))
gt.graph_draw(g,
    pos=pos_tree, output="radial.png", output_size=(2000, 1600),
    vertex_fill_color=prank, vertex_size=gt.prop_to_size(prank, mi=10, ma=50))

## degree
deg = g.degree_property_map("in")
deg.a = 2 * (sqrt(deg.a) * 0.5 + 0.4)
ebet = gt.betweenness(g)[1]
gt.graphviz_draw(g, vcolor=deg, vorder=deg, elen=10,
                 ecolor=ebet, eorder=ebet, output="degree.png")

## sfdp
sfdp_pos = gt.sfdp_layout(g)
gt.graph_draw(g, sfdp_pos, output_size=(2000, 1600), vertex_color=[1,1,1,0],
           vertex_fill_color=prank, vertex_size=1, edge_pen_width=1.2,
           vcmap=matplotlib.cm.gist_heat_r, output="sfdp.png")