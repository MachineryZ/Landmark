import numpy as np
import networkx as nx 


def planning(env, path_nodes, start, goal):
    """
    env : environment parameter
    path_node : list of [x, y] represents the path nodes
    """
    # create the nodes
    G = nx.graph()
    G.add_node(0, loc = start)
    for i in len(path_nodes):
        G.add_node(i+1, loc = path_nodes[i])
    G.add_node(len(path_nodes) + 1, loc = path_nodes[i])

    # create the edges
    for i in range(len(path_nodes) + 2):
        for j in range(len(path_nodes) + 2):
            if collision_detection(env = env, graph = G , node1 = i, node2 =  j, step = 0.1):
                G.add_edge(i, j)
    
    # use bellman_ford algorithm to compute the distance
    


    


    return distance, node_list


def bellman_ford(graph):
    size = len(list(graph.nodes()))
    distance = np.zeros(size)
    for i in range(1, size):
        distance[i] = -1
    

    return distance, node_list


def collision_detection(env, graph, node1, node2, step = 20):
    loc1 = G.nodes()[node1]['loc']
    loc2 = G.nodes()[node2]['loc']
    delta_x = (loc2[0] - loc1[0])/step
    delta_y = (loc2[1] - loc1[1])/step
    dim_x = env.shape[0]
    dim_y = env.shape[1]
    for i in range(step+1):
        x = loc1[0] + delta_x * i 
        y = loc1[1] + delta_y * i 
        if env[int(x*dim_x)][int(y*dim_y)] == 0:
            return False
    return True


path_nodes = [[1,2], [3,4]]
G = nx.Graph()
for i in range(2):
    G.add_node(i, loc = path_nodes[i])
for i in range(2):
    print(G.nodes()[i]['loc'][])

