import matplotlib.pyplot as plt
import cv2
import os
import time
import random
import operator
from PIL import Image
from env.movo_env import MovoEnv
from env.base.default_arena import DefaultArena
import heapq
import networkx as nx
import numpy as np
from matplotlib import collections as mc

class RobotArmEnv:
    num_ball = [6, 10, 10]
    begin = np.array([0.6, -0.8, 0.])
    end = np.array([1.2, 0.8, 1.])

    def __init__(self, obstacles=None, use_viewer=True, type=0):
        self.mesh = []
        self.robot_mesh = None
        if type == 1:
            self.num_ball = [6, 8, 5]
            self.begin = np.array([0.6, -0.8, 0.5])
            self.end = np.array([1.2, 0.8, 1.])
        if obstacles is not None:
            import open3d as o3d
            num_ob = len(obstacles)
            self.pos = obstacles[:, 0:3]
            self.rad = obstacles[:, 3]

            for i in range(num_ob):
                ball = o3d.geometry.TriangleMesh.create_sphere(self.rad[i])
                ball.translate(self.pos[i])
                ball.paint_uniform_color([1,0,0])
                self.mesh.append(ball)
            temp = self.mesh[0]
            for i in range(1, num_ob):
                temp = temp + self.mesh[i]
            temp = temp.compute_vertex_normals()
            o3d.io.write_triangle_mesh('/tmp/ball.stl', temp)
            self.arena = DefaultArena()
            self.arena.add_fixed_object(stl_path=os.path.join('/tmp', 'ball.stl'),
                                        xyz=[1, 0, 1], quaternion=[1, 0, 0, 0],
                                        rgba=[1, 0, 0, 0.5])
        else:
            num_ball = self.num_ball
            begin = self.begin
            end = self.end

            self.arena = DefaultArena()
            self.block_size = [ (end[i] - begin[i]) / (self.num_ball[i]-1) * 0.45 for i in range(3)]
            
            self.xyz = []

            for i in range(num_ball[0]):
                x = (end[0] - begin[0]) * i/(self.num_ball[0]-1) + begin[0]
                for j in range(num_ball[1]):
                    y = (end[1] - begin[1]) * j/(self.num_ball[1]-1) + begin[1]
                    for k in range(num_ball[2]):
                        z = (end[2] - begin[2]) * k/(self.num_ball[2]-1) + begin[2]
                        self.xyz.append([x, y, z])
                        self.arena.add_block((x, y, z), size=self.block_size, name = f'{i} {j} {k}', rgba=(1, 0, 1,1))
            
            self.xyz = np.array(self.xyz)
            self.num_obj = len(self.xyz)

        self.env = MovoEnv(arena=self.arena, visualize_collision=False,
                           checker_board_config=None, use_viewer=use_viewer)
        self.env.reset()


    def set_full(self):
        return self.set_env(self.xyz.copy()[:,0] * 0 + 1)

    def set_empty(self):
        return self.set_env(self.xyz.copy()[:,0] * 0)
    
    def set_env(self, map, noise=None):
        xyz = self.xyz.copy()
        map = map.reshape(-1)

        xyz[:, 0] += (1-map) * 2 # move the obstacles to somewhere unknown
        self.set_obstacles(xyz)

    def set_obstacles(self, xyz):
        self.env.sim.model.body_pos[1:1+len(xyz)] = xyz
        self.env.sim.forward()

    def get_obstacles(self):
        return self.env.sim.model.body_pos[1:1+len(self.xyz)].copy() # x, y, z

    def step(self, action):
        assert len(action) == 7
        collision_num = self.env.step(action)
        contact = self.env.sim.data.contact
        object_collision = 0
        self_collision = 0
        gap = 100000
        self.collision_objects = set()
        for j in range(collision_num):
            a = contact[j].geom1
            b = contact[j].geom2
            if a>=1 and a<self.num_obj + 1:
                self.collision_objects.add(a-1)
            if b>=1 and b<self.num_obj + 1:
                self.collision_objects.add(b-1)

            if a>=self.num_obj + 1 and b >= self.num_obj + 1:
                if a != 4 + self.num_obj - 1 or (b != 17 + self.num_obj - 1 and b != 18 + self.num_obj - 1):
                    self_collision = 1
        return len(self.collision_objects)>0, self_collision

    def get_collision_map(self, action):
        _, a = self.step(action)
        map = self.xyz[:, 0] * 0
        for i in self.collision_objects:
            map[i] = 1
        return np.append(map, a)

    def get_joint_xyz(self, action):
        self.step(action)
        return self.env.sim.data.body_xpos[608:620].copy()

    def remove_obstacles(self, map, action):
        self.step(action)

        map = map.reshape(-1).copy()
        for i in self.collision_objects:
            map[i] = 0
        return map

    def sample_valid_state(self, default=None, times=None):
        not_found = True
        while not_found:
            if times is not None: 
                times -= 1
                if times <= 0:
                    return default
            action = self.sample_action()
            ob_collision, self_collision = self.step(action)
            not_found = ob_collision or self_collision
        return action

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self, mode='rgb_array', width=640, height=480, cameras=['default']):
        if mode == 'human':
            self.env.render()
        elif mode == 'rgb_array':
            return self.env.current_obs_numpy_array(cameras=cameras, height=height, width=width)
        else:
            raise NotImplementedError


# Calculate the edge
def calc_edge_weight(graph, pose1, pose2):
    return np.linalg.norm(graph.nodes[pose1]['pose'] - graph.nodes[pose2]['pose'])    

# Rondomly create the poses
def create_graph(env, n, k):
    # Create the graph
    N = n
    K = k
    
    # Randomly sample the poses
    node = []
    for i in range(N):
        action = env.sample_action()
        node.append(action)
    node = np.array(node)
    
    # Knn to produce the edges
    edge = []
    for i in range(N):
        distances = []
        for j in range(n):
            if i == j: 
                continue
            dist = np.linalg.norm(node[i] - node[j])
            distances.append((dist, j))
        distances.sort(key = operator.itemgetter(0))
        for k in range(K):
            edge.append(distances[k][1])
    edge = np.array(edge)
    #print(edge.shape)
    edge = edge.reshape(N, K)
    
    # Graph initialization
    graph = nx.Graph()
    
    # Adding nodes:
    for i in range(n):
        graph.add_node(i, pose = node[i])
        
    # Adding edges:
    for i in range(N):
        for k in range(K):
            graph.add_edge(i, edge[i][k], 
            weight = calc_edge_weight(graph, i, edge[i][k]))
            
    return graph

# Check the validity of the pose
def is_pose_free(env, pose):
    env.step(pose)
    if env.collision_objects == set():
        return True
    return False
    
# Check the validity of 2 poses' edge
def is_edge_free(env, pose_1, pose_2, edge_discretization = 20):
    # edge_discretization : divide the edge into x steps
    diff = pose_2 - pose_1
    step = diff/edge_discretization
    for i in range(edge_discretization + 1):
        new_pose = pose_1 + diff * i
        if(not is_pose_free(env, new_pose)):
            return False
    return True

# Get 2 valid start and goal state:
def get_valid_start_goal(env, graph):
    start_pose = random.choice(list(graph.nodes()))
    goal_pose = random.choice(list(graph.nodes()))
    
    start = graph.nodes[start_pose]['pose']
    goal = graph.nodes[goal_pose]['pose']
    
    # If the start and goal can directly connect, then 
    # the case is too simple
    while is_edge_free(env, start, goal, edge_discretization = 50) or not \
        (is_pose_free(env, start) and is_pose_free(env, goal)):
        start_pose = random.choice(list(graph.nodes()))
        goal_pose = random.choice(list(graph.nodes()))
    
        start = graph.nodes[start_pose]['pose']
        goal = graph.nodes[goal_pose]['pose']
    return start_pose, goal_pose

def get_heuristic(graph, v, goal_v):
    return np.linalg.norm(graph.nodes[v]['pose'] - graph.nodes[goal_v]['pose'])

# Astar algorithm to create the path:
def astar(env, graph, start_pose, goal_pose, h_weight = 1):
    queue = []

    # elements for the heapq: (queue, (cost, dist, current point)
    heapq.heappush(queue, (0, 0, start_pose))
    nodes = dict()
    nodes[start_pose] = (0, [])

    count = 0
    while len(queue):
        heu, dis, cur = heapq.heappop(queue)

        if dis > nodes[cur][0]:
            continue

        if cur == goal_pose:
            add_pose = goal_pose
            plan = []
            while add_pose != []:
                plan.append(add_pose)
                add_pose = nodes[add_pose][1]
            print(" count = ", count)
            print(" dis = ", dis)
            return np.array(plan[::-1]), dis
        
        next_cur = graph.neighbors(cur)

        for v in next_cur:
            dis_v = dis + graph[cur][v]['weight']

            if (v not in nodes) or nodes[v][0] > dis_v:
                count += 1
                cost_v = dis_v + h_weight * get_heuristic(graph, v, goal_pose)
                node1_pos = graph.nodes[v]['pose']
                node2_pos = graph.nodes[cur]['pose']

                lines = []
                #colors = []
                lines.append([node1_pos, node2_pos])
                if not is_edge_free(env = env, pose_1 = node1_pos, pose_2 = node2_pos):
                    #colors.append((1, 0, 0, 0.3))
                    #lc = mc.LineCollection(lines, colors = colors, linewidths = 1)
                    continue
                #colors.append((0, 1, 0, 0.3))
                heapq.heappush(queue, (cost_v, dis_v, v))
                nodes[v] = (dis_v, cur)
            print(" count = ", count)

            return [], None

# Create the path:
def create_path(env, graph, start_pose, goal_pose):
    #start_pose, goal_pose = get_valid_start_goal(env, graph)
    path = astar(env, graph, start_pose, goal_pose)
    return path

# Numpy to string
def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

# String to numpy
def numpy_to_state(array):
    state = ""
    for i in range(len(array)):
        state += str(array[i]) + ""
    return state

def path_create(env, graph):
    start, goal = get_valid_start_goal(graph)
    return

if __name__ == '__main__':

    # Create the environment
    env = RobotArmEnv()

    dense_graph = create_graph(env = env, n = 1000, k = 20)
    sparse_graph = create_graph(env = env,n = 50, k = 5)

    path_nodes = []
    start_nodes = []
    goal_nodes = []
    env_nodes = []
    
    # number of path we want to sample
    num = 1
    for i in range(num):
        # Randomly choose which obstacles should be removed
        num_ball = np.array(env.num_ball)
        remove = np.random.rand(num_ball[0], num_ball[1], num_ball[2])
        # Only remain 15% obstacles
        remove[remove < 0.85] = 0
        remove[remove >= 0.85] = 1
        env.set_env(remove)

        # Randomly choose the start and goal
        start_pose, goal_pose = get_valid_start_goal(env = env, graph = dense_graph)
        # Utilize astar to search the path
        tmp_path = create_path(env, dense_graph, start_pose, goal_pose)
        # Append the path, start, goal
        tmp_path = np.array(tmp_path)
        goal_pose = np.array(goal_pose)
        start_pose = np.array(start_pose)

        path_nodes.append(tmp_path)
        goal_nodes.append(goal_pose)
        start_nodes.append(start_pose)
        env_nodes.append(remove)

    path_nodes = np.array(path_nodes)
    goal_nodes = np.array(goal_nodes)
    start_nodes = np.array(start_nodes)
    env_nodes = np.array(env_nodes)

    # Change the array into the string
    for i in list(dense_graph.nodes()):
        dense_graph.nodes[i]['pose'] = numpy_to_state(dense_graph.nodes[i]['pose'])
    for i in list(sparse_graph.nodes()):
        sparse_graph.nodes[i]['pose'] = numpy_to_state(sparse_graph.nodes[i]['pose'])
        
        


    # Store the data in server
    nx.write_gml(dense_graph, "/home/zhizuo/lego/Robot Arm/Dataset/dense_graph.graphml")
    nx.write_gml(sparse_graph, "/home/zhizuo/lego/Robot Arm/Dataset/sparse_graph.graphml")
    np.savetxt("/home/zhizuo/lego/Robot Arm/Dataset/path_nodes", path_nodes)
    np.savetxt("/home/zhizuo/lego/Robot Arm/Dataset/goal_nodes", goal_nodes)
    np.savetxt("/home/zhizuo/lego/Robot Arm/Dataset/start_nodes", start_nodes)
    np.savetxt("/home/zhizuo/lego/Robot Arm/Dataset/env_nodes", env_nodes)

