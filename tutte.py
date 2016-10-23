import numpy as np


# In[50]:
def tutte_barycentric_layout(graph, h):
    l = len(graph)
    W = np.zeros(shape=(l, l), dtype='float')
    F = np.array([[np.cos(j*2*np.pi / h), -np.sin(j*2*np.pi / h)] for j in range(h)])
    for i, neighbors in graph.iteritems():
        for j in graph.keys():
            if i == j:
                W[i, j] = len(neighbors)   
            else:
                W[i, j] = -1.0 if j in neighbors else 0.0
    B = np.matmul(W,np.pad(F, (0, l-h), 'constant'))
    return np.concatenate((F, np.linalg.solve(W[h:,h:], B[h:,0:2])))

# from itertools import chain
# edges = set(chain(*[[(i, j) if i <= j else (j, i) for j in neigbors] for i, neigbors in graph.items()]))


# # In[119]:

# plt.axis()
# for i, j in edges:
#     plt.plot(xs[[i, j]], ys[[i, j]], 'k')
# plt.show()


# # In[6]:

# def normalize(V):
#     norm = np.linalg.norm(V)
#     return V / norm if norm > 0 else V


# # In[7]:

# def normalize_all(V):
#     normalizer = np.linalg.norm(V, axis=1)
#     normalizer[normalizer == 0] = 1
#     return V / np.expand_dims(normalizer, 1)


# # In[8]:

# def generation(graph, vectors):
#     next_gen = {}
#     l = len(vectors)
#     for i in range(l):
#         neigbors = graph.get(i)
#         v = vectors[i, :]
#         n1 = vectors[neigbors[0], :]-v
#         n2 = vectors[neigbors[1], :]-v
#         n3 = vectors[neigbors[2], :]-v
#         #n1 = m2 - m1
#         #n2 = m3 - m3
#         #n3 = m1 - m3
#         next_vec = v# + normalize((np.cross(n1, n2) + np.cross(n2, n3) + np.cross(n3, n1)))/30
#         for j in range(l):
#             next_vec = next_vec + (vectors[j, :] - v) / 3
#         for j in neigbors:
#             next_vec = next_vec - (vectors[j, :] - v) / 10
#         next_gen[i] = next_vec
#     next_gen_mat = np.array([next_gen[i] for i in range(l)]).reshape((l,3))
#     next_gen_mat = normalize_all(next_gen_mat)
#     return next_gen_mat


# # In[9]:

# vectors = np.random.normal(size=(len(graph), 3))
# normals = normalize_all(np.random.normal(size=(len(graph), 3))) / 4


# # In[10]:

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cursor = vectors
# for i in range(100):
#     cursor = generation(graph, cursor)
# for (a, b) in edges:
#     plt.plot(cursor[[a,b],0],cursor[[a,b],1],cursor[[a,b],2])
# for i in range(normals.shape[0]):
#     src = cursor[i, :]
#     dest = cursor[i, :] + normals[i, :]
#     plt.plot([src[0], dest[0]],
#              [src[1], dest[1]],
#              [src[2], dest[2]])    


# # In[77]:

# (S,V,D) = np.linalg.svd(cursor)


# # In[78]:

# V


# # In[70]:

# V


# # In[71]:

# V[1:] - V[0:1]


# # In[82]:

# (S, V, D)


# # In[87]:

# cursor.mean(axis=0)


# # In[93]:

# plt.scatter([0, 1, 2], (V*1000).tolist())


# # In[95]:

# S.shape


# # In[96]:

# V[-1]


# # In[97]:

# V[0]


# # In[170]:

# np.array([[1, 2], [2, 3]]) / np.array([[2], [2]], dtype="float")


# # In[179]:

# get_ipython().magic(u'pinfo np.expand_dims')


# # In[ ]:



