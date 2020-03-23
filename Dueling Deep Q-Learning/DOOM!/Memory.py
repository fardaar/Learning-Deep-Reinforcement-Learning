from collections import deque
import numpy as np
import environment
import random
import stack_frames
import copy


class Node:
    def __init__(self, left, right, is_leaf=False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.id = idx
        if not self.is_leaf:
            self.value = self.right.value + self.left.value
            self.left.parent = self
            self.right.parent = self
        self.parent = None

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, True, idx)
        leaf.value = value
        return leaf


def create_sumtree(buffer, a, sum_probs):
    nodes = []
    for i in list(buffer.keys()):
        nodes.append(Node.create_leaf(value=(buffer[i].td_error ** a) / sum_probs, idx=i))
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes


def retrieve(tree: Node, probs):
    batch = []
    for prob in probs:
        node = tree
        while not node.is_leaf:
            if prob < node.left.value and not node.is_leaf:
                node = node.left
            elif prob > node.left.value and not node.is_leaf:
                prob = prob - node.right.value
                node = node.right
            elif node.is_leaf:
                node = node
        batch.append(node.id)
    return batch


class Experience:
    def __init__(self, state, action, reward, next_state, done, id, td_error):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.td_error = td_error
        self.id = id


class Memory:

    def __init__(self, a, maximum_length=10000):
        self.buffer = {}
        self.counter = 0
        self.sum_probs = 0
        self.a = a
        self.maxlen = maximum_length

    def add(self, state, action, reward, next_state, done, td_error):
        self.counter += 1
        if len(self.buffer.keys()) + 1 < self.maxlen:
            self.buffer[self.counter] = Experience(state=state, action=action, reward=reward,
                                                   next_state=next_state, done=done, id=self.counter,
                                                   td_error=td_error)
            self.sum_probs += td_error ** self.a
        else:
            rand_idx = random.choice(list(self.buffer.keys()))
            self.sum_probs -= self.buffer[rand_idx].td_error ** self.a
            del self.buffer[rand_idx]
            self.buffer[self.counter] = Experience(state=state, action=action, reward=reward, next_state=next_state,
                                                   done=done, id=self.counter, td_error=td_error)
            self.sum_probs += td_error ** self.a

    def sample(self, batch_size):
        tree, leafs = create_sumtree(self.buffer, self.a, self.sum_probs)
        batch_probs = [random.uniform(0, tree.value) for _ in range(batch_size)]
        batch = retrieve(tree, batch_probs)
        sample = [self.buffer[i] for i in batch]
        return sample

