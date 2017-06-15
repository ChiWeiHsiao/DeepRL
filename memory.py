import numpy as np

np.random.seed(1)
class _SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity    # for all priority values
        self.tree = np.zeros(2*capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)    # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root


class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    abs_err_upper = 1.0              # clipped abs error
    def __init__(
        self,
        capacity,
        enable_pri = False,
        epsilon = 0.01,              # small amount to avoid zero priority
        alpha = 0.6,                 # [0~1] convert the importance of TD error to priority
        beta = 0.4,                  # importance-sampling, from initial value increasing to 1
        beta_increment_per_sampling = 0.001
    ):
        self.len = 0
        self.enable_pri = enable_pri
        self.capacity = capacity
        self.tree = _SumTree(capacity)
        if self.enable_pri:
            self.epsilon = epsilon
            self.alpha = alpha
            self.beta = beta
            self.beta_increment_per_sampling = beta_increment_per_sampling

    def store(self, transition):
        if self.len < self.capacity:
            self.len = self.len + 1

        print('Store', self.len)

        if self.enable_pri:
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p < self.abs_err_upper:
                max_p = self.abs_err_upper
            self.tree.add_new_priority(max_p, transition)   # set the max p for new p
        else:
            self.tree.data[self.tree.data_pointer] = transition
            self.tree.data_pointer = (self.tree.data_pointer + 1) % self.capacity

    def sample(self, n):
        print('Sample')
        if self.enable_pri:
            batch_idx, batch_memory, ISWeights = [], [], []
            segment = self.tree.root_priority / n
            self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
            maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
            for i in range(n):
                a = segment * i
                b = segment * (i + 1)
                for _ in range(10):
                    lower_bound = np.random.uniform(a, b)
                    idx, p, data = self.tree.get_leaf(lower_bound)
                    prob = p / self.tree.root_priority
                    if self.tree.capacity * prob < self.abs_err_upper:
                        continue
                    ISWeights.append(self.tree.capacity * prob)
                    batch_idx.append(idx)
                    batch_memory.append(data)
                    break

            ISWeights = np.vstack(ISWeights)
            ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
            return batch_memory, batch_idx, ISWeights
        else:
            idx = np.random.choice(self.len, n)
            return np.vstack(self.tree.data[idx]), idx, np.ones(n)

    def update(self, idx, error):
        if self.enable_pri:
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

    def __len__(self):
        return self.len
