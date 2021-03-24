# mismatchtree.py

import numpy as np

def integerized(sequence):
    
    key_dict = sorted(set(sequence))
    int_seq = []
    for char in sequence:
        to_int = key_dict.index(char)
        int_seq.append(to_int)

    return int_seq

def preprocess(sequences, ignoreLower=True):
    
    upper_seq = []
    len_record = []
    for seq in sequences:
        if ignoreLower:
            seq = [x for x in seq if 'A' <= x <= 'Z']
        else:
            seq = seq.upper()
        upper_seq.append(integerized(seq))
        len_record.append(len(seq))

    length_used = min(len_record)
    post_seq = []
    for seq in upper_seq:
        seq = seq[:length_used]
        post_seq.append(seq)

    return post_seq

def normalize_kernel(kernel):

    nkernel = np.copy(kernel)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    for i in range(nkernel.shape[0]):
        for j in range(i + 1, nkernel.shape[0]):
            q = np.sqrt(nkernel[i, i] * nkernel[j, j])
            if q > 0:
                nkernel[i, j] /= q
                nkernel[j, i] = nkernel[i, j]

    np.fill_diagonal(nkernel, 1.)

    return nkernel
    
    
class MismatchTrie(object):
    def __init__(self, label=None, parent=None):
        self.label = label
        self.level = 0 
        self.children = {}
        self.full_label = ""
        self.kmers = {}
        self.parent = parent
        if not parent is None:
            parent.add_child(self)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def is_empty(self):
        return len(self.kmers) == 0

    def copy_kmers(self):
        return {index: np.array(substring_pointers) for index, substring_pointers in self.kmers.items()}

    def add_child(self, child):

        child.kmers = self.copy_kmers()

        child.level = self.level + 1

        child.full_label = '%s%s' % (self.full_label, child.label)

        self.children[child.label] = child

        child.parent = self

    def delete_child(self, child):
        label = child.label if isinstance(child, MismatchTrie) else child

        del self.children[label]

    def compute_kmers(self, training_data, k):

        for index in range(len(training_data)):
            self.kmers[index] = np.array([(offset, 0) for offset in range(len(training_data[index])-k+1)])

    def process_node(self, training_data, k, m):
        if self.is_root():
            self.compute_kmers(training_data, k)
        else:
            for index, substring_pointers in self.kmers.items():
                substring_pointers[..., 1] += (training_data[index][
                        substring_pointers[..., 0] + self.level - 1
                        ] != self.label)
                self.kmers[index] = np.delete(substring_pointers,
                                               np.nonzero(substring_pointers[..., 1] > m),
                                               axis=0)
            self.kmers = {index: substring_pointers for (
                    index, substring_pointers) in self.kmers.items(
                    ) if len(substring_pointers)}

        return not self.is_empty()

    def update_kernel(self, kernel):
        for i in self.kmers:
            for j in self.kmers:
                kernel[i, j] += len(self.kmers[i]) * len(self.kmers[j])


    def traverse(self, training_data, l, k, m, kernel=None, kernel_update_callback=None):
        if kernel is None:
            kernel = np.zeros((len(training_data), len(training_data)))
        
        n_surviving_kmers = 0

        go_ahead = self.process_node(training_data, k, m)

        if go_ahead:
            if k == 0:
                n_surviving_kmers += 1

                self.update_kernel(kernel)

            else:
                for j in range(l):
                    child = MismatchTrie(label=j, parent=self)

                    kernel, child_n_surviving_kmers, \
                        child_go_ahead = child.traverse(
                        training_data, l, k - 1, m, kernel=kernel)

                    if child.is_empty():
                        self.delete_child(child)

                    n_surviving_kmers += child_n_surviving_kmers if \
                        child_go_ahead else 0

        return kernel, n_surviving_kmers, go_ahead


class MismatchKernel(MismatchTrie):
    def __init__(self, l= 4, k=None, m=None, **kwargs):
        MismatchTrie.__init__(self, **kwargs)
        self.l = l
        self.k = k
        self.m = m

    def get_kernel(self, X, normalize = True, **kwargs):
        self.kernel, _, _ = self.traverse(X, self.l, self.k, self.m, **kwargs)
        
        if normalize:
            self.kernel = normalize_kernel(self.kernel)
            
        return self
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    