import numpy as np
import matplotlib.pyplot as plt


class GraphMaxPooling:
    """
    This class performs max pooling (after storing multiple coarsening) on a
    graph given by its weight matrix W.
    """

    def __init__(self, W, k):
        """
        W: initial graph weight matrix.
        k: number of time to coarsen
        """
        self.W = W
        self.k = k
        self.new_order = None  # new order for indices
        self.W_s = []  # list with weight matrix for each version of the graph

        # perform the coarsening
        self.W_s, self.new_order = GraphMaxPooling.coarsening(W, self.k)

    def one_max_pooling(self, x, k):
        """
        :param x: 1d signal on the intial graph W to max-pool
        :param k: number of coarsening
        :return: max-pooled signal on the k-th graph (including virtual nodes)
        """
        # we need to add the virtual nodes
        x_ = np.zeros(self.new_order.shape)
        x_[0:784] = x
        x_ = x_[self.new_order]

        return GraphMaxPooling.repeat(
            GraphMaxPooling.max_pooling_1d_by_two, k)(x_)

    def max_pooling(self, x):
        """
        Returns maxpooled versions of the signal for all graphs (i.e. for
        every scale from biggest to smallest graph)
        :param x: signal on initial graph
        :return: list of pooled signals (including added virtual nodes)
        """
        # we need to add the virtual nodes
        x_ = np.zeros(self.new_order.shape)
        x_[0:784] = x
        x_ = x_[self.new_order]

        to_return = []
        pooled = x_
        to_return.append(pooled)
        for k in range(self.k):
            pooled = GraphMaxPooling.max_pooling_1d_by_two(pooled)
            to_return.append(pooled)

        return to_return

    def pool_and_project_back(self, x):
        """Given a signal 1d x performs pooling according to the coarsening and
        then project them back to the initial space to vizualize the pooling
        operations in the initial space
        returns: list with projected back signals at all levels
        """
        # get pooled signal
        x_ = np.zeros(self.new_order.shape)
        x_[0:x.size] = x
        x_ = x_[self.new_order]

        to_return = [x]  # we include the original signal
        pooled = x_
        for k in range(self.k):
            pooled = GraphMaxPooling.max_pooling_1d_by_two(pooled)
            projected_signal = np.zeros(x.shape)  # reconvert at first level
            # now operate the projection
            for i, ancestors in enumerate(
                    self.new_order.reshape((-1, 2 ** (k+1)))):
                projected_signal[ancestors[ancestors < x.shape]] = pooled[i]
            to_return.append(projected_signal)
        return to_return

    @staticmethod
    def repeat(f, n):
        """ To repeat a function"""
        if n == 0:
            return (lambda x: x)
        return (lambda x: f(GraphMaxPooling.repeat(f, n - 1)(x)))

    @staticmethod
    def max_pooling_1d_by_two(t):
        """ Return half the array with the max of side by side coeff"""
        return np.max(t.reshape((-1, 2)), 1)

    def get_laplacians(self, how='unn'):
        """
        how: 'unn' for combinatorial laplacians (L = D - W) or 'norm' for
        normalized laplacians (L = I - D^-0.5 W D ^-0.5)
        Returns a list with the Laplacian of each graph
        """
        if how=='unn':
            return [np.diag(np.sum(W, 0)) - W for W in self.W_s]
        elif how=='norm':
            to_return = []
            for W in self.W_s:
                D_inv_sqrt = np.diag(np.sqrt(1/np.sum(W, 0)))
                to_return.append(np.eye(W.shape[0]) - D_inv_sqrt*W*D_inv_sqrt)
            return to_return
        else:
            raise ValueError("how parameter must be 'unn' or 'norm'")

    @staticmethod
    def graclus_pairs(W, verbose=False):
        """
        Finds which nodes to pair together
        returns a symetric matrix with a 1 at each pair (i,j) to put together
        """
        n = W.shape[0]
        assigned = np.zeros(n)  # 0 => not assigned, 1 => assigned
        d = np.sum(W, 0)  # node degrees
        pairs = np.zeros((n, n))  # matrix with all pairs of matched nodes (1 if paired, O o.w.)

        while not np.all(assigned):
            node = np.random.choice(np.where(1 - assigned)[0])
            cut = W[:, node] * (1 / d[node] + 1 / d)
            assigned[node] = 1  # we don't want to assign the node with itself!

            potential_matches = np.where(1 - assigned)[
                0]  # index of potential matches

            if np.all(cut[potential_matches] == 0):
                # in this case the node has no neighbors
                if verbose:
                    print("{} has no match".format(node))
                pairs[node, node] = 1  # match with itself
            elif not np.all(assigned):

                match = potential_matches[np.argmax(cut[potential_matches])]
                assigned[match] = 1
                pairs[node, match] = 1
                pairs[match, node] = 1
                if verbose:
                    print("{} was matched with {}".format(node, match))
            else:
                if verbose:
                    print("{} has no match".format(node))
                pairs[node, node] = 1  # match with itself

        return pairs

    @staticmethod
    def one_coarsening(W):
        """Performs one coarsening, returns the new weight matrix and map_1_to_2
        map_1_to_2: vector of size W.shape[0], map_1_to_2[i] gives the index of
        the node to which i is mapped.
        """
        pairs = GraphMaxPooling.graclus_pairs(W)

        map_1_to_2 = np.zeros(pairs.shape[0], dtype=int)
        current_index = 0  # smallest index still available in the new graph 2
        for i, j in zip(*np.where(np.triu(pairs) == 1)):
            # we check and if i and j are grouped together we assign them the
            # two next indices
            # except if i==j then we give node j current_index and keep current_
            # index+1 for an virtual node
            if pairs[i, j] == 1:
                map_1_to_2[i] = current_index
                map_1_to_2[j] = current_index
                current_index += 1

        n_coarsened = max(map_1_to_2) + 1
        W_coarsened = np.zeros((n_coarsened, n_coarsened))
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                # we are looking at edge (i,j), we add its weights to the
                # corresponding nodes in the coarsened graph
                if W[i, j] != 0:
                    i_new_node = map_1_to_2[i]  # where i was sent in graph 2
                    j_new_node = map_1_to_2[j]  # where j was sent in graph 2
                    # FOR NOW WE DON't KEEP WEIGHTS ON THE DIAGONAL
                    if i_new_node != j_new_node:
                        W_coarsened[i_new_node, j_new_node] += W[i, j]

        return (W_coarsened, map_1_to_2)

    @staticmethod
    def add_all_virtuals_nodes(list_all_mappings):
        """ Add virtual nodes toat the end of all mappings. all_mappings
        is a list of arrays"""
        all_mappings = list_all_mappings.copy()
        for i in range(len(all_mappings) - 1, -1, -1):
            # the trick is to think of propagating the virtual nodes
            # e.g. if a virtual nodes is create for all_mappings[i] then
            # two virtual nodes
            # pointing to it must be added to all_mappings[i-1]
            mapping_plus_virtuals = all_mappings[i].copy()
            # now we look for loners, but each time we find one, we don't
            # forget to add 2 virtual nodes
            # to all_mappings[i-1]
            for node in range(max(all_mappings[i]) + 1):
                if sum(all_mappings[i] == node) < 2:
                    mapping_plus_virtuals = np.append(mapping_plus_virtuals,
                                                      node)
                    # don't forget to add two virtual nodes to the previous
                    # mapping pointing on this node
                    # and 4 to the anteprevious...
                    # and 8 to the one before...
                    # It is a bit complicated with the indices and all that so
                    # I don't detail... sorry
                    to_add_above_twice = [len(mapping_plus_virtuals) - 1]
                    for k in range(i - 1, -1, -1):
                        first_node_added = len(all_mappings[k])
                        all_mappings[k] = np.append(all_mappings[k], np.repeat(
                            to_add_above_twice, 2))
                        last_node_added = len(all_mappings[k]) - 1
                        to_add_above_twice = list(
                            range(first_node_added, last_node_added + 1))
                        # if i >0:
                        #    node_just_added = len(mapping_plus_virtuals) - 1
                        #    all_mappings[i-1] = np.append(all_mappings[i-1],
                        # [node_just_added]*2)
            all_mappings[i] = mapping_plus_virtuals.copy()
        # return list of augmented mappings
        return all_mappings

    @staticmethod
    def reorder_mapping(coarsened_indices, to_coarsen_indices):
        """
        Given a mapping (including virtual nodes) it outputs how to reorder
        the nodes for pairs to be side by side
        INPUTS
            coarsened_indices: nodes ordering for previous graph (smaller)
            to_coarsen_indices: mapping giving ...
        RETURNS:
            order for this graph
        """
        # Must be performed before with add_all_virtuals_nodes
        # to_coarsen_indices=Coarsening.add_virtuals_nodes(to_coarsen_indices)
        to_return = np.array([], dtype=int)
        for i in coarsened_indices:
            to_return = np.append(to_return,
                                  np.where(to_coarsen_indices == i)[0])
        return to_return

    @staticmethod
    def get_all_mappings(W, n):
        """ Coarsen n times and store all mappings for one graph to the next
        one in a list"""
        current_W = W.copy()
        all_mappings = []  # to keep all mapping arrays
        for i in range(n):
            current_W, mapping = GraphMaxPooling.one_coarsening(current_W)
            all_mappings.append(mapping)
        return all_mappings

    @staticmethod
    def find_first_graph_reindexing(W, n):
        """W initial weight matrix, n number of times to coarsen"""

        # first we apply one_coarsening n times and keep all weights matrices
        # and maps
        all_mappings = GraphMaxPooling.get_all_mappings(W, n)
        all_mappings = GraphMaxPooling.add_all_virtuals_nodes(all_mappings)

        # now we get back from the smallest graph to the intial one and
        # propagate the indices
        coarsened_indices = np.arange(max(all_mappings[-1]) + 1)
        for i in range(1, len(all_mappings) + 1):
            temp = coarsened_indices
            coarsened_indices = GraphMaxPooling.reorder_mapping(
                coarsened_indices, all_mappings[-i]
            )
            to_coarsen_indices = temp
        # print final ordering
        return coarsened_indices

    @staticmethod
    def merge_2_by_2(new_W):
        """
        Given a matrix of weights (including virtual nodes),
        it merges them 2 by 2 (coarsening) and returns the matrix of
        weights for the coarsend graph (half the number of nodes)"""
        n = new_W.shape[0]
        if n % 2 != 0:
            raise ValueError('Uneven number of nodes')
        W_coarsened = np.zeros((n // 2, n // 2))
        for i in range(n):
            for j in range(n):
                # we are looking at edge (i,j), we add its weights
                # to the corresponding nodes in the coarsened graph
                if new_W[i, j] != 0:
                    i_new_node = i // 2  # where i was sent in coarsened graph
                    j_new_node = j // 2  # where j was sent in coarsened graph
                    # FOR NOW WE DON't KEEP WEIGHTS ON THE DIAGONAL
                    if i_new_node != j_new_node:
                        W_coarsened[i_new_node, j_new_node] += new_W[i, j]
        return W_coarsened

    @staticmethod
    def coarsening(W, k):
        """
        Main function to use
        Returns: list of W's, ordering
        """
        # find how to reorder nodes
        new_order = GraphMaxPooling.find_first_graph_reindexing(W, k)

        n_init = W.shape[0]  # initial number of nodes
        n_final = max(new_order) + 1  # final number of nodes
        new_W = np.zeros((n_final, n_final))
        # we add all virtual nodes and reorder W
        new_W[0:n_init, 0:n_init] = W
        new_W = new_W[:, new_order][new_order, :]

        # retrieve the n (reordered inluding virtual nodes) weight matrices
        weight_matrices = [new_W]
        for i in range(k):
            new_W = GraphMaxPooling.merge_2_by_2(new_W)
            weight_matrices.append(new_W)

        return weight_matrices, new_order

    ###########################################################################
    # function a bit too specific perhaps but still nice for images and grids
    ###########################################################################
    @staticmethod
    def generate_grid(h, w):
        """
        Generate the weight matrix for a simple grid connecting neighbors
        :param h: height
        :param w: width
        :return: W matrix (h*w, h*w)
        """

        def index_to_pixels(i, h, w):
            """i: index, h: image height, w: image width
            returns: pixel height, pixel width
            """
            return (i // h, i % h)

        def pixel_to_index(x, y, h, w):
            """x: pixel height, y: pixel width, h: image height,
            w: image width"""
            return x * h + y

        def valid_pixel(x, y, h, w):
            return (0 <= x) and (x < h) and (0 <= y) and (y < w)

        def find_neighbors(i, h, w):
            """i: index, h: image height, w: image width
            returns: list of neighbors indices"""
            x, y = index_to_pixels(i, h, w)
            neighbors = [pixel_to_index(x + eps_x, y + eps_y, h, w) for
                         eps_x, eps_y in [(0, -1), (0, 1), (1, 0), (-1, 0)] if
                         valid_pixel(x + eps_x, y + eps_y, h, w)]
            return neighbors

        W = np.zeros((h * w, h * w))  # empty weight matrix
        # loop and complete the weight matrix
        for i in range(h * w):
            for j in find_neighbors(i, h, w):
                W[i, j] = 1
        return W

    def plot_coarsened_W_s(self, figsize=(13, 5)):
        f, axarr = plt.subplots(1, self.k+1, figsize=figsize)
        for i in range(self.k+1):
            axarr[i].imshow(self.W_s[i], cmap='Greys', interpolation='none')
            axarr[i].axis('off')

    def plot_pooled_images(self, x, height=None, width=None, figsize=(13, 5)):
        """
        Plotted projected pooled signals (in the intial space), for an image
        of size width*height
        :param x:
        :param height: image height, if None => squared img
        :param width: image width, if None => squared img
        :return: None
        """
        if height is None or width is None:
            img_shape = (
                int(np.sqrt(x.shape[0])),
                int(np.sqrt(x.shape[0]))
            )
        else:
            img_shape = (height, width)
        # get the list of pooled signals
        pooled = self.pool_and_project_back(x)
        f, axarr = plt.subplots(1, self.k+1, figsize=figsize)
        for i in range(self.k+1):
            axarr[i].imshow(pooled[i].reshape(img_shape),
                            cmap='Greys', interpolation='none')
            axarr[i].axis('off')


if __name__ == "main":
    import scipy.misc
    # small demo
    mnist = scipy.misc.imread("../drafts/mnist.png", flatten=True) / 255
    # reshape the signal to a vector, r
    x = mnist.reshape((-1,))
    # take a grid as graph
    W = GraphMaxPooling.generate_grid(*mnist.shape)

    # instantiate a pooling object to perform 4 coarsening + maxpooling
    foo_pooling = GraphMaxPooling(W, 4)
    # plot all the W's from the coarsened graphs
    foo_pooling.plot_coarsened_W_s()
    # plot the signal reprojected on the initial graph
    foo_pooling.plot_pooled_images(x)