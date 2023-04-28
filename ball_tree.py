import numpy as np


class Node:
    """
    a node of a tree that has a left and right child
    """
    def __init__(self, value=None, left_child=None, right_child=None,
                 sphere_radius=None):
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.sphere_radius = sphere_radius


class Tree:
    def __init__(self, original_row):
        self.origin_row = original_row
        self.root_node = None
        self.curr_parent = self.root_node

    def create_node(self, data, radius):
        """
        creates a new node.
        Args:
            data:the new node's value
            radius: distance from furthest point in sphere tothe mean/data
        """
        return Node(value=data, sphere_radius=radius)

    def insert_root(self, node, data, sphere_radius):
        """
        inserts a node as root tree.
        """
        # if tree is empty , return a root node
        if node is None:
            self.root_node = self.create_node(data=data, radius=sphere_radius)

    def insert_child(self, node, data, radius, parent=None):
        """
        inserts a child to a parent node.
        """
        parent = self.root_node if parent is None else parent
        # if no node was passed in, create one
        if node is None:
            node = self.create_node(data, radius)

        # if data is smaller than parent , insert it into left side
        if data < parent.value:
            # if the parent has no leaf as the left child
            if parent.left_child is None:
                parent.left_child = node
            else:
                parent = parent.left_child
                self.insert_child(node, data, radius, parent)
        elif data > parent.value:
            if parent.right_child is None:
                parent.right_child = node
            else:
                parent = parent.right_child
                self.insert_child(node, data, radius, parent)
        return node

    def insert_children(self, list_data, parent=None):
        """
        inserts a children to a parent node.
        """
        parent = self.root_node if parent is None else parent
        node_data = [item for item in list_data]
        nodes = {(mean, radius): self.create_node(mean, radius) for mean, radius in node_data}

        for item, node in nodes.items():
            # if data is smaller than parent , insert it into left side
            # item is a tuple of central tendency and radius
            if item[0] < parent.value:
                # ie if mean < parent.value
                if parent.left_child is None:
                    parent.left_child = node
                else:
                    new_parent = parent.left_child
                    self.insert_child(node, item[0], new_parent)
            elif item[0] > parent.value:
                if parent.right_child is None:
                    parent.right_child = node
                else:
                    new_parent = parent.right_child
                    self.insert_child(node, item[0], new_parent)
        self.root_node = parent
        return parent

    def check(self, value, curr_check_node, temp_best):
        """
        checks the nodes of the tree to find the node with the smalest distance from out new node
        """
        curr_value = curr_check_node.value
        diff = abs(value - curr_value)

        # temp_best = (temp best distance, value that gave best temp distance)
        if diff < temp_best[0]:
            temp_best[0] = diff
            temp_best[1] = curr_value
        if value > curr_value:
            # go right
            if curr_check_node.right_child:
                curr_check_node = curr_check_node.right_child
                return self.check(value=value, curr_check_node=curr_check_node, temp_best=temp_best)
            else:
                return temp_best

        elif value < curr_value:
            # go left
            if curr_check_node.left_child:
                curr_check_node = curr_check_node.left_child
                return self.check(value=value, curr_check_node=curr_check_node, temp_best=temp_best)
            else:
                return temp_best
        else:
            # stay
            return 'booo'

    def search(self, new_row, idx_dist_dict):
        """
        search for where (in tree) that the new point is closest to.
        """
        # calc distance from 1st furthest point
        dist = (np.sum((self.origin_row - new_row) ** 2)) ** 1 / 2
        # pass value gotten to be searched for in tree
        current_node = self.root_node
        difference = abs(dist - current_node.value)
        temp_best_pair = [difference, current_node.value]
        best_distance_tuple = self.check(value=dist, curr_check_node=current_node, temp_best=temp_best_pair)
        return best_distance_tuple

class BallTree:
    def distance_calc(self, p1, p2, dist=2):
        """
        Calculates the distances between two points in space and returns a scalar value
        Args:
            p1(float): point 1
            p2(float): point 2
            dist: norm of distance
        Returns:
            a scalar quantity(float)
        """
        # dist = 2
        distance_scalar = (np.absolute(np.sum((p1 - p2) ** dist))) ** 1 / dist
        return distance_scalar

    def furthest_neighbor(self, train_data, test_point):
        """
          Calculates nearest K neighbors.

          Args:
              train_data(numpy table): the table with the traindata
              test_point(numpy array): a row of values from the test data
          Returns:
              neighbors(list)
        """
        dist = 2

        # save the distances  from test point without index
        distances_sans_idx = [self.distance_calc(row, test_point, dist=2) for row in train_data]
        # save the distances  from test point with index
        distances_avc_idx = {row_idx: self.distance_calc(train_data[row_idx], test_point, dist=2) for row_idx in
                             range(len(train_data))}

        sorted_dists = np.argsort(distances_sans_idx)  # sorts the neighbors from lowest to highest

        furthest_dist_idx = np.argmax(distances_sans_idx)  # neighbours.append(k_indices[0])

        return furthest_dist_idx, distances_sans_idx, distances_avc_idx, sorted_dists

    def get_mean_nd_radii(self, x_samples, y_samples):
        """
          Gets the mean as midpoint and radii for the ball spheres so we can build the ball tree
          Args:
            x_samples: features of the sample data
            y_samples: labels of th esample data
          Returns:
            root_data, root_radius and list of mean radii tuples
        """
        # pick a point at random
        rand_idx = np.random.randint(x_samples.shape[0] - 1)

        random_point = x_samples[rand_idx]
        random_point_label = y_samples[rand_idx]

        # calculate the furthest distance from that point
        furthest_point_frm_rand_idx, _, _, _ = self.furthest_neighbor(x_samples, random_point)

        # calculate the furthest distance from the initial furthest distance(w)
        furthest_point_frm_furtst_point_idx, dists, dists_w_idx, sorted_dists_idx = self.furthest_neighbor(
            x_samples, x_samples[furthest_point_frm_rand_idx])
        dist_frm_furthst = dists[furthest_point_frm_furtst_point_idx]

        enumed_length = len(sorted_dists_idx)

        mid_point_idx = enumed_length // 2
        median_idx = sorted_dists_idx[mid_point_idx]

        # use the median as the value for the root of the tree
        root_data = dists_w_idx[median_idx]
        root_radius = dist_frm_furthst - root_data

        means_nd_radii = []

        left = np.array([dist for dist in dists if dist < root_data])
        right = np.array([dist for dist in dists if dist > root_data])

        queue = []
        queue.append(left)
        queue.append(right)

        while queue:
            # find what set of distances to evaluate next using bfs
            left = queue.pop(0)

            right = queue.pop(0)

            left_mean = np.sum(left) / left.shape[0]
            right_mean = np.sum(right) / right.shape[0]

            # get the furthest point from mean
            if left.any() and right.any():
                left_radius = np.max(left) - left_mean
                right_radius = np.max(right) - right_mean

                # add mean and radii to list
                means_nd_radii.append((left_mean, left_radius))
                means_nd_radii.append((right_mean, right_radius))

                # left of left
                left_of_left = np.array([dist for dist in left if dist < left_mean])
                queue.append(left_of_left)
                # right of left
                right_of_left = np.array([dist for dist in left if dist > left_mean])
                queue.append(right_of_left)
                # left of right
                left_of_right = np.array([dist for dist in right if dist < right_mean])
                queue.append(left_of_right)
                # right of right
                right_of_right = np.array([dist for dist in right if dist > right_mean])
                queue.append(right_of_right)

            else:
                break
        return x_samples[furthest_point_frm_rand_idx], root_data, root_radius, means_nd_radii, dists_w_idx

    def build_ball_tree(self, original_row, root_value, root_radius_val, means_nd_radii):
        """
        builds ball tree using train data
        Args:
            original_row: row that all the distances were computed wrt
            root_value: the value fo the root of the tree(a scalar)
            root_radius_val: the radius for the root if the tree (a scalar)
            means_nd_radii: a list of tuples s.t (mean, radius)
        Returns:
            a root node full connected to all other nodes, hence a tree
        """

        tree_structure = Tree(original_row)
        tree_structure.insert_root(node=None, data=root_value, sphere_radius=root_radius_val)
        complete_tree = tree_structure.insert_children(list_data=means_nd_radii, parent=None)
        return tree_structure, complete_tree

    def ball_tree_knn(self, idx_dist_dict, test, built_ball_tree):
        """
        gets labels of test df using the tree built
        Args:
            idx_dist_dict:
            test: an array/table of rows to predict labels for
            built_ball_tree: ball tree that has a saved rootnode in it

        Returns:
            An array of the indices of predicted labels

        """
        predicted_labels = []
        for row in test:
            best_dist_tuple = built_ball_tree.search(new_row=row, idx_dist_dict=idx_dist_dict)
            # use the value that gave the best distance(best_distance_tuple[1]) to get the index in df
            dict_keys = np.array(list(idx_dist_dict.keys()))
            dict_vals = np.array(list(idx_dist_dict.values()))
            differences = best_dist_tuple[1] - dict_vals
            minii = min(number for number in differences if number > 0)
            closest_neigh_idx_in_arr = np.where(differences == minii)
            predicted_label = dict_keys[closest_neigh_idx_in_arr]
            predicted_labels = np.append(predicted_labels, predicted_label)
        return predicted_labels


