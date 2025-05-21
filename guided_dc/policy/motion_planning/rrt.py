from typing import Union

import numpy as np
import torch


class Node:
    def __init__(self, pos: Union[np.ndarray, torch.Tensor], parent=None):
        self.pos = pos
        self.parent = parent

    def set_parent(self, parent_node):
        self.parent = parent_node

    def __repr__(self):
        return f"Node({self.pos})"


class Tree:
    def __init__(self, root: Node):
        self.nodes = [root]

    def add_node(self, new_node: Node):
        self.nodes.append(new_node)

    def find_nearest_node(self, new_node: Node) -> Node:
        min_dist = np.inf
        nearest_node = None
        for _, node in enumerate(self.nodes):
            dist = self.distance(node, new_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node.pos, nearest_node

    @staticmethod
    def distance(node_1: Node, node_2: Node) -> float:
        # Handle both numpy and torch tensor distances
        return float(((node_1.pos - node_2.pos) ** 2).sum().sqrt())

    def reconstruct_path(self, node: Node):
        path = []
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

    def __len__(self):
        return len(self.nodes)


def steer(
    x_nearest: Union[np.ndarray, torch.Tensor],
    x_rand: Union[np.ndarray, torch.Tensor],
    step_size: float,
):
    direction = x_rand - x_nearest

    if isinstance(direction, np.ndarray):
        norm = np.linalg.norm(direction)
        if norm == 0:
            return x_nearest  # Already at the position
        # Move from x_nearest towards x_rand, scaled by step_size or up to the distance
        return x_nearest + direction / norm * min(step_size, norm)

    elif isinstance(direction, torch.Tensor):
        norm = torch.norm(direction)
        if norm == 0:
            return x_nearest  # Already at the position
        # Move from x_nearest towards x_rand, scaled by step_size or up to the distance
        return x_nearest + direction / norm * torch.min(
            torch.tensor(step_size, device=norm.device), norm
        )

    else:
        raise TypeError("Input must be either numpy.ndarray or torch.Tensor")


def rrt(
    get_pos,
    get_goal,
    sampler,
    collision_checker,
    step_size: float,
    threshold: float,
    iters=int(1e5),
):
    # Initialization
    x_start = get_pos()
    x_goal = get_goal()
    tree = Tree(Node(x_start))
    goal_node = Node(x_goal)

    # Sample new points, check collision, and add to the tree recursively
    for _ in range(iters):
        x_rand = sampler.sample()
        x_nearest, nearest_node = tree.find_nearest_node(Node(x_rand))
        x_new = steer(x_rand, x_nearest, step_size)
        if not collision_checker(
            x_nearest, x_new
        ):  # TODO: figure out how to simulate paths between x_nearest and x_new
            new_node = Node(pos=x_new, parent=nearest_node)
            tree.add_node(new_node)

            if Tree.distance(new_node, goal_node) < threshold:
                return tree.reconstruct_path(new_node)
    return None


def try_connect(
    tree_a: Tree,
    tree_b: Tree,
    collision_checker,
    step_size: float,
    threshold: float,
):
    """Try to connect the two trees."""
    node_a = tree_a.nodes[-1]  # Last added node in tree A
    node_b = tree_b.find_nearest_node(node_a)  # Nearest node in tree B to node A

    # Steer tree A towards tree B
    new_pos = steer(node_a.pos, node_b.pos, step_size)

    if not collision_checker(node_a, new_pos):
        new_node = Node(pos=new_pos, parent=node_a)
        tree_a.add_node(new_node)
        # Check if trees are close enough to connect
        if Tree.distance(new_node, node_b) < threshold:
            return new_node, node_b  # Successfully connected

    return None, None  # Trees not yet connected


def rrt_connect(
    get_pos,
    get_goal,
    sampler,
    collision_checker,
    step_size: float,
    threshold: float,
    iters=int(1e5),
):
    # Initialization
    x_start = get_pos()
    x_goal = get_goal()
    tree_start = Tree(Node(x_start))
    tree_goal = Tree(Node(x_goal))

    # Sample new points, check collision, and add to the tree recursively
    for iteration in range(iters):
        tree = [tree_start, tree_goal][iteration % 2]
        x_rand = sampler.sample()
        x_nearest, nearest_node = tree.find_nearest_node(Node(x_rand))
        x_new = steer(x_rand, x_nearest, step_size)
        if not collision_checker(
            x_nearest, x_new
        ):  # TODO: figure out how to simulate paths between x_nearest and x_new
            new_node = Node(pos=x_new, parent=nearest_node)
            tree.add_node(new_node)

            node_a, node_b = try_connect(
                tree,
                [tree_start, tree_goal].remove(tree),
                collision_checker,
                step_size,
                threshold,
            )

            if node_a and node_b:
                if iteration % 2:
                    path_a = tree_start.reconstruct_path(node_a)
                    path_b = tree_goal.reconstruct_path(node_b)[::-1]
                    return path_a + path_b
                else:
                    path_a = tree_goal.reconstruct_path(node_a)[::-1]
                    path_b = tree_start.reconstruct_path(node_b)
                    return path_b + path_a
    return None
