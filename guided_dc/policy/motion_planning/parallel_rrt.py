import torch

PAD_VALUE = torch.inf
PAD_INDEX = 10000


class Node:
    def __init__(self, cfg: torch.Tensor, extra_cfg=None, parent=None):
        self.cfg = cfg  # Shape: (num_envs, cfg_dim)
        self.parent = parent
        self.extra_cfg = extra_cfg

    def set_parent(self, parent_node):
        self.parent = parent_node

    def __repr__(self):
        return f"Node({self.cfg})"


class Tree:
    def __init__(self, root: Node):
        self.nodes = [root]

    def add_node(self, new_node: Node):
        ## Testing code
        # if not isinstance(new_node.parent, torch.Tensor):
        #     raise ValueError
        # for env_idx, parent_idx in enumerate(new_node.parent):
        #     if parent_idx == PAD_INDEX:
        #         continue
        #     else:
        #         if PAD_VALUE in self.nodes[parent_idx].cfg[env_idx]:
        #             raise ValueError
        self.nodes.append(new_node)

    def find_nearest_node(self, new_node: Node, active_idx: torch.Tensor):
        # Stack all node cfgs into a single tensor of shape (num_nodes, num_envs, cfg_dim)
        tree_cfgs = torch.stack([node.cfg for node in self.nodes], dim=0)
        new_cfg = new_node.cfg  # Shape: (num_envs, cfg_dim)

        # Initialize nearest_node_cfgs with pad_value and nearest_nodes with None
        nearest_node_cfgs = torch.full(
            (tree_cfgs.size(1), new_cfg.size(1)), PAD_VALUE, device=tree_cfgs.device
        )
        # nearest_nodes = [None] * tree_cfgs.size(1)
        min_indices_full = torch.tensor(
            [PAD_INDEX] * tree_cfgs.size(1), device=tree_cfgs.device
        )

        assert active_idx.any()
        # Only perform calculations for active environments
        dist = torch.norm(
            tree_cfgs[:, active_idx] - new_cfg[active_idx].unsqueeze(0), dim=-1
        )  # Shape: (num_nodes, num_active_env_idx)
        dist = torch.nan_to_num(dist, nan=torch.inf)
        _, min_indices = torch.min(dist, dim=0)  # Shape: (num_active_env_idx,)
        nearest_node_cfgs[active_idx] = tree_cfgs[
            min_indices,
            torch.arange(tree_cfgs.size(1), device=tree_cfgs.device)[active_idx],
        ]
        min_indices_full[active_idx] = min_indices

        return nearest_node_cfgs, min_indices_full

    @staticmethod
    def distance(node_1: Node, node_2: Node) -> torch.Tensor:
        cfg_1 = node_1.cfg
        cfg_2 = node_2.cfg
        # Create masks for rows containing PAD_VALUE
        mask_1 = (cfg_1 == PAD_VALUE).any(dim=-1)  # Shape: (num_envs,)
        mask_2 = (cfg_2 == PAD_VALUE).any(dim=-1)  # Shape: (num_envs,)
        # Initialize distance tensor with torch.inf
        dist = torch.full((cfg_1.shape[0],), PAD_VALUE, device=cfg_1.device)
        # Calculate distances only for rows where neither contains PAD_VALUE
        valid_mask = ~(mask_1 | mask_2)  # Rows where neither is PAD_VALUE
        if valid_mask.any():
            dist[valid_mask] = (
                ((cfg_1[valid_mask] - cfg_2[valid_mask]) ** 2).sum(dim=-1).sqrt()
            )
        return dist

    def reconstruct_path_with_extra(
        self,
        env_done_nodes: dict[int : tuple[Node]],
        node_order,
        reverse_path,
        num_envs,
        pad_for_failed_envs,
        pad_for_extra,
    ):
        path_dict = {}
        extra_path_dict = {}
        for env_idx, final_node_pair in env_done_nodes.items():
            path = []
            extra_path = []
            node = final_node_pair[node_order]
            idx = 0
            while True:
                idx += 1
                ## Testing code
                # if (node.cfg[env_idx] == PAD_VALUE).all():
                #     print(idx)
                #     raise ValueError("Null-filled node traced incorrectly.")
                path.append(node.cfg[env_idx])
                extra_path.append(node.extra_cfg[env_idx])
                if node.parent is not None:
                    assert not (node.parent[env_idx] == PAD_INDEX)
                    node_parent_idx = node.parent[env_idx]
                    node = self.nodes[node_parent_idx]
                else:
                    break
            path_dict[env_idx] = path[::-1] if reverse_path else path
            extra_path_dict[env_idx] = extra_path[::-1] if reverse_path else extra_path
        for i in range(num_envs):
            if i not in list(path_dict.keys()):
                path_dict[i] = [pad_for_failed_envs[i]]
                extra_path_dict[i] = [pad_for_extra[i]]
        path_dict = dict(sorted(path_dict.items()))
        extra_path_dict = dict(sorted(extra_path_dict.items()))
        paths = [v for v in path_dict.values()]
        extra_paths = [v for v in extra_path_dict.values()]
        return paths, extra_paths

    def reconstruct_path(
        self,
        env_done_nodes: dict[int : tuple[Node]],
        node_order,
        reverse_path,
        num_envs,
        pad_for_failed_envs,
    ):
        path_dict = {}
        for env_idx, final_node_pair in env_done_nodes.items():
            path = []
            node = final_node_pair[node_order]
            idx = 0
            while True:
                idx += 1
                path.append(node.cfg[env_idx])
                if node.parent is not None:
                    assert not (node.parent[env_idx] == PAD_INDEX)
                    node_parent_idx = node.parent[env_idx]
                    node = self.nodes[node_parent_idx]
                else:
                    break
            path_dict[env_idx] = path[::-1] if reverse_path else path
        for i in range(num_envs):
            if i not in list(path_dict.keys()):
                path_dict[i] = [pad_for_failed_envs[i]]
        path_dict = dict(sorted(path_dict.items()))
        paths = [v for v in path_dict.values()]
        return paths

    def __len__(self):
        return len(self.nodes)


def steer(
    x_nearest: torch.Tensor,
    x_rand: torch.Tensor,
    step_size: float,
    active_env_idx: torch.Tensor,
):
    # Ensure the input tensors are of the same shape and are torch.Tensor
    assert isinstance(x_nearest, torch.Tensor) and isinstance(
        x_rand, torch.Tensor
    ), "Inputs must be torch.Tensors for parallel computation."
    assert (
        x_nearest.shape == x_rand.shape
    ), "x_nearest and x_rand must have the same shape."

    x_new = torch.full(x_rand.size(), fill_value=PAD_VALUE, device=x_nearest.device)

    # Only compute for active environments
    if active_env_idx.any():
        direction = (
            x_rand[active_env_idx] - x_nearest[active_env_idx]
        )  # Shape: (num_active_env_idx, cfg_dim)
        norm = torch.norm(
            direction, dim=-1, keepdim=True
        )  # Shape: (num_active_env_idx, 1)
        # Avoid division by zero: if norm is zero, we are already at the position, so return x_nearest for those
        norm = torch.where(
            norm == 0, torch.ones_like(norm), norm
        )  # Replace zero norms with 1 to avoid division by zero
        # Move from x_nearest towards x_rand, scaled by step_size or up to the distance to x_rand
        step_sizes = torch.min(
            torch.full_like(norm, step_size, device=direction.device), norm
        )  # Shape: (num_active_env_idx, 1)
        # Update the new positions for active environments only
        x_new[active_env_idx] = (
            x_nearest[active_env_idx] + direction / norm * step_sizes
        )  # Shape: (num_active_env_idx, cfg_dim)
    return x_new


def try_connect(
    tree_a: Tree,
    tree_b: Tree,
    collision_checker,
    step_size: float,
    threshold: float,
    active_env_idx: torch.Tensor,
    collision_idx: torch.Tensor,
    env_done_nodes: dict,
    is_start,
    get_extra_cfg,
    use_extra,
):
    """Try to connect the two trees in parallel, neglecting inactive environments."""

    # The last node is the x_new just added. x_new has PAD_VALUE for both collision_idx and inactive enviroments
    node_now_a = tree_a.nodes[-1]  # Node for all envs
    node_now_a_idx = len(tree_a) - 1
    active_idx = active_env_idx & (~collision_idx)

    new_active_env_idx = active_env_idx

    if active_idx.any():
        # x_nearest_b and nearest_nodes_indices_b has pad values for ~active_idx
        x_nearest_b, nearest_nodes_indices_b = tree_b.find_nearest_node(
            node_now_a, active_idx=active_idx
        )

        # x_now_a has PAD_VALUE for ~active_idx environments
        x_now_a = node_now_a.cfg  # Shape: (num_envs, cfg_dim)

        # Only steer in active environments.
        # TODO: actually we can steer in collided environments, but that requires we know the last non-collided node for the collided enviroment
        # x_now_a and x_nearest_b have PAD_VALUE for ~active_idx enviroments
        x_new_a = steer(x_now_a, x_nearest_b, step_size, active_idx)
        # x_new_a has PAD_VALUE for ~active_idx environments
        new_collision_idx = collision_checker(
            x_new_a, active_idx
        )  # Shape: (num_envs, )
        x_new_a[new_collision_idx] = PAD_VALUE
        parent_nodes_indices_a = torch.full(
            (x_now_a.size(0),), PAD_INDEX, device=x_new_a.device
        )
        parent_nodes_indices_a[active_idx & ~new_collision_idx] = node_now_a_idx
        new_node_a = Node(
            x_new_a,
            parent=parent_nodes_indices_a,
            extra_cfg=get_extra_cfg() if use_extra else None,
        )

        ## Testing code
        # assert torch.inf not in x_new_a[active_idx & ~new_collision_idx]
        # assert (x_new_a[~(active_idx & ~new_collision_idx)] == torch.inf).all()
        # assert (PAD_INDEX not in parent_nodes_indices_a[active_idx & ~new_collision_idx]) and (parent_nodes_indices_a[~(active_idx & ~new_collision_idx)] == PAD_INDEX).all()

        tree_a.add_node(new_node_a)

        # If has collision, the enviroment cannot be done.
        # Check the distance between x_nearest_b and x_new_a. x_new_a has more PAD_VALUES than x_nearest_b
        # This means the nearest nodes in tree_b to the new node in tree_a.
        # x_nearest_b[new_collision_idx] = PAD_VALUE
        # nearest_nodes_indices_b[new_collision_idx] = PAD_INDEX

        # assert torch.inf not in x_nearest_b[active_idx & ~new_collision_idx]
        # assert (x_nearest_b[~(active_idx & ~new_collision_idx)] == torch.inf).all()
        # assert (PAD_INDEX not in nearest_nodes_indices_b[active_idx & ~new_collision_idx]) and (nearest_nodes_indices_b[~(active_idx & ~new_collision_idx)] == PAD_INDEX).all()

        dist = Tree.distance(new_node_a, Node(x_nearest_b))  # Shape: (num_envs, )
        new_inactive_env_idx = dist < threshold

        # Has torch.inf = previous inactive or has collision <-> dist>threshold <-> new_inactive_env_idx=False
        # If active env idx is already set to False, it shouldn't be set to True again.
        # i.e., considering (active_env_idx, new_inactive_env_idx), (False, True) -> False, (False, False) -> False, (True, False) -> True, (True, True) -> False
        new_active_env_idx = active_env_idx & ~new_inactive_env_idx

        if new_active_env_idx.equal(active_env_idx):
            return env_done_nodes, new_active_env_idx

        for i in range(x_now_a.size(0)):
            # Only those environments without collision and are still active satisfie the criteria
            if new_active_env_idx[i] == 0 and (
                new_active_env_idx[i] != active_env_idx[i]
            ):
                new_node_b = Node(
                    x_nearest_b,
                    parent=nearest_nodes_indices_b,
                    extra_cfg=get_extra_cfg() if use_extra else None,
                )
                env_done_nodes[i] = (
                    (new_node_a, new_node_b)
                    if not is_start
                    else (new_node_b, new_node_a)
                )

    return env_done_nodes, new_active_env_idx


def parallel_rrt_connect(
    start_cfg,
    goal_cfg,
    iterations: int,
    sampler,
    step_size: float,
    collision_checker,
    threshold: float,
    get_extra_cfg,
    start_ee,
    goal_ee,
    active_joint_indices=[0, 1, 2, 3, 4, 5, 6],
    pad_inactive_joints_value=1,
):
    # Cfg can be ee pose, joint pos
    if start_ee is None or goal_ee is None:
        use_extra = False
    else:
        use_extra = True
    tree_start = Tree(Node(start_cfg, extra_cfg=get_extra_cfg() if use_extra else None))
    tree_goal = Tree(Node(goal_cfg, extra_cfg=get_extra_cfg() if use_extra else None))

    device = start_cfg.device
    num_envs = start_cfg.shape[0]

    active_env_idx = torch.ones((num_envs), dtype=torch.bool, device=device)
    env_done_nodes = {}

    for iteration in range(iterations):
        trees = [tree_start, tree_goal]
        is_start = iteration % 2
        tree = trees.pop(is_start)
        x_rand = sampler(
            active_env_idx=active_env_idx,
            pad_value=PAD_VALUE,
            active_joint_indices=active_joint_indices,
            pad_inactive_joints_value=pad_inactive_joints_value,
        )  # Shape: (num_envs, cfg_dim), PAD_VALUE for done environments

        # If x_rand has PAD_VALUE in it, then it must mean the corresponding env is inactive, so it won't be calculated in find_nearest_node
        x_nearest, nearest_node_indices = tree.find_nearest_node(
            Node(x_rand), active_idx=active_env_idx
        )  # Shape: (num_envs, cfg_dim); PAD_INDEX for inactive enviroments

        # x_nearest should contain torch.inf for [~active_env_idx,:], and no torch.inf for [active_env_idx, :]
        # If x_rand has PAD_VALUE in it, corresponding x_nearest should also have PAD_VALUE for ~active_env_idx
        # x_rand and x_nearest both have and only have PAD_VALUE in [~active_env_idx, :]
        x_new = steer(
            x_nearest, x_rand, step_size, active_env_idx
        )  # Shape: (num_envs, cfg_dim)

        # x_new, x_rand and x_nearest all have and only have PAD_VALUE in [~active_env_idx, :]
        collision_idx = collision_checker(x_new, active_env_idx)  # Shape: (num_envs, )

        # x_new has PAD_VALUE for both collision_idx and inactive enviroments. Essentially, inactive environments and collision are the same, because in single env setting
        # these nodes won't be added. The reason we use PAD_VALUE is to notify we shouldn't add the nodes in corresponding environments.
        # Adding these nodes are just for padding purposes, as they should never be traced when reconstructing paths or used to calculate nearest nodes.
        x_new[collision_idx] = PAD_VALUE
        nearest_node_indices[~active_env_idx | collision_idx] = PAD_INDEX

        # Only save indices to save space. For those nodes that shouldn't be add, their parents should be set to PAD_INDEX.
        # In this way, x_new is consistent with parent nodes
        tree.add_node(
            Node(
                x_new,
                parent=nearest_node_indices,
                extra_cfg=get_extra_cfg() if use_extra else None,
            )
        )
        env_done_nodes, active_env_idx = try_connect(
            tree_a=tree,
            tree_b=trees[0],
            collision_checker=collision_checker,
            step_size=step_size,
            threshold=threshold,
            active_env_idx=active_env_idx,
            collision_idx=collision_idx,
            env_done_nodes=env_done_nodes,
            is_start=is_start,
            get_extra_cfg=get_extra_cfg,
            use_extra=use_extra,
        )

        if len(env_done_nodes) == num_envs:
            break
        print(iteration, active_env_idx)
    if use_extra:
        path_start, extra_path_start = tree_start.reconstruct_path_with_extra(
            env_done_nodes,
            node_order=0,
            reverse_path=True,
            num_envs=len(active_env_idx),
            pad_for_failed_envs=start_cfg,
            pad_for_extra=start_ee,
        )
        path_goal, extra_path_goal = tree_goal.reconstruct_path_with_extra(
            env_done_nodes,
            node_order=1,
            reverse_path=False,
            num_envs=len(active_env_idx),
            pad_for_failed_envs=goal_cfg,
            pad_for_extra=goal_ee,
        )
        return merge_path(path_start, path_goal), merge_path(
            extra_path_start, extra_path_goal
        )
    else:
        path_start = tree_start.reconstruct_path(
            env_done_nodes,
            node_order=0,
            reverse_path=True,
            num_envs=len(active_env_idx),
            pad_for_failed_envs=start_cfg,
        )
        path_goal = tree_goal.reconstruct_path(
            env_done_nodes,
            node_order=1,
            reverse_path=False,
            num_envs=len(active_env_idx),
            pad_for_failed_envs=goal_cfg,
        )
        return merge_path(path_start, path_goal), None


def merge_path(
    path_start: list[list[torch.Tensor]], path_goal: list[list[torch.Tensor]]
):
    # Calculate the maximum trajectory length across all environments
    len_paths = [len(path_start[i]) + len(path_goal[i]) for i in range(len(path_start))]
    max_len = max(len_paths)

    paths = []
    for env_idx in range(len(path_start)):
        _path_start = path_start[env_idx]
        _path_goal = path_goal[env_idx]

        # Concatenate the start and goal paths
        full_path = _path_start + _path_goal

        # Convert the full path to a tensor and pad with zeros
        full_path_tensor = torch.stack(full_path, dim=0)

        # Calculate how much padding is needed
        padding_len = max_len - full_path_tensor.size(0)

        if padding_len > 0:
            # Extract the last row of the full_path_tensor
            last_row = full_path_tensor[-1, :]

            # Repeat the last row padding_len times
            padding = last_row.unsqueeze(0).repeat(padding_len, 1)

            # Concatenate the padding to the full_path_tensor
            full_path_tensor = torch.cat([full_path_tensor, padding], dim=0)

        paths.append(full_path_tensor)

    # Stack the padded paths into a single tensor with shape (num_envs, max_traj_len, cfg_dim)
    return torch.stack(paths, dim=0)
