def tree_walk(tree, node):
    yield node
    while node in tree:
        node = tree[node]
        yield node

def preorder_apply(tree, f, *args):
    visited = {}
    for node in tree:
        path = [node]
        while (node in tree) and (node not in visited):
            node = tree[node]
            path.append(node)
        if node not in visited:
            visited[node] = f(node, None, *args)
        for i in range(-2, -len(path) - 1, -1):
            visited[path[i]] = f(path[i], visited[path[i+1]], *args)
    return visited

def increment_chain_len(_, parent_chain_len):
    if not parent_chain_len: return 1
    return parent_chain_len + 1

def get_ancestor_chain_lens(tree: dict) -> dict:
    '''
    Get lengths of ancestor chains in a { child: parent } dictionary tree

    Examples
    --------
    >>> get_ancestor_chain_lens({ 0:1, 1:2, 2:3, 4:5, 5:6, 7:8 })
    {3: 1, 2: 2, 1: 3, 0: 4, 6: 1, 5: 2, 4: 3, 8: 1, 7: 2}

    Parameters
    ----------
    tree: dict
        A tree in { child: parent } format.

    Returns
    -------
    lengths: dict
        The lengths of the path to the root from each node { node: length }

    '''
    return preorder_apply(tree, increment_chain_len)

def get_roots(tree):
    ancestor_chain_lens = get_ancestor_chain_lens(tree)
    return [node for node in ancestor_chain_lens if ancestor_chain_lens[node] == 1]

def append_to_parentchild_tree(node, ancestral_chain, parentchild_tree):
    ancestral_chain = ancestral_chain or []
    for parent in ancestral_chain:
        parentchild_tree = parentchild_tree[parent]
    if node not in parentchild_tree:
        parentchild_tree[node] = {}
    return ancestral_chain + [node]

def invert_childparent_tree(tree):
    '''
    inverts a {child: parent} tree into a {root: tree} tree
    '''
    parentchild_tree = {}
    preorder_apply(tree, append_to_parentchild_tree, parentchild_tree)
    return parentchild_tree