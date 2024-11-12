## FunctionDef _tree_apply_over_list(list_of_trees, fn)
**_tree_apply_over_list**: The function of _tree_apply_over_list is to transform a list-of-trees into a tree-of-lists and then apply a given function `fn` to each inner list.

**parameters**:
· parameter1: list_of_trees - A Python list of trees.
· parameter2: fn - the function applied on the list of leaves.

**Code Description**: The `_tree_apply_over_list` function is designed to process a collection of tree-like structures (where each structure can be thought of as a nested dictionary or a custom tree object) and apply a specified function `fn` to corresponding elements across these trees. Here’s a detailed breakdown:

1. **Input Validation**: The function first checks if the input `list_of_trees` is empty. If it is, an error is raised because at least one element is required to infer the tree structure.

2. **Flattening Trees**: It flattens each tree in the list using the `flatten` method (assuming such a method exists), which converts nested structures into simple lists. This step ensures that all trees are represented as flat lists, making it easier to process them together.

3. **Stacking Lists**: For every position within these flattened lists, it collects corresponding elements from each tree and forms new lists. These lists exclude `None` values since they represent filtered or missing data.

4. **Applying the Function**: The collected lists are then passed to the function `fn`, which processes them as needed (e.g., summing, averaging, etc.).

5. **Reconstructing Trees**: Finally, these processed lists are restructured into a tree-like format using the `unflatten_as` method, ensuring that the output maintains the original structure of the input trees.

This function is particularly useful when dealing with hierarchical data structures where some elements might be missing (represented as `None`). By ignoring these `None` values and processing only available data, `_tree_apply_over_list` ensures robust handling of incomplete or filtered tree-like inputs.

**Note**: The function assumes that all input trees have the same structure. This is a critical assumption to ensure proper alignment during the process.

**Output Example**: Suppose you have three trees representing agent states at different time steps:
```python
trees = [
    {'a': 1, 'b': [2, 3]},
    {'a': 4, 'b': [5, None]}
]
```
If you apply `_tree_apply_over_list` with `fn=lambda l: np.sum(l)`, the output might look like:
```python
{
    'a': 5,
    'b': [7, 3]
}
```
Here, `'a'` values are summed up (1 + 4 = 5), and for list `'b'`, valid elements are summed while `None` is ignored.
## FunctionDef tree_stack(list_of_trees, axis)
**tree_stack**: The function of `tree_stack` is to transform a list-of-trees into a tree-of-arrays by applying `np.stack` to corresponding elements across these trees.

**parameters**:
· parameter1: `list_of_trees` - A Python list containing multiple trees, where each tree can be a nested dictionary or custom tree object.
· parameter2: `axis` - Optional; the axis argument for `np.stack`. Default is 0.

**Code Description**: The function `tree_stack` processes a collection of tree-like structures (each with potentially different leaf values) and stacks corresponding elements from these trees along the specified axis. Here’s how it works:

1. **Input Validation**: It first checks if the input `list_of_trees` contains at least one element to ensure there is enough data for processing.

2. **Flattening Trees**: Each tree in the list is flattened into a simple list using an assumed `flatten` method, which converts nested structures into linear sequences. This step ensures that all trees are represented as flat lists, making it easier to process them together.

3. **Stacking Lists**: For every position within these flattened lists, corresponding elements from each tree are collected and form new lists. These lists filter out `None` values since they represent missing or filtered data.

4. **Applying the Function**: The collected lists are then passed to the function `np.stack`, which stacks them along the specified axis (default is 0).

5. **Reconstructing Trees**: Finally, these stacked arrays are restructured into a tree-like format using an assumed `unflatten_as` method, ensuring that the output maintains the original structure of the input trees.

This approach ensures that elements from corresponding positions in each tree are combined while ignoring any missing or filtered data (`None`). The function is particularly useful when dealing with hierarchical data where some elements might be missing due to filtering processes.

**Note**: It assumes that all input trees have the same structure, meaning they contain the same set of keys (for dictionaries) at corresponding positions. If this assumption does not hold, the output may not reflect the intended stacking behavior accurately.

**Output Example**: Suppose you have two trees: `tree1 = {'a': [1, 2], 'b': [3, 4]}` and `tree2 = {'a': [5, 6], 'b': [7, 8]}`. Calling `tree_stack([tree1, tree2])` with the default axis would result in a new tree like:
```
{'a': [[1, 5], [2, 6]], 'b': [[3, 7], [4, 8]]}
```

Each list within the output tree corresponds to stacked elements from `tree1['a']` and `tree2['a']`, and similarly for `tree1['b']` and `tree2['b']`.
## FunctionDef tree_expand_dims(tree_of_arrays, axis)
**tree_expand_dims**: The function of `tree_expand_dims` is to expand dimensions along a specified axis across a nested structure of arrays.
**parameters**:
· parameter1: `tree_of_arrays`: A nested structure (tuple, list) containing NumPy arrays or other nested structures that need to have their dimensions expanded.
· parameter2: `axis`: An integer specifying the axis along which to expand the dimension. The default value is 0.

**Code Description**: 
The function `tree_expand_dims` takes a nested structure of arrays (`tree_of_arrays`) and an optional `axis` parameter, then expands the dimension of each array in the structure along the specified axis using NumPy's `expand_dims` method. This operation preserves the overall structure of the input while increasing the dimensions of its elements.

The function uses `tree.map_structure`, which is a utility from TensorFlow that applies a given function (in this case, `lambda arr: np.expand_dims(arr, axis)`) to each element in the nested structure recursively. The result is a new nested structure with the same shape as the input but with expanded dimensions at the specified axis for all arrays.

For example:
```python
import tensorflow as tf

# Example input tree_of_arrays
tree = [np.array([1, 2]), (np.array([[3], [4]]), np.array([5, 6]))]

# Expanding dimensions along axis 0
expanded_tree = tree_expand_dims(tree)
print(expanded_tree)
```
This will output:
```python
[ array([[1],
         [2]]),
  (array([[3],
          [4]]), array([[5],
                        [6]])) ]
```

**Note**: Ensure that the input `tree_of_arrays` contains only NumPy arrays or other nested structures, as the function relies on NumPy's `expand_dims` method. Also, be aware of the chosen axis value to avoid unintended dimensionality changes.

**Output Example**: The output will be a new nested structure with the same shape as the input but with an additional dimension added along the specified axis for each array in the input.
