## FunctionDef _tree_apply_over_list(list_of_trees, fn)
**_tree_apply_over_list**: The function of _tree_apply_over_list is to transform a list-of-trees into a tree-of-lists and apply a given function to each inner list.

parameters: 
· list_of_trees: A Python list of trees, where all trees are assumed to have the same structure. Elements within the tree may be None, which will be ignored during processing.
· fn: The function to be applied on each list of leaves extracted from the list-of-trees.

Code Description: The _tree_apply_over_list function first checks if the input list_of_trees is empty and raises a ValueError if it is. It then flattens each tree in the list into a flat list using the `flatten` method, assuming that all trees have the same structure. A new list, flat_tree_of_stacks, is created to store the results of applying the function fn to each position across the flattened trees. For each position in the first flattened tree, it collects corresponding elements from all other flattened trees into a new list, ignoring any None values. This new list is then passed through the function fn and its result is appended to flat_tree_of_stacks. Finally, the function reconstructs the original tree structure using `unflatten_as` with the first tree in list_of_trees as the reference structure.

The _tree_apply_over_list function serves as a utility for operations that need to be performed on corresponding elements of multiple trees structured identically. It is used by other functions such as tree_stack, which applies np.stack to each inner list after transforming the list-of-trees into a tree-of-lists.

Note: The input list_of_trees must contain at least one element and all trees within it should have the same structure. Elements that are None in any of the trees will be ignored during the operation.

Output Example: If list_of_trees contains two trees, each with leaves [1, 2] and [3, 4], and fn is a function that sums its input, then _tree_apply_over_list would return a tree with leaves [4, 6].
## FunctionDef tree_stack(list_of_trees, axis)
**tree_stack**: The function of tree_stack is to transform a list-of-trees into a tree-of-arrays by stacking corresponding elements from each tree along a specified axis.

parameters: 
· list_of_trees: A Python list of trees, where all trees are assumed to have the same structure. Elements within the tree may be None, which will be ignored during processing.
· axis: Optional parameter specifying the axis along which to stack arrays. The default value is 0.

Code Description: The tree_stack function leverages the _tree_apply_over_list utility to first convert a list-of-trees into a tree-of-lists. It then applies the numpy `stack` function to each inner list, effectively stacking corresponding elements from each tree along the specified axis. This process is particularly useful for operations involving agent states where parts of the state tree may have been filtered out (i.e., set to None), which are ignored during the stacking operation.

The function assumes that all trees in the input list_of_trees share the same structure, and it processes them accordingly. By using _tree_apply_over_list, tree_stack efficiently handles the transformation and application steps, ensuring that the resulting output maintains the original tree structure but with arrays formed by stacking corresponding elements.

Note: The input list_of_trees must contain at least one element, and all trees within it should have the same structure. Elements that are None in any of the trees will be ignored during the operation.

Output Example: If list_of_trees contains two trees, each with leaves [1, 2] and [3, 4], then tree_stack would return a tree with leaves [[1, 3], [2, 4]] when using the default axis=0. This indicates that elements from corresponding positions in each tree have been stacked into arrays along the specified axis.
## FunctionDef tree_expand_dims(tree_of_arrays, axis)
**tree_expand_dims**: The function of tree_expand_dims is to expand dimensions along a specified axis across all arrays within a nested structure (tree) of arrays.

parameters: 
· tree_of_arrays: A nested structure (often referred to as a tree) containing numpy arrays. This can be a dictionary, list, or any other data structure that the `tree.map_structure` function can traverse.
· axis: An integer indicating the axis along which to expand the dimensions of each array in the tree. By default, this is set to 0.

Code Description:
The `tree_expand_dims` function utilizes the `tree.map_structure` method from a library (presumably JAX's tree utilities or similar) to apply a transformation across all elements within a nested structure (`tree_of_arrays`). The transformation applied here is `np.expand_dims`, which adds an extra dimension to each array at the specified axis. This operation is performed element-wise, meaning that every numpy array found in the nested structure will have its dimensions expanded by one along the given axis.

Note: 
It is important to ensure that all elements within `tree_of_arrays` are indeed numpy arrays or other objects compatible with `np.expand_dims`. If any element does not support this operation, an error will be raised. Additionally, the behavior of `np.expand_dims` depends on the shape and dimensionality of the input arrays; users should be aware of how expanding dimensions affects array shapes.

Output Example:
If `tree_of_arrays` is a dictionary with two numpy arrays as values, for example: 
```python
{
    'array1': np.array([1, 2, 3]),
    'array2': np.array([[4], [5]])
}
```
and the function is called with `axis=0`, the output will be:
```python
{
    'array1': np.array([[1],
                         [2],
                         [3]]),
    'array2': np.array([[[4]],
                         [[5]]])
}
```
Here, each array in the dictionary has an additional dimension added along axis 0.
