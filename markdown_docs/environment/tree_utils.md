## FunctionDef _tree_apply_over_list(list_of_trees, fn)
## Function Overview

The `_tree_apply_over_list` function is designed to apply a given function `fn` across a list of trees, transforming them into a tree-of-lists and then applying the function to each inner list.

## Parameters

- **list_of_trees**: A Python list containing multiple trees. Each tree in the list must have the same structure.
- **fn**: The function that will be applied to each list of leaves extracted from the trees.

## Return Values

The function returns a new tree-of-lists, where each leaf is the result of applying `fn` to the corresponding lists of leaves from the input trees.

## Detailed Explanation

1. **Input Validation**:
   - The function first checks if `list_of_trees` is empty or None. If so, it raises a `ValueError`.

2. **Flattening Trees**:
   - Each tree in `list_of_trees` is flattened into a list of its leaves using the `tree.flatten(n)` method. This results in `list_of_flat_trees`, where each element is a flat list representing one tree.

3. **Creating Flat Tree of Stacks**:
   - An empty list `flat_tree_of_stacks` is initialized to store the results.
   - The function iterates over each position in the flattened trees (assuming all trees have the same structure).
   - For each position, it collects the corresponding leaf from each tree into a new list `new_list`.
   - Any `None` values are removed from `new_list` since they do not contribute to the stacking process.
   - The function `fn` is applied to `new_list`, and the result is appended to `flat_tree_of_stacks`.

4. **Unflattening the Result**:
   - Finally, the `tree.unflatten_as` method is used to convert `flat_tree_of_stacks` back into a tree structure that matches the original tree's structure.

## Usage Notes

- **Assumption of Tree Structure**: The function assumes all trees in `list_of_trees` have the same structure. If this assumption is violated, the behavior is undefined.
  
- **Handling None Values**: Elements with `None` values are ignored during the stacking process. This can be useful when dealing with partially filled state trees.

- **Performance Considerations**:
  - The function's performance may degrade with a large number of trees or deeply nested structures due to the repeated flattening and unflattening operations.
  - If `fn` is computationally expensive, this will also impact overall performance.
## FunctionDef tree_stack(list_of_trees, axis)
## Function Overview

The `tree_stack` function is designed to stack a list of trees into a tree-of-lists structure by applying the `np.stack` operation to each inner list. This function assumes that all input trees have the same structure and ignores any `None` values during the stacking process.

## Parameters

- **list_of_trees**: A Python list containing multiple trees. Each tree in the list must have the same structure.
- **axis** (optional): The axis along which to stack the arrays. This parameter is passed directly to the `np.stack` function.

## Return Values

The function returns a new tree-of-arrays, where each leaf is an array formed by stacking the corresponding lists of leaves from the input trees.

## Detailed Explanation

1. **Input Validation**:
   - The function first checks if `list_of_trees` is empty or None. If so, it raises a `ValueError`.

2. **Transformation to Tree-of-Lists**:
   - Each tree in `list_of_trees` is flattened into a list of its leaves using the `tree.flatten(n)` method. This results in `list_of_flat_trees`, where each element is a flat list representing one tree.

3. **Stacking Inner Lists**:
   - An empty list `flat_tree_of_stacks` is initialized to store the results.
   - The function iterates over each position in the flattened trees (assuming all trees have the same structure).
   - For each position, it collects the corresponding leaf from each tree into a new list `new_list`.
   - Any `None` values are removed from `new_list` since they do not contribute to the stacking process.
   - The `np.stack` function is applied to `new_list` with the specified `axis`, and the result is appended to `flat_tree_of_stacks`.

4. **Unflattening the Result**:
   - Finally, the `tree.unflatten_as` method is used to convert `flat_tree_of_stacks` back into a tree structure that matches the original tree's structure.

## Usage Notes

- **Assumption of Tree Structure**: The function assumes all trees in `list_of_trees` have the same structure. If this assumption is violated, the behavior is undefined.
  
- **Handling None Values**: Elements with `None` values are ignored during the stacking process. This can be useful when dealing with partially filled state trees.

- **Performance Considerations**:
  - The function's performance may degrade with a large number of trees or deeply nested structures due to the repeated flattening and unflattening operations.
  - If `np.stack` is computationally expensive, this will also impact overall performance.
## FunctionDef tree_expand_dims(tree_of_arrays, axis)
---

**Function Overview**

The `tree_expand_dims` function is designed to expand a specified dimension along a given axis across all arrays contained within a nested structure (`tree_of_arrays`).

**Parameters**

- **tree_of_arrays**: A nested structure (e.g., a dictionary, list, or tuple) where each leaf node is an array. This structure represents the input data that needs to have its dimensions expanded.
  
- **axis**: An integer representing the axis along which to expand the dimension of each array within `tree_of_arrays`. The default value is 0.

**Return Values**

The function returns a new nested structure with the same hierarchical layout as `tree_of_arrays`, where each array has an additional dimension inserted at the specified `axis`.

**Detailed Explanation**

`tree_expand_dims` leverages the `tree.map_structure` method to apply the `np.expand_dims` operation across all arrays in the input `tree_of_arrays`. The `np.expand_dims` function is a NumPy utility that increases the dimensionality of an array by inserting a new axis at the specified position (`axis`). 

The logic flow within `tree_expand_dims` can be broken down into the following steps:
1. **Input Validation**: Although not explicitly shown in the code snippet, it's assumed that `tree_of_arrays` is a valid nested structure where all leaf nodes are arrays.
2. **Mapping Function Application**: The `tree.map_structure` function iterates over each element in the `tree_of_arrays`. For each array encountered, it applies the lambda function `lambda arr: np.expand_dims(arr, axis)`.
3. **Dimension Expansion**: The lambda function calls `np.expand_dims`, which adds a new dimension to the array at the specified `axis`.
4. **Result Construction**: The results of applying the lambda function are aggregated into a new nested structure that mirrors the original `tree_of_arrays`.

**Usage Notes**

- **Nested Structures**: Ensure that `tree_of_arrays` is a well-formed nested structure where all leaf nodes are arrays. If the structure contains non-array elements, they will be ignored during the dimension expansion process.
  
- **Axis Specification**: The `axis` parameter must be an integer within the valid range for the dimensions of the arrays in `tree_of_arrays`. Specifying an invalid axis can lead to runtime errors.

- **Performance Considerations**: The performance of `tree_expand_dims` is primarily dependent on the size and complexity of `tree_of_arrays`, as well as the number of arrays it contains. For very large or deeply nested structures, this function may consume significant computational resources.

---

This documentation provides a comprehensive guide to understanding and using the `tree_expand_dims` function within the specified project structure.
