## FunctionDef _tree_apply_over_list(list_of_trees, fn)
**_tree_apply_over_list**: Applies a function over a list of trees, transforming it into a tree of arrays.

**Parameters:**

- `list_of_trees`: A Python list of trees (nested structures). Each tree in the list should have the same structure.

- `fn`: The function to apply to each list of leaves collected from corresponding positions in the trees.

**Code Description:**

The `_tree_apply_over_list` function is designed to handle operations on lists of nested data structures, or "trees," by applying a specified function to corresponding elements across these trees. This is particularly useful for operations like stacking arrays from multiple states in reinforcement learning environments, where the state may have a complex nested structure.

### Functionality

1. **Structure Validation:**
   - The function first checks if the `list_of_trees` is empty. If it is, either because it's an empty list or a zero-sized numpy array, it raises a ValueError. This ensures that there are elements to process.
   
2. **Flattening Trees:**
   - It flattens each tree in the list using `tree.flatten(n)`, where `n` is each tree in the list. This step converts the nested structures into flat sequences, making it easier to handle corresponding elements.

3. **Collecting and Processing Leaves:**
   - It iterates over the positions in the flattened trees, collecting lists of leaves from the same position across all trees.
   - For each position, it gathers the elements from each flattened tree, filters out any `None` values (which might represent missing parts in the tree structure), and applies the provided function `fn` to this list of elements.
   - The results of applying `fn` are collected into a new flat sequence.

4. **Unflattening the Results:**
   - Finally, it uses `tree.unflatten_as` to reconstruct the nested structure of the original tree, using the first tree in the list as the reference structure and the processed flat sequence as the data.

### Relationship with Callers

This function is a utility used by other functions in the module, such as `tree_stack`, which specializes in stacking arrays along a specified axis. By abstracting the process of applying a function across a list of trees, `_tree_apply_over_list` allows for more modular and reusable code. For example, `tree_stack` uses this utility to apply `np.stack` to corresponding leaves across multiple trees, effectively stacking them into a new tree of stacked arrays.

### Note

- **Structure Consistency:** It is crucial that all trees in the list have the same structure; otherwise, the flattening and unflattening processes may not align correctly, leading to errors.
  
- **Handling None Values:** The function ignores `None` values in the trees, which can be useful when dealing with partial or filtered states. However, this also means that if a position in some trees is `None`, those entries are skipped in the application of `fn`.

- **Performance Considerations:** For large lists of deep trees, the flattening and unflattening steps could introduce performance overhead. It's important to consider the structure and size of the data when using this function.

### Output Example

Suppose we have a list of two trees:

```
tree1 = {'a': 1, 'b': [2, 3]}
tree2 = {'a': 4, 'b': [5, 6]}
list_of_trees = [tree1, tree2]
```

And we want to stack them using `np.stack`. The function would first flatten each tree:

```
flat_tree1 = [1, 2, 3]
flat_tree2 = [4, 5, 6]
```

Then, it would collect and stack the elements at each position:

```
stacked_a = np.stack([1, 4])
stacked_b0 = np.stack([2, 5])
stacked_b1 = np.stack([3, 6])
```

Finally, it would unflatten these stacked arrays back into the original tree structure:

```
result = {'a': array([1, 4]), 'b': [array([2, 5]), array([3, 6])]}
```

This result is a tree where each leaf is now an array formed by stacking the corresponding leaves from the input trees.
## FunctionDef tree_stack(list_of_trees, axis)
**tree_stack**: Stacks a list of trees into a tree of stacked arrays along a specified axis.

**Parameters:**

- `list_of_trees`: A Python list of trees (nested structures). Each tree in the list should have the same structure.
  
- `axis` (optional): The axis along which the stacking is performed. Defaults to 0.

**Code Description:**

The `tree_stack` function is designed to handle the stacking of multiple nested data structures, referred to as "trees," into a single nested structure where corresponding leaves are stacked into arrays. This is particularly useful in scenarios where you have multiple instances of similarly structured data (like agent states in reinforcement learning) that need to be combined for batch processing or other operations.

The function leverages the utility function `_tree_apply_over_list`, which handles the general process of applying a function across corresponding elements of a list of trees. In this case, the function applied is `np.stack`, which stacks arrays along a specified axis.

Here's a step-by-step breakdown of how `tree_stack` works:

1. **Input Validation and Structure Assumption:**
   - It assumes that all trees in the list have the same structure. This is crucial for correctly aligning and stacking the leaves.
   - Elements within the trees can be `None`, which are ignored during stacking. This feature is useful when dealing with partial or filtered data.

2. **Flattening Trees:**
   - Each tree in the list is flattened into a linear sequence of leaves using `tree.flatten(n)`. This step simplifies the process of accessing corresponding elements across different trees.

3. **Collecting and Stacking Leaves:**
   - For each position in the flattened trees, it collects the corresponding leaves from all trees.
   - `None` values are filtered out to ensure that only actual data points are stacked.
   - `np.stack` is applied to the collected lists of leaves along the specified axis to create stacked arrays.

4. **Reconstructing the Tree Structure:**
   - The stacked arrays are then reassembled into a nested structure that matches the original tree structure, using `tree.unflatten_as`.

This approach ensures that the complex nested structures are handled efficiently, and the stacking operation is applied consistently across corresponding elements.

**Note:**

- **Structure Consistency:** It is essential that all trees in the list have identical structures. Any discrepancy can lead to errors during flattening and unflattening.
  
- **Handling None Values:** Leaves with `None` values are ignored during stacking, which can be beneficial when dealing with incomplete data but requires careful management to avoid unexpected behavior.

- **Performance Considerations:** For large trees or a large number of trees, the operations involved in flattening, stacking, and unflattening may have performance implications. It's important to consider the computational resources when using this function with extensive data.

**Output Example:**

Suppose we have a list of two trees:

```
tree1 = {'a': 1, 'b': [2, 3]}
tree2 = {'a': 4, 'b': [5, 6]}
list_of_trees = [tree1, tree2]
```

Applying `tree_stack(list_of_trees)` would result in:

```
{
  'a': array([1, 4]),
  'b': [array([2, 5]), array([3, 6])]
}
```

In this output, each corresponding leaf from the input trees has been stacked into arrays along axis 0. This results in a new tree where the leaves are now arrays composed of the stacked values from the input trees.

**Relationship with Callees:**

- **_tree_apply_over_list:** This utility function is called by `tree_stack` to handle the general process of applying a function over a list of trees. It ensures that the function (in this case, `np.stack`) is applied correctly to corresponding elements across the trees, maintaining the structural integrity of the nested data.

- **np.stack:** This NumPy function is used within `_tree_apply_over_list` to stack arrays along a specified axis. It is crucial for combining the collected leaves into stacked arrays, which form the leaves of the resulting tree.

By utilizing these components, `tree_stack` provides a powerful and flexible way to handle complex nested data structures, making it easier to perform operations that require batch processing or aggregation of similar structured data.
## FunctionDef tree_expand_dims(tree_of_arrays, axis)
**tree_expand_dims**: The function `tree_expand_dims` is used to expand dimensions along a specified axis across a tree of arrays.

**Parameters**:
- `tree_of_arrays`: A tree structure (typically a nested structure like a dict or list) where each leaf is an array.
- `axis=0`: The axis along which to expand the dimension for each array in the tree.

**Code Description**:
The function `tree_expand_dims` takes a tree of arrays and an axis as input. It uses `tree.map_structure` to apply the `np.expand_dims` function to each array in the tree structure, expanding the dimension along the specified axis. The result is a new tree with the same structure where each array has an additional dimension added at the specified axis.

Here's a breakdown of the function:

1. **Input**:
   - `tree_of_arrays`: This can be a nested structure (like dictionaries or lists) containing arrays at the leaf nodes.
   - `axis` (optional, default is 0): The axis along which to expand the dimensions of the arrays.

2. **Processing**:
   - The function uses `tree.map_structure`, which applies a given function to each entry in the tree structure recursively.
   - For each array in the tree, `np.expand_dims(arr, axis)` is called to add a new dimension at the specified axis.

3. **Output**:
   - A new tree with the same structure as `tree_of_arrays`, where each array has an additional dimension added at the specified axis.

**Note**:
- Ensure that the `tree` module and `numpy` are imported before using this function.
- The `axis` parameter should be appropriate for the arrays in the tree; otherwise, it may lead to errors or unexpected behavior.
- This function is useful when you need to add a dimension to arrays in a nested structure uniformly, such as preparing data for machine learning models that expect inputs of certain shapes.

**Output Example**:
Suppose `tree_of_arrays` is a dictionary:
```python
{
    'a': np.array([1, 2, 3]),
    'b': {
        'c': np.array([4, 5, 6])
    }
}
```
After applying `tree_expand_dims(tree_of_arrays, axis=0)`, the output would be:
```python
{
    'a': np.array([[1, 2, 3]]),
    'b': {
        'c': np.array([[4, 5, 6]])
    }
}
```
Here, each array has an additional dimension added at axis 0.
