## FunctionDef bits_between(number, start, end)
---

**Function Overview**

The `bits_between` function extracts a specified range of bits from an integer number.

**Parameters**

- **number**: An integer (`int`) from which bits are extracted.
- **start**: The starting position (inclusive) of the bit range to extract, measured from the least significant bit (LSB), with 0 being the LSB.
- **end**: The ending position (exclusive) of the bit range to extract.

**Return Values**

The function returns an integer representing the bits extracted from the specified range.

**Detailed Explanation**

The `bits_between` function is designed to isolate a specific range of bits within an integer. This is achieved through bitwise operations:

1. **Bitmask Creation**: The expression `(1 << end)` creates a binary number with all bits set to 0 except for the bit at position `end`. For example, if `end` is 3, this results in the binary number `0b1000`.

2. **Modulo Operation**: Applying the modulo operation (`%`) between `number` and `(1 << end)` effectively masks all bits above the `end` position, leaving only the bits from positions `start` to `end-1`.

3. **Right Shift**: The result of the modulo operation is then right-shifted by `start` positions using the expression `// (1 << start)`. This operation moves the desired bit range to the least significant positions, making it easy to extract.

The combination of these operations isolates and returns the bits between positions `start` and `end-1`.

**Usage Notes**

- **Range Specification**: The `start` and `end` parameters define a half-open interval `[start, end)`, meaning the bit at position `start` is included in the extraction, but the bit at position `end` is not.
  
- **Bit Positioning**: Ensure that `start` and `end` are within the valid range for the integer size. For example, if working with 32-bit integers, `start` should be between 0 and 31, inclusive, and `end` should be between 1 and 32.

- **Performance Considerations**: The function is efficient due to its reliance on bitwise operations, which are generally fast. However, for very large numbers or in performance-critical applications, consider the impact of integer size on operation speed.

**Examples**

```python
# Extract bits from position 2 to 4 (inclusive) of the number 18 (binary: 0b10010)
result = bits_between(18, 2, 5)  # result is 2 (binary: 0b10)

# Extract bits from position 0 to 3 (inclusive) of the number 29 (binary: 0b11101)
result = bits_between(29, 0, 4)  # result is 13 (binary: 0b1101)
```

---

This documentation provides a comprehensive understanding of the `bits_between` function, its parameters, return values, and usage considerations.
## FunctionDef actions_for_province(legal_actions, province)
**Function Overview**

The `actions_for_province` function filters and returns actions from a given list that have their main unit located in a specified province.

**Parameters**

- **legal_actions**: A sequence of `Action` objects representing legal actions to be filtered. Each action is expected to have a structure that allows the extraction of a province ID.
  
- **province**: An instance of `utils.ProvinceID` specifying the province for which actions are to be returned.

**Return Values**

The function returns a sequence of `Action` objects from `legal_actions` where the main unit's province matches the specified `province`.

**Detailed Explanation**

The `actions_for_province` function iterates over each action in the provided list of legal actions. For each action, it uses the `ordered_province` function to extract the province ID associated with the action's main unit. If the extracted province ID matches the specified province and the action is valid (i.e., not null), the action is added to the result list.

1. **Initialization**: An empty list named `actions` is initialized to store actions that meet the criteria.
  
2. **Iteration and Filtering**:
   - The function iterates over each `action` in `legal_actions`.
   - For each `action`, it calls `ordered_province(action)` to obtain the province ID associated with the action's main unit.
   - It checks if the `action` is valid (i.e., not null) and if the extracted province ID matches the specified `province`.
   - If both conditions are met, the `action` is appended to the `actions` list.

3. **Return**: After iterating through all actions, the function returns the `actions` list containing only those actions whose main unit's province matches the specified `province`.

**Usage Notes**

- **Input Validation**: Ensure that all actions in `legal_actions` are valid and have a structure compatible with the `ordered_province` function.
  
- **Performance Considerations**: The function's performance is linear with respect to the number of actions in `legal_actions`. For large lists, consider optimizing the extraction process or parallelizing the filtering operations.

- **Edge Cases**:
  - If no actions match the specified province, an empty list is returned.
  - If any action in `legal_actions` is null or improperly structured, it will be ignored during the filtering process.

**Examples**

```python
# Example usage of actions_for_province

# Assume legal_actions is a list of Action objects and target_province is a ProvinceID instance
filtered_actions = actions_for_province(legal_actions, target_province)

# filtered_actions now contains only those actions from legal_actions where the main unit's province matches target_province
```

This documentation provides a clear understanding of the `actions_for_province` function, its parameters, return values, and usage considerations.
## FunctionDef construct_action(order, ordering_province, target_province, third_province)
**Function Overview**

The `construct_action` function is designed to construct an action representation for a given order, incorporating details about the ordering province, target province, and third province. The function returns this action as an integer without including an index.

**Parameters**

- **order (Order)**: An enumeration representing the type of action to be performed.
- **ordering_province (utils.ProvinceWithFlag)**: A tuple containing the identifier for the province from which the order originates and a flag indicating whether it is a coast. This parameter can be `None` if not applicable.
- **target_province (utils.ProvinceWithFlag)**: A tuple containing the identifier for the province to which the order is directed and a flag indicating whether it is a coast. This parameter can be `None` if not applicable.
- **third_province (utils.ProvinceWithFlag)**: A tuple containing the identifier for a third province involved in the action, such as in a convoy order. This parameter can be `None` if not applicable.

**Return Values**

- Returns an integer (`ActionNoIndex`) representing the constructed action.

**Detailed Explanation**

The function `construct_action` constructs an integer representation of an action by encoding various details into specific bit positions within this integer. The process involves bitwise operations to set different parts of the integer based on the provided parameters:

1. **Order Representation**: 
   - The order type is encoded into the integer using a left shift operation (`<<`) with `ACTION_ORDER_START` as the shift amount.
   - This sets the bits corresponding to the order type within the integer.

2. **Ordering Province**:
   - If an ordering province is provided, its identifier is encoded similarly by shifting it left by `ACTION_ORDERED_PROVINCE_START`.
   - Additionally, if the order is of type `BUILD_FLEET`, the coast flag from the ordering province is also encoded using a shift with `ACTION_ORDERED_PROVINCE_COAST`.

3. **Target Province**:
   - If a target province is provided, its identifier is encoded by shifting it left by `ACTION_TARGET_PROVINCE_START`.
   - For orders of type `MOVE_TO` or `RETREAT_TO`, the coast flag from the target province is also encoded using a shift with `ACTION_TARGET_PROVINCE_COAST`.

4. **Third Province**:
   - If a third province is provided, its identifier is encoded by shifting it left by `ACTION_THIRD_PROVINCE_START`.

The bitwise OR operation (`|=`) is used to combine these encoded parts into the final integer representation of the action.

**Usage Notes**

- The function assumes that the input parameters are correctly formatted as specified (i.e., `utils.ProvinceWithFlag` tuples).
- If any of the provinces (`ordering_province`, `target_province`, `third_province`) are not applicable, they should be passed as `None`. This will result in those parts being omitted from the final action representation.
- The function does not handle invalid orders or province identifiers. It is expected that these checks are performed prior to calling this function.
- Performance considerations: The function performs a fixed number of bitwise operations and conditional checks, making it efficient for constructing action representations.
## FunctionDef action_breakdown(action)
**Function Overview**

The `action_breakdown` function **breaks down a given action into its component parts**, specifically extracting an order and three province identifiers with coast indicators.

**Parameters**

- **action**: A 32-bit or 64-bit integer representing the action to be broken down.

**Return Values**

- **order**: An integer between 1 and 13, indicating the type of action.
- **p1**: A tuple containing a province ID and a coast indicator bit for the first ordered province.
- **p2**: A tuple containing a province ID and a coast indicator bit for the target province.
- **p3**: A tuple containing a province ID and a coast indicator bit for the third province.

**Detailed Explanation**

The `action_breakdown` function is designed to dissect an integer action into its constituent parts using bitwise operations. This process involves extracting specific ranges of bits from the action integer, which represent different components of the action:

1. **Extracting the Order**: The order of the action is extracted by calling the `bits_between` function with parameters that define the start and end positions of the order bits within the action integer. The result is an integer representing the order.

2. **Extracting Province IDs and Coast Indicators**:
   - For each province (p1, p2, p3), the process involves two steps:
     1. Extract the province ID by calling `bits_between` with parameters that define the start and end positions of the province bits.
     2. Extract the coast indicator bit by calling `bits_between` again with parameters that define the start and end positions of the coast indicator bits.
   - These two values are then combined into a tuple, where the first element is the province ID and the second element is the coast indicator bit.

The function returns these extracted components as a tuple containing the order and three tuples for the provinces.

**Usage Notes**

- **Coast Indicator Bits**: The coast indicator bits returned by this function are not area IDs as returned by `province_id` or `area`. Developers should be aware of this distinction when interpreting the results.
  
- **Action Size**: The function assumes that the action is either a 32-bit or 64-bit integer. Passing an action outside this size range may lead to unexpected behavior.

- **Performance Considerations**: The function relies on efficient bitwise operations, making it suitable for performance-critical applications where actions need to be processed quickly.

**Examples**

```python
# Example of breaking down a hypothetical action
action = 0b10100000000000000000000000000001  # Hypothetical action binary representation
order, p1, p2, p3 = action_breakdown(action)
print(order)  # Output: The extracted order (integer between 1 and 13)
print(p1)     # Output: Tuple (province ID, coast indicator bit) for the first ordered province
print(p2)     # Output: Tuple (province ID, coast indicator bit) for the target province
print(p3)     # Output: Tuple (province ID, coast indicator bit) for the third province
```

This documentation provides a clear understanding of how the `action_breakdown` function operates, its parameters, return values, and usage considerations.
## FunctionDef action_index(action)
**Function Overview**: The `action_index` function returns the index of a given action among all possible unit actions by performing a bitwise right shift operation.

**Parameters**:
- **action**: A required parameter that can be either an instance of `Action` or a NumPy array (`np.ndarray`). This represents the action for which the index is to be determined.

**Return Values**:
- The function returns either an integer or a NumPy array, depending on the type of the input `action`. It represents the index of the action among all possible unit actions.

**Detailed Explanation**:
The `action_index` function calculates the index of an action by performing a bitwise right shift operation (`>>`) on the `action` parameter. The number of positions to shift is determined by the constant `ACTION_INDEX_START`. This operation effectively extracts the higher-order bits from the `action`, which correspond to its index among all possible unit actions.

The logic behind this function assumes that each action is represented as a binary number where the lower-order bits represent specific attributes or details of the action, and the higher-order bits represent its position or order in the sequence of all possible actions. By shifting these higher-order bits to the rightmost positions, the function isolates the index value.

**Usage Notes**:
- **Input Type**: The `action` parameter must be either an instance of `Action` or a NumPy array. Other types will result in a TypeError.
- **Bitwise Operation**: The effectiveness of this function depends on the correct setting of `ACTION_INDEX_START`. If this constant does not accurately reflect the bit positions used to encode action indices, the returned index will be incorrect.
- **Performance Considerations**: The bitwise operation is computationally efficient and operates in constant time. However, if `action` is a large NumPy array, memory usage should be considered due to the return of an array of indices.

This function is crucial for mapping actions to their respective indices within a larger system that manages unit actions, allowing for efficient action management and retrieval based on index.
## FunctionDef is_waive(action)
**Function Overview**

The `is_waive` function determines whether a given action is marked as "waived" by examining specific bits within the action's integer representation.

**Parameters**

- **action**: An instance of either `Action` or `ActionNoIndex`, from which the bit range is extracted to check if it corresponds to the "waived" status.

**Return Values**

The function returns a boolean (`bool`) indicating whether the action is waived (`True`) or not (`False`).

**Detailed Explanation**

The `is_waive` function checks if an action is marked as "waived" by extracting a specific range of bits from its integer representation. This is achieved through the following steps:

1. **Bit Extraction**: The function calls `bits_between(action, ACTION_ORDER_START, ACTION_ORDER_START + ACTION_ORDER_BITS)` to extract the bit range that represents the order of the action.

2. **Comparison with WAIVE Constant**: The extracted bit value is then compared with the constant `WAIVE`. If they match, it indicates that the action is marked as "waived".

3. **Return Result**: The function returns `True` if the extracted bit value equals `WAIVE`, otherwise it returns `False`.

**Usage Notes**

- **Action Types**: The function accepts actions of type `Action` or `ActionNoIndex`. Ensure that the action object passed to the function is correctly instantiated and contains the necessary bit information.

- **Constants Definition**: The constants `ACTION_ORDER_START`, `ACTION_ORDER_BITS`, and `WAIVE` must be defined elsewhere in the codebase. Their values determine the specific bit range used for checking the "waived" status.

- **Performance Considerations**: Since the function relies on bitwise operations, it is efficient and suitable for use in performance-critical applications. However, ensure that the constants defining the bit positions are correctly set to avoid incorrect results.

**Examples**

```python
# Assuming ACTION_ORDER_START = 0, ACTION_ORDER_BITS = 3, and WAIVE = 7 (binary: 0b111)
action1 = Action(0b111)  # action is marked as waived
action2 = Action(0b001)  # action is not marked as waived

print(is_waive(action1))  # Output: True
print(is_waive(action2))  # Output: False
```

This documentation provides a comprehensive understanding of the `is_waive` function, its parameters, return values, and usage considerations.
## FunctionDef ordered_province(action)
**Function Overview**

The `ordered_province` function extracts a specific range of bits from an action, representing a province ID.

**Parameters**

- **action**: The input action from which the province ID is extracted. This can be an instance of `Action`, `ActionNoIndex`, or a NumPy array (`np.ndarray`).

**Return Values**

The function returns either a `utils.ProvinceID` if the input is a single action, or a NumPy array of `utils.ProvinceID` values if the input is an array of actions.

**Detailed Explanation**

The `ordered_province` function leverages the `bits_between` utility to isolate and extract bits representing a province ID from an action. This extraction is based on predefined constants (`ACTION_ORDERED_PROVINCE_START` and `ACTION_PROVINCE_BITS`) that define the bit range within the action where the province ID is stored.

1. **Bit Range Specification**: The function specifies the start and end positions for the bit range using the constants `ACTION_ORDERED_PROVINCE_START` and `ACTION_PROVINCE_BITS`. The start position marks the beginning of the bit range, while the end position indicates the first bit beyond the desired range.

2. **Extraction Process**: The `bits_between` function is called with the action and the defined bit range. This function performs bitwise operations to isolate and extract the bits between the specified start and end positions.

3. **Return Value**: Depending on whether the input is a single action or an array of actions, the function returns either a single `utils.ProvinceID` value or a NumPy array containing multiple `ProvinceID` values.

**Usage Notes**

- **Input Types**: The function accepts actions in various forms (`Action`, `ActionNoIndex`, or `np.ndarray`). Ensure that the input type is compatible with the expected operations.
  
- **Bit Range Constants**: The correctness of the extraction depends on accurate definitions of `ACTION_ORDERED_PROVINCE_START` and `ACTION_PROVINCE_BITS`. Misalignment can lead to incorrect province ID extraction.

- **Performance Considerations**: The function's performance is influenced by the efficiency of the `bits_between` utility. For large arrays of actions, consider optimizing bitwise operations or using vectorized NumPy functions for better performance.

**Examples**

```python
# Extract province ID from a single action
action = Action(...)  # Assume this represents an action with a defined bit structure
province_id = ordered_province(action)  # Returns the ProvinceID extracted from the action

# Extract province IDs from an array of actions
actions = np.array([Action(...), Action(...)])  # Array of actions
province_ids = ordered_province(actions)  # Returns an array of ProvinceIDs extracted from each action
```

This documentation provides a clear understanding of the `ordered_province` function, its parameters, return values, and usage considerations.
## FunctionDef shrink_actions(actions)
**Function Overview**

The `shrink_actions` function retains the top and bottom byte pairs of actions, effectively shrinking them while preserving critical information such as index, order, and ordered unit's area.

**Parameters**

- **actions**: This parameter accepts an action or a sequence of actions in a specific format. It can be a single `Action`, a collection of `Actions` (like a list or tuple), or a NumPy array (`np.ndarray`). The function processes these inputs to extract and retain the relevant byte pairs.

**Return Values**

- Returns a NumPy array containing the "shrunk" actions, where each action is represented as an integer (`np.int32`). If the input `actions` are empty, it returns an empty array of type `np.int32`.

**Detailed Explanation**

The function begins by converting the input `actions` into a NumPy array using `np.asarray(actions)`. This ensures that all subsequent operations can be performed uniformly regardless of the initial format.

If the size of the actions array is zero, indicating no actions were provided, the function immediately returns an empty array cast to `np.int32`.

For non-empty arrays, the function performs a bitwise operation to retain the top and bottom byte pairs:
- `(actions >> 32) & ~0xffff`: This part shifts the bits of each action right by 32 positions, effectively moving the top byte pair to the least significant position. The `& ~0xffff` then masks out the lower two bytes, leaving only the top byte pair.
- `(actions & 0xffff)`: This operation retains the bottom byte pair by performing a bitwise AND with `0xffff`, which has bits set in the lower two bytes and zero elsewhere.

The results of these two operations are added together using `((actions >> 32) & ~0xffff) + (actions & 0xffff)`. The addition combines the top and bottom byte pairs into a single integer, effectively shrinking each action while preserving the critical information encoded in these byte pairs. Finally, the result is cast to `np.int32` using `np.cast[np.int32]`.

**Usage Notes**

- **Input Format**: Ensure that the input actions are provided in a format compatible with NumPy arrays or can be converted into such an array. The function assumes a specific bit-level encoding of actions.
  
- **Empty Input Handling**: If no actions are provided (i.e., `actions.size == 0`), the function returns an empty array, which is useful for handling cases where action data might not always be available.

- **Performance Considerations**: The function efficiently processes actions using NumPy operations, which are optimized for performance. However, users should ensure that the input size does not exceed memory limits to avoid potential performance degradation or errors.

- **Edge Cases**: The function handles empty inputs gracefully by returning an empty array. It also assumes that the bit-level encoding of actions is consistent with the expected format; deviations may lead to incorrect results.
## FunctionDef find_action_with_area(actions, area)
**Function Overview**

The **`find_action_with_area`** function identifies and returns the first action in a list that corresponds to a specified area. If no such action exists, it returns 0.

**Parameters**

- **actions**: A sequence of actions (`Sequence[Union[Action, ActionNoIndex]]`) representing potential actions associated with units.
  
- **area**: An `utils.AreaID` specifying the area for which an action is sought.

**Return Values**

The function returns either:
- The first action (`int`) in the list that corresponds to the specified area.
- 0 if no such action exists.

**Detailed Explanation**

The `find_action_with_area` function operates by iterating through a sequence of actions and checking each one to see if it matches the provided area. This matching is performed using the `ordered_province` function, which extracts the province ID from an action. The function then compares this extracted province ID with the province ID associated with the specified area.

1. **Extracting Province ID from Area**: 
   - The function begins by extracting the province ID from the provided area using `utils.province_id_and_area_index(area)[0]`. This method returns a tuple where the first element is the province ID, and the second element is an index related to the area.

2. **Iterating Through Actions**:
   - The function then iterates through each action in the provided sequence.
   
3. **Matching Province IDs**:
   - For each action, it uses `ordered_province(a)` to extract the province ID associated with that action.
   - It compares this extracted province ID with the province ID of the specified area.

4. **Returning the Matching Action**:
   - If a match is found (i.e., the extracted province ID from the action matches the province ID of the area), the function immediately returns the action.
   
5. **Handling No Match**:
   - If no matching action is found after iterating through all actions, the function returns 0.

**Usage Notes**

- **Action Types**: The function accepts actions in the form of `Action` or `ActionNoIndex`. Ensure that the input actions are compatible with these types.
  
- **Area Specification**: The area must be a valid `utils.AreaID`. Incorrect or invalid area specifications can lead to unexpected behavior.

- **Performance Considerations**: For large sequences of actions, consider optimizing the iteration process or using more efficient data structures to improve performance.

**Examples**

```python
# Example 1: Finding an action for a specific area
actions = [Action(...), ActionNoIndex(...)]  # List of actions
area = utils.AreaID(...)  # Specify the area
matching_action = find_action_with_area(actions, area)  # Returns the first matching action or 0

# Example 2: Handling no matching action
actions_no_match = [Action(...), ActionNoIndex(...)]  # Actions that do not match the area
area = utils.AreaID(...)  # Specify the area
result = find_action_with_area(actions_no_match, area)  # Returns 0 as no matching action is found
```

This documentation provides a clear understanding of the `find_action_with_area` function, its parameters, return values, and usage considerations.
