## FunctionDef bits_between(number, start, end)
**Function Overview**
The `bits_between` function extracts a specific range of bits from an integer number. This function is used in various parts of the project, particularly for breaking down actions into their component parts.

**Parameters**
- **number (int)**: The input integer from which bits are extracted.
- **start (int)**: The starting position of the bit range to extract.
- **end (int)**: The ending position of the bit range to extract. The end position is exclusive, meaning it does not include the bit at this index.

**Return Values**
The function returns an integer that represents the bits extracted from the specified range in `number`.

**Detailed Explanation**
The `bits_between` function works by using bitwise operations to isolate a specific segment of bits within the input number. Here’s how the logic unfolds:

1. **Bitwise AND Operation**: The expression `1 << end` shifts 1 to the left by `end` positions, creating a bitmask where only the bit at position `end-1` is set.
2. **Modulo Operation**: `number % (1 << end)` effectively masks out all bits beyond position `end-1`, leaving only the lower `end` bits of the number.
3. **Bitwise AND Operation with Mask**: The result from step 2 is then masked again using a bitmask created by shifting 1 to the left by `start` positions, which is done with `(1 << start)`. This operation isolates the desired bit range.

The final value returned is the bits between positions `start` and `end-1`.

**Interactions with Other Components**
This function is used in several other functions within the project, such as `action_breakdown`, to extract specific parts of an action number. For example, it helps in extracting coast indicator bits from actions.

**Usage Notes**
- **Preconditions**: Ensure that `start` and `end` are valid indices for the input integer. The function assumes that `0 <= start < end`.
- **Performance Considerations**: This function is efficient as it uses bitwise operations, which are generally fast.
- **Edge Cases**: If `start` equals `end`, the function will return 0 because no bits are extracted in this case.

**Example Usage**
Here’s an example of how to use the `bits_between` function:

```python
# Example input number and bit positions
number = 0b10101010101010101010101010101010  # Binary representation for clarity

# Extract bits from position 5 to 9 (inclusive of 5, exclusive of 10)
start = 5
end = 10
result = bits_between(number, start, end)

print(f"Extracted bits: {bin(result)}")  # Output will be the binary representation of the extracted bits

# Expected output for the example: 0b1010 (binary representation of decimal 10)
```

This documentation provides a comprehensive understanding of how the `bits_between` function operates and its usage within the project. It is designed to help developers effectively integrate this function into their code while ensuring they are aware of potential edge cases and performance considerations.
## FunctionDef actions_for_province(legal_actions, province)
### Function Overview
The `actions_for_province` function filters a list of legal actions based on whether they involve units in a specific province.

### Parameters
1. **legal_actions (Sequence[Action])**: A sequence containing valid action objects that need to be filtered according to their main unit's province.
2. **province (utils.ProvinceID)**: The province ID for which the function filters the actions, ensuring only those with units in this specific province are returned.

### Return Values
- **Sequence[Action]**: A list of actions from `legal_actions` where the main unit is located in the specified `province`.

### Detailed Explanation
The `actions_for_province` function iterates through each action in the `legal_actions` sequence. For each action, it determines the province ID using the `ordered_province` function and checks if this province matches the provided `province`. If a match is found, the action is added to the resulting list of actions.

1. **Initialization**: An empty list named `actions` is initialized to store the filtered actions.
2. **Iteration and Filtering**:
   - For each action in `legal_actions`, the function calls `ordered_province(action)` to extract the province ID from the action.
   - The `ordered_province` function returns a value of type `utils.ProvinceID` or an array, which is then compared with the provided `province`.
   - If the extracted province ID matches the specified `province`, the action is appended to the `actions` list.

### Interactions with Other Components
- **ordered_province**: This function interacts with `actions_for_province` by providing the province ID for each action. The interaction ensures that actions are filtered based on their associated provinces.

### Usage Notes
1. **Preconditions**:
   - Ensure that all elements in `legal_actions` are valid action objects.
   - Verify that `province` is a valid `utils.ProvinceID`.

2. **Performance Considerations**: 
   - The function's performance depends on the length of `legal_actions`. If this list is very large, consider optimizing or caching province IDs to improve efficiency.

3. **Edge Cases**:
   - If no actions involve units in the specified province, an empty list will be returned.
   - If `province` is not a valid `utils.ProvinceID`, the function may raise an error depending on how `ordered_province` handles invalid inputs.

4. **Common Pitfalls**: 
   - Ensure that all action objects have a consistent structure and that the `ordered_province` function correctly extracts province IDs from them.
   - Be cautious with actions involving multiple units, as only one unit's province is considered for filtering.

### Example Usage
```python
from typing import Sequence

# Assuming Action and utils.ProvinceID are defined elsewhere in the codebase
actions = [Action1, Action2, Action3]  # List of action objects
province_id = utils.ProvinceID(5)      # Specific province ID to filter by

filtered_actions = actions_for_province(actions, province_id)
print(filtered_actions)  # Output: List of actions involving units in province 5
```

This example demonstrates how to use the `actions_for_province` function to filter a list of actions based on their association with a specific province.
## FunctionDef construct_action(order, ordering_province, target_province, third_province)
**Function Overview**
The `construct_action` function constructs an action representation based on the provided order and provinces without including the action index.

**Parameters**

1. **order (Order)**: The type of action being performed, such as building a fleet or moving troops.
2. **ordering_province (utils.ProvinceWithFlag)**: A province that is involved in the ordering process, which may be `None` if not applicable.
3. **target_province (utils.ProvinceWithFlag)**: The target province for the order, which may also be `None` if not applicable.
4. **third_province (utils.ProvinceWithFlag)**: An additional province that might be relevant in certain orders, such as a third province to support an action like a move or retreat.

**Return Values**
- **ActionNoIndex**: A numeric representation of the constructed action without the index.

**Detailed Explanation**

The `construct_action` function constructs an action representation by encoding various parameters into a single integer. The process involves several bitwise operations and conditional checks:

1. **Initialization**: Start with `order_rep = 0`, which will be used to build the final action representation.
2. **Order Encoding**: Encode the order type using bit shifting and bitwise OR operations:
   - Shift the order value left by `ACTION_ORDER_START` bits and perform a bitwise OR with `order_rep`.
3. **Ordering Province Handling**:
   - If an ordering province is provided, encode its relevant information (likely a flag or identifier) by shifting it left by `ACTION_ORDERED_PROVINCE_START` bits and performing a bitwise OR.
   - For certain orders like `BUILD_FLEET`, additional information about the coast of the ordering province can be encoded similarly.
4. **Target Province Handling**:
   - If a target province is provided, encode its relevant information (likely a flag or identifier) by shifting it left by `ACTION_TARGET_PROVINCE_START` bits and performing a bitwise OR.
   - For orders like `MOVE_TO` or `RETREAT_TO`, the coast of the target province can also be encoded similarly.
5. **Third Province Handling**:
   - If a third province is provided, encode its relevant information (likely a flag or identifier) by shifting it left by `ACTION_THIRD_PROVINCE_START` bits and performing a bitwise OR.

The final value stored in `order_rep` represents the constructed action without an index.

**Interactions with Other Components**

- The function interacts with other parts of the project through its parameters, which are likely defined elsewhere in the codebase.
- It relies on constants like `ACTION_ORDER_START`, `ACTION_ORDERED_PROVINCE_START`, and `ACTION_TARGET_PROVINCE_START` to determine where bits should be shifted.

**Usage Notes**

- Ensure that all input parameters (`order`, `ordering_province`, `target_province`, and `third_province`) are appropriately defined before calling this function.
- The function does not handle cases where the same province is both an ordering province and a target province, so ensure these scenarios are managed elsewhere in your application logic.
- Performance considerations: This function operates efficiently as it uses bitwise operations, which are generally fast. However, if performance becomes an issue, consider profiling to determine whether optimizations are necessary.

**Example Usage**

```python
from environment import Order, utils

# Example order and provinces
order = Order.BUILD_FLEET
ordering_province = utils.ProvinceWithFlag(10, 2)  # (province_id, flag)
target_province = utils.ProvinceWithFlag(5, 3)     # (province_id, flag)
third_province = None

# Construct the action without an index
action_representation = construct_action(order, ordering_province, target_province, third_province)

print(f"Constructed Action: {action_representation}")
```

This example demonstrates how to use `construct_action` with a specific order and provinces. Adjust the parameters according to your application's requirements.
## FunctionDef action_breakdown(action)
**Function Overview**
The `action_breakdown` function breaks down an action into its component parts, extracting specific bits from a 32-bit or 64-bit integer.

**Parameters**
1. **action (Union[Action, ActionNoIndex])**: The input action represented as a 32-bit or 64-bit integer. This parameter is the primary data source for the function to extract relevant information.

**Return Values**
The function returns a tuple containing:
- A 32-bit integer representing the extracted bits from the specified positions.
- The remaining bits after extraction, which are not part of the current action's components.

**Detailed Explanation**
1. **Input Validation**: The `action` parameter is assumed to be valid and within the expected range for a 32-bit or 64-bit integer.
2. **Bit Extraction Logic**:
    - The function uses the `bits_between` helper function to extract bits between specified positions.
    - Specifically, it extracts bits from position `start` to `end-1`.
3. **Helper Function `bits_between`**:
    - This function calculates the extracted bits using the formula: `number % (1 << end) // (1 << start)`.
    - The `%` operator is used to isolate the relevant bit sequence.
    - The `//` operator shifts the isolated sequence back to its original position, effectively extracting the desired bits.

**Interactions with Other Components**
- The `action_breakdown` function interacts with other parts of the project by providing a standardized way to extract and process action data. It is likely used in conjunction with other functions that handle different aspects of action processing or storage.

**Usage Notes**
1. **Preconditions**: Ensure that the input `action` parameter is a valid 32-bit or 64-bit integer.
2. **Performance Considerations**: The function performs basic arithmetic operations, which are generally efficient for small to medium-sized integers. However, if performance becomes an issue with very large datasets, consider optimizing the bit extraction logic.
3. **Edge Cases**:
    - If `start` is greater than or equal to `end`, the function will return 0 as no bits are extracted.
    - If `action` is outside the valid range for a 32-bit integer, unexpected behavior may occur.

**Example Usage**
```python
# Example: Extracting specific bits from an action
def example_usage():
    # Sample input action represented as a 32-bit integer
    sample_action = 0b10101010101010101010101010101010

    # Extract bits from position 5 to 9 (inclusive of 5, exclusive of 10)
    start = 5
    end = 10

    # Call the function
    extracted_bits, remaining_bits = action_breakdown(sample_action, start, end)

    print(f"Extracted bits: {bin(extracted_bits)}")
    print(f"Remaining bits: {bin(remaining_bits)}")

# Output:
# Extracted bits: 0b1010 (binary representation of decimal 10)
# Remaining bits: 0b1010101010101010101010101010
```

This example demonstrates how to use the `action_breakdown` function to extract specific bits from an action and handle the remaining bits.
## FunctionDef action_index(action)
**Function Overview**
The `action_index` function returns the index of a given action among all possible unit actions. This function is useful in scenarios where actions need to be mapped to numerical indices, facilitating easier handling and processing.

**Parameters**
- **action (Union[Action, np.ndarray])**: The input can either be an individual `Action` object or an array-like structure containing multiple actions. Each action represents a specific unit action within the environment's action space.

**Return Values**
- **Union[int, np.ndarray]**: If a single `Action` object is passed, the function returns an integer representing its index in the list of all possible actions. If an array-like structure is provided, it returns an array containing indices corresponding to each action in the input.

**Detailed Explanation**
The `action_index` function operates by performing a bitwise shift operation on the input `action`. The bitwise shift right (`>>`) operator shifts the bits of the binary representation of the input value to the right by `ACTION_INDEX_START` positions. This effectively reduces the input value, which is an index within the action space, to its final numerical form.

The key variable `ACTION_INDEX_START` is assumed to be defined elsewhere in the codebase and represents the starting point for indexing actions. The shift operation ensures that each unique action maps to a distinct integer index, making it easier to handle and process these actions programmatically.

**Interactions with Other Components**
This function interacts with other components of the environment by providing a standardized way to convert actions into indices. This conversion is essential for various operations such as updating state representations, logging, or implementing decision-making algorithms that rely on numerical action indices.

**Usage Notes**
- **Preconditions**: Ensure that the input `action` is either an individual `Action` object or an array-like structure containing valid actions.
- **Performance Considerations**: The function performs a simple bitwise shift operation, which is efficient. However, for large arrays of actions, consider performance implications and potential optimization strategies if necessary.
- **Edge Cases**:
  - If the input `action` is not of type `Action` or an array-like structure, it may result in unexpected behavior or errors.
  - The function assumes that `ACTION_INDEX_START` is correctly defined elsewhere; otherwise, incorrect indexing will occur.

**Example Usage**
```python
# Example with a single action
from environment.action_utils import Action

action = Action()
index = action_index(action)  # Returns the index of the given action

# Example with multiple actions in an array-like structure
actions = [Action(), Action()]
indices = action_index(actions)  # Returns an array of indices corresponding to each action
```

This documentation provides a clear understanding of the `action_index` function, its parameters, return values, and usage scenarios. It also highlights important considerations for developers working with this code in their projects.
## FunctionDef is_waive(action)
**Function Overview**
The `is_waive` function determines whether a given action corresponds to a waive order by examining specific bits within an action number.

**Parameters**
- **action (Union[Action, ActionNoIndex])**: The input action or action without index. This parameter is expected to be an instance of the `Action` or `ActionNoIndex` class, which likely represents different types of actions in the project.

**Return Values**
The function returns a boolean value:
- `True` if the order extracted from the action corresponds to a waive order.
- `False` otherwise.

**Detailed Explanation**
The `is_waive` function works by using the `bits_between` utility function to extract specific bits from the input action number. The logic is as follows:

1. **Extract Order Bits**: 
   - The `bits_between` function is called with three parameters: the `action`, a starting position (`ACTION_ORDER_START`), and an ending position (`ACTION_ORDER_START + ACTION_ORDER_BITS`).
   - This extracts a segment of bits from the action number that represents the order type.

2. **Compare to Waive Order**:
   - The extracted bits are then compared against the constant `WAIVE`.
   - If the extracted bits match `WAIVE`, the function returns `True`, indicating that the action corresponds to a waive order.
   - Otherwise, it returns `False`.

The key steps in this process involve bitwise operations and bit masking. Specifically:
- The `bits_between` function uses modulo and division operations to isolate the relevant bits from the input number.
- These isolated bits are then compared against the constant value `WAIVE`.

**Interactions with Other Components**
This function interacts with other parts of the project, particularly where action types need to be determined or validated. It is used in scenarios where actions need to be categorized based on their order type.

**Usage Notes**
- **Preconditions**: The input `action` must be an instance of either `Action` or `ActionNoIndex`.
- **Performance Considerations**: The function performs a simple bitwise operation and comparison, which is efficient.
- **Edge Cases**: Ensure that the `WAIVE` constant accurately represents the expected order type. Any mismatch could lead to incorrect results.

**Example Usage**
Here is an example of how the `is_waive` function might be used in practice:

```python
# Assuming Action and ActionNoIndex are defined classes, and ACTION_ORDER_START and ACTION_ORDER_BITS are constants.
from some_module import Action, ActionNoIndex

def check_action_type(action):
    if isinstance(action, Action):
        return is_waive(action)
    elif isinstance(action, ActionNoIndex):
        return False  # No index actions do not correspond to waive orders
    else:
        raise ValueError("Invalid action type")

# Example usage
action1 = Action()
action2 = ActionNoIndex()

print(check_action_type(action1))  # Output: True or False based on the action's order bits
print(check_action_type(action2))  # Output: False

```

This example demonstrates how `is_waive` can be integrated into a larger system to determine if an action corresponds to a waive order.
## FunctionDef ordered_province(action)
**Function Overview**
The `ordered_province` function extracts a specific range of bits from an action number, representing the province ID in the context of the game environment.

**Parameters**
- **action (Union[Action, ActionNoIndex, np.ndarray])**: The input action or array of actions from which to extract the province ID. This parameter can be either a single `Action` object, a `ActionNoIndex` object, or an array-like structure containing multiple actions.

**Return Values**
The function returns a value of type `Union[utils.ProvinceID, np.ndarray]`, representing the extracted province ID(s) from the input action(s).

**Detailed Explanation**
The `ordered_province` function utilizes the `bits_between` helper function to extract bits from an integer representation of an action. The specific range of bits is determined by the constants `PROVINCE_BITS_LOW` and `PROVINCE_BITS_HIGH`, which define the low and high bit positions, respectively.

1. **Bit Extraction**: 
   - The function first calls `bits_between(action, PROVINCE_BITS_LOW, PROVINCE_BITS_HIGH)` to extract a sub-range of bits from the action.
   - These extracted bits represent the province ID within the action's integer representation.

2. **Return Value**:
   - If the input is a single `Action` or `ActionNoIndex`, the function returns the extracted province ID as an instance of `utils.ProvinceID`.
   - If the input is an array-like structure, the function returns an array containing the province IDs for each action.

**Interactions with Other Components**
- **bits_between**: The `ordered_province` function relies on the `bits_between` helper function to perform bit extraction. This function is responsible for isolating and returning a specific range of bits from the input integer.
- **utils.ProvinceID**: The province ID extracted by `ordered_province` is represented as an instance of `utils.ProvinceID`, which likely contains additional metadata or validation logic related to province IDs in the game environment.

**Usage Notes**
- **Preconditions**: Ensure that the input action(s) are valid and correctly formatted. Invalid inputs may lead to unexpected behavior.
- **Performance Considerations**: The function is designed to be efficient, but performance can vary depending on the size of the input array. For large arrays, consider optimizing or parallelizing the processing if necessary.
- **Edge Cases**:
  - If `PROVINCE_BITS_LOW` and `PROVINCE_BITS_HIGH` are out of bounds for the action's integer representation, the function may return incorrect results.
  - If the input is an empty array or a single invalid action, the function will handle these cases gracefully by returning appropriate values.

**Example Usage**
```python
# Example with a single Action object
action = Action(...)  # Assume this creates a valid Action object
province_id = ordered_province(action)
print(province_id)  # Output: <utils.ProvinceID object at ...>

# Example with an array of Action objects
actions = [Action(...), Action(...), Action(...)]  # List of valid Action objects
province_ids = ordered_province(actions)
print(province_ids)  # Output: [<utils.ProvinceID object at ...>, <utils.ProvinceID object at ...>, <utils.ProvinceID object at ...>]
```

This documentation provides a clear understanding of the `ordered_province` function, its parameters, return values, and usage scenarios. It also highlights key interactions with other components and offers practical advice for effective use in game development contexts.
## FunctionDef shrink_actions(actions)
**Function Overview**
The `shrink_actions` function retains the top and bottom byte pairs of actions, effectively reducing the size of action data while preserving critical information such as indices, order, and unit areas.

**Parameters**
- **actions**: A parameter representing the input action(s) in a format described at the top of this file. This can be an `Action` object, a sequence of `Action` objects, or a NumPy array (`np.ndarray`). The type is specified as `Union[Action, Sequence[Action], np.ndarray]`.

**Return Values**
- **shrunk actions**: A NumPy array containing the shrunk action(s) with only the top and bottom byte pairs retained. If the input array is empty, it returns an empty NumPy array of type `int32`.

**Detailed Explanation**
The function processes the input `actions` to extract specific byte pairs from each element in the sequence or array. Here’s a step-by-step breakdown:

1. **Convert Input to NumPy Array**: The first line converts the input `actions` to a NumPy array if it is not already one.
2. **Check for Empty Input**: If the size of the actions array is zero, an empty NumPy array of type `int32` is returned immediately.
3. **Bitwise Operations**:
   - The bitwise right shift operation (`>> 32`) shifts the bits to the right by 32 positions, effectively moving the lower byte pair (bits 0-15) to the higher bit positions.
   - The bitwise AND operation `& ~0xffff` masks out all but the top 16 bits of the shifted value. This ensures that only the top byte pair is retained.
   - Another bitwise AND operation `& 0xffff` extracts the lower 16 bits (bottom byte pair) from the original action.
4. **Combine Top and Bottom Byte Pairs**: The final line combines the top and bottom byte pairs by adding them together, resulting in a single value that retains both critical pieces of information.

**Interactions with Other Components**
- This function interacts with other parts of the project where actions need to be processed or reduced in size for efficiency. It is typically used in scenarios where detailed action data needs to be summarized without losing essential information.

**Usage Notes**
- **Preconditions**: The input `actions` should be a valid sequence or array as defined by the type hints.
- **Performance Considerations**: While this function is efficient, it may not be suitable for extremely large datasets due to potential memory constraints when converting between different data types and sizes.
- **Edge Cases**:
  - If the input array is empty, an empty NumPy array of `int32` type is returned. This handles cases where no actions are present without errors.
  - The function assumes that each action contains at least two byte pairs (top and bottom), which must be correctly formatted for this operation to work as intended.

**Example Usage**
Here’s a simple example demonstrating the usage of `shrink_actions`:

```python
import numpy as np

# Example actions represented as integers for simplicity
actions = [0x123456789abcdef0, 0x23456789abcdef1, 0x3456789abcdef2]

# Convert the list of actions to a NumPy array
actions_array = np.array(actions)

# Apply the shrink_actions function
shrunk_actions = shrink_actions(actions_array)

print(shrunk_actions)
```

This example converts a list of integer values representing actions into a NumPy array and then applies `shrink_actions` to reduce their size while retaining necessary information.
## FunctionDef find_action_with_area(actions, area)
**Function Overview**
The `find_action_with_area` function searches through a list of actions to find the first action associated with a specific area. It returns the index of this action, or 0 if no such action exists.

**Parameters**

- **actions (Sequence[Union[Action, ActionNoIndex]])**: A sequence containing one or more `Action` objects or `ActionNoIndex` objects. These actions are to be searched for an association with a specific area.
- **area (utils.AreaID)**: An identifier representing the area within which the action is sought.

**Return Values**

The function returns an integer value:

- The index of the first action in the list that corresponds to the specified area, or 0 if no such action exists.

**Detailed Explanation**

1. **Initialization**: 
   - The `province` variable is assigned the province ID corresponding to the given `area`. This is achieved by calling `utils.province_id_and_area_index(area)[0]`.

2. **Iteration and Condition Check**: 
   - A loop iterates over each action in the `actions` sequence.
   - For each action, the function checks if the province extracted from the action using `ordered_province(a)` matches the `province` obtained earlier.

3. **Action Matching**:
   - If a match is found, the index of that action is returned immediately.
   - The loop continues until all actions are checked or an action matching the specified area is found.

4. **No Match Found**:
   - If no action matches the given area, the function returns 0.

**Interactions with Other Components**

- The `find_action_with_area` function interacts with the `ordered_province` function to extract the province ID from each action.
- It also relies on the `province_id_and_area_index` function provided by `utils` to map an area identifier to its corresponding province ID.

**Usage Notes**

- **Preconditions**: 
  - The input `actions` must be a non-empty sequence of valid `Action` or `ActionNoIndex` objects.
  - The `area` parameter should be a valid `AreaID` that corresponds to one of the areas associated with the actions in the list.

- **Performance Considerations**:
  - The function performs a linear search through the `actions` list, making its time complexity O(n), where n is the number of actions.
  - For large lists of actions, this could be inefficient. If performance is critical, consider optimizing by using more efficient data structures or algorithms.

- **Common Pitfalls**:
  - Ensure that all actions in the `actions` sequence are correctly associated with areas to avoid unexpected results.
  - Verify that the area identifier provided matches an existing area in the list of actions.

**Example Usage**

```python
from typing import Sequence, Union

# Sample action and area data
class Action:
    pass

class ActionNoIndex:
    pass

area_id = utils.AreaID("Area1")
actions = [Action(), ActionNoIndex(), Action()]

index = find_action_with_area(actions, area_id)
print(f"Index of the first matching action: {index}")
```

In this example, `find_action_with_area` is called with a list of actions and an area identifier. The function will return the index of the first action that corresponds to "Area1", or 0 if no such action exists in the provided list.
