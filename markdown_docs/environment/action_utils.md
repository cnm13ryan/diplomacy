## FunctionDef bits_between(number, start, end)
**bits_between**: The function of bits_between is to extract a specific bit sequence from an integer.
**parameters**: 
· parameter1: number (int) - The input integer from which bits are extracted.
· parameter2: start (int) - The starting position of the bit sequence to be extracted.
· parameter3: end (int) - The ending position of the bit sequence to be extracted.

**Code Description**: This function is designed to extract a bit sequence from an integer `number` between positions `start` and `end`. It achieves this by using bitwise operations. Specifically, it first masks out all bits in `number` that are not within the range `[start, end]` using `1 << end`, which effectively shifts 1 left by `end` positions to create a bitmask with only the most significant bit (MSB) at position `end`. The modulo operation `%` is then used to retain only these bits. Next, it divides the result by `1 << start` to shift the retained bits so that they are aligned starting from position 0. This effectively isolates and extracts the desired bit sequence.

This function is called in several places within the project, particularly in the `action_breakdown`, `is_waive`, and `ordered_province` functions. For instance, it plays a crucial role in breaking down an action into its component parts by extracting specific coast indicator bits from different regions of the input integer `action`. The `action_breakdown` function uses this to identify the order of the action, as well as the province IDs with their respective coast indicators for different targets. Similarly, `is_waive` and `ordered_province` also leverage this functionality to extract specific information from actions.

**Note**: Ensure that the input integer `number` is valid and within the expected range to avoid unexpected behavior or errors. Additionally, the positions specified by `start` and `end` should be within the bounds of the integer's bit representation (i.e., 0 to 31 for a 32-bit integer).

**Output Example**: If `number = 259`, `start = 2`, and `end = 4`, then `bits_between(number, start, end)` would return `1` because the bits at positions 2 through 4 in binary representation of 259 (which is `100000011`) are `001`.
## FunctionDef actions_for_province(legal_actions, province)
**actions_for_province**: The function of actions_for_province is to filter legal actions based on whether they involve a specific province.

**parameters**:
· parameter1: legal_actions (Sequence[Action]) - A sequence of legal actions from which those involving the specified province are filtered.
· parameter2: province (utils.ProvinceID) - The ID of the province for which relevant actions need to be identified.

**Code Description**: This function iterates over a list of `legal_actions` and checks each action's province using the `ordered_province` function. It returns a new sequence containing only those actions that have their main unit located in the specified `province`.

1. **Initialization**: An empty list named `actions` is initialized to store the filtered actions.
2. **Iteration Over Legal Actions**: The function then iterates over each action in `legal_actions`.
3. **Determining Province of Action**: For each action, it calls the `ordered_province` function to determine the province associated with the action. This step is crucial as it uses bit manipulation to extract the relevant province information.
4. **Filtering Actions**: If an action's province matches the specified `province`, it is appended to the `actions` list.
5. **Return Filtered Actions**: Finally, the function returns the filtered sequence of actions that are relevant to the given `province`.

The relationship with its callees in the project from a functional perspective:
- The `ordered_province` function plays a critical role by extracting province-related information from each action. This extracted information is then used to filter the legal actions based on their association with the specified province.
- By filtering actions, this function helps in scenarios where specific provincial strategies or policies need to be applied.

**Note**: Ensure that all input actions are valid and properly formatted as integers or numpy arrays. The `ORDERED_PROVINCE_START` and `PROVINCE_BITS` constants should be correctly set for the bit positions used by the `ordered_province` function, otherwise, incorrect results might occur.

**Output Example**: If `legal_actions = [102, 153, 204]`, `province = utils.ProvinceID(2)`, and assuming that `action 153` has its main unit in province `2`, the return value would be `[153]`.
## FunctionDef construct_action(order, ordering_province, target_province, third_province)
**construct_action**: The function of construct_action is to create an action representation based on given order details and provinces involved.
**parameters**:
· parameter1: `order` - An instance of the Order class representing the type of action (e.g., build fleet, move to, etc.).
· parameter2: `ordering_province` - A ProvinceWithFlag object indicating the province from which an order is being made or the province that has ordered.
· parameter3: `target_province` - A ProvinceWithFlag object indicating the target province for the action (e.g., where a fleet will move to).
· parameter4: `third_province` - An optional ProvinceWithFlag object, used in certain scenarios like building fleets.

**Code Description**: The function constructs an integer value (`order_rep`) that represents the action based on the provided parameters. This representation is a bitmask where different bits are set according to the type of order and the provinces involved.
1. **Initialization**: `order_rep` starts as 0, which will be used to build up the final action representation.
2. **Order Bitmasking**:
   - The `order` parameter determines the initial part of the bitmask (`order_rep |= order << ACTION_ORDER_START`). This shifts the value of `order` by a certain number of bits starting from `ACTION_ORDER_START`, effectively setting these bits in `order_rep`.
3. **Ordering Province Bitmasking**:
   - If an ordering province is provided, its first element (likely representing the province ID) is used to set additional bits (`order_rep |= ordering_province[0] << ACTION_ORDERED_PROVINCE_START`).
   - For certain orders like `BUILD_FLEET`, the second element of `ordering_province` may specify if it's a coastal or inland action.
4. **Target Province Bitmasking**:
   - If a target province is provided, its first element sets more bits (`order_rep |= target_province[0] << ACTION_TARGET_PROVINCE_START`).
   - For actions like `MOVE_TO` or `RETREAT_TO`, the second element of `target_province` specifies the coast.
5. **Third Province Bitmasking**:
   - If a third province is provided, its first element sets additional bits (`order_rep |= third_province[0] << ACTION_THIRD_PROVINCE_START`).
6. **Return Value**: The function returns the constructed bitmask value `order_rep`, which encapsulates all the necessary information about the action.

**Note**: Ensure that the provinces and orders are correctly formatted according to their respective classes (Order and ProvinceWithFlag) to avoid errors in bitmasking.
**Output Example**: If `order` is `BUILD_FLEET`, `ordering_province` is a coastal province, and there's no target or third province, the output might be `0b101000` (assuming `ACTION_ORDER_START` is 4 bits and `ACTION_ORDERED_PROVINCE_COAST` is another 2 bits).
## FunctionDef action_breakdown(action)
**action_breakdown**: The function of action_breakdown is to break down an action into its component parts.
· parameter1: action (Union[Action, ActionNoIndex]) - A 32-bit or 64-bit integer representing an action.
**Code Description**: This function decomposes a given action into four main components: the order and three coast indicator bits of provinces. Here's how it works:

1. **Order Extraction**: The first step is to extract the order from the input `action`. This is done by calling the `bits_between` function with parameters `ACTION_ORDER_START` and `ACTION_ORDER_START + ACTION_ORDER_BITS`, which isolates a specific bit sequence representing the order of the action.
2. **Province 1 (p1) Extraction**: Next, the function extracts the province ID and its coast indicator for the first target by calling `bits_between` twice:
   - The first call is with parameters `ACTION_ORDERED_PROVINCE_START` and `ACTION_ORDERED_PROVINCE_START + ACTION_PROVINCE_BITS`, which isolates the province ID bits.
   - The second call is to extract the coast indicator bit for this province, using `ACTION_ORDERED_PROVINCE_COAST`.
3. **Province 2 (p2) Extraction**: Similarly, the function extracts the province ID and its coast indicator for the second target:
   - The first call uses parameters `ACTION_TARGET_PROVINCE_START` and `ACTION_TARGET_PROVINCE_START + ACTION_PROVINCE_BITS`, which isolates the province ID bits.
   - The second call is to extract the coast indicator bit, using `ACTION_TARGET_PROVINENCE_COAST`.
4. **Province 3 (p3) Extraction**: Finally, the function extracts the province ID and its coast indicator for the third target:
   - The first call uses parameters `ACTION_THIRD_PROVINCE_START` and `ACTION_THIRD_PROVINCE_START + ACTION_PROVINCE_BITS`, which isolates the province ID bits.
   - The second call is to extract the coast indicator bit, using `ACTION_THIRD_PROVINCE_COAST`.

The function returns a tuple containing the order and the three extracted provinces as `(order, p1, p2, p3)`. Each element of this tuple represents the breakdown of the action into its component parts.

**Note**: The coast indicator bits returned by this function are not area_ids as returned by `province_id` and `area`. Ensure that the input `action` is a valid 32-bit or 64-bit integer to avoid unexpected behavior. Additionally, the positions specified for extracting bits should be within the bounds of the integer's bit representation.

**Output Example**: If `action = 1075`, then `action_breakdown(action)` might return `(3, (10, True), (25, False), (40, True))`. Here, `order` is `3`, and provinces `p1`, `p2`, and `p3` are represented by their province IDs and coast indicators.
## FunctionDef action_index(action)
**action_index**: The function of action_index is to return the index of actions among all possible unit actions.
**parameters**: 
· parameter1: action (Union[Action, np.ndarray]) - This parameter can be either an instance of Action or a NumPy array containing action data.

**Code Description**: The function `action_index` takes an input `action`, which can be either a single action object or an array of actions. It then returns the index of this action among all possible unit actions. The operation `>> ACTION_INDEX_START` is used to shift the value, implying that it might be part of a bit manipulation process where `ACTION_INDEX_START` is likely a predefined constant representing the starting point for indexing.

The function utilizes type hinting with `Union[Action, np.ndarray]`, indicating flexibility in handling both individual action objects and arrays of actions. The return type is also annotated as `Union[int, np.ndarray]`, suggesting that the output can be either an integer or a NumPy array depending on the input context.

**Note**: Ensure that the constant `ACTION_INDEX_START` is defined elsewhere in your codebase for this function to work correctly. Be mindful of potential edge cases where the input might not align with expected data types, and handle such scenarios appropriately.

**Output Example**: If `action` is an instance of Action corresponding to the 5th possible unit action (assuming indexing starts from 0), and `ACTION_INDEX_START` is set to 10, then `action_index(action)` would return 14. If `action` is a NumPy array with multiple actions, it could return a NumPy array containing their respective indices.
## FunctionDef is_waive(action)
**is_waive**: The function of is_waive is to determine whether an action is marked as waive based on its order bits.
**parameters**: 
· parameter1: action (Union[Action, ActionNoIndex]) - The input action or no index action from which the waiver status needs to be determined.

**Code Description**: This function checks if a given action should be considered waived by examining specific bits within an integer representation of the action. Here’s a detailed breakdown:

1. **Parameter Handling**: The function receives `action`, which can be either an instance of `Action` or `ActionNoIndex`. These types are assumed to have an integer representation where certain bits hold information about the action's properties.

2. **Order Extraction**: The function calls `bits_between(action, ACTION_ORDER_START, ACTION_ORDER_START + ACTION_ORDER_BITS)`. This call extracts a specific bit sequence from the `action` using the predefined constants `ACTION_ORDER_START` and `ACTION_ORDER_BITS`. These constants define the starting position and length of the order bits within the integer representation.

3. **Comparison**: The extracted bit sequence is then compared to the constant `WAIVE`, which represents the binary value indicating a waive status (e.g., 1).

4. **Return Value**: If the extracted bit sequence matches `WAIVE`, the function returns `True`, signifying that the action is marked as waived. Otherwise, it returns `False`.

**Note**: Ensure that the input `action` is valid and correctly formatted to avoid incorrect results. The positions specified by `ACTION_ORDER_START` and `ACTION_ORDER_BITS` should be accurate to ensure correct bit extraction.

**Output Example**: If `action` has an integer value where bits between `ACTION_ORDER_START` and `ACTION_ORDER_START + ACTION_ORDER_BITS` are set to 1 (representing the `WAIVE` status), then `is_waive(action)` will return `True`. Otherwise, it returns `False`.
## FunctionDef ordered_province(action)
**ordered_province**: The function of ordered_province is to extract province-related information from an action based on specific bit positions.

**parameters**:
· parameter1: action (Union[Action, ActionNoIndex, np.ndarray]) - The input action or actions from which the province information is extracted.
 
**Code Description**: This function utilizes the `bits_between` method to isolate a particular segment of bits within an integer representing an action. Specifically, it extracts bits starting from `ACTION_ORDERED_PROVINCE_START` and ending at `ACTION_ORDERED_PROVINCE_START + ACTION_PROVINCE_BITS`. These bit positions are defined constants that indicate where province-related information is stored in the action's binary representation.

The function plays a crucial role in breaking down actions to identify which provinces an action pertains to. By extracting this specific segment of bits, it helps determine the province associated with the given action. This extracted information can then be used by other functions such as `actions_for_province` and `find_action_with_area`, which rely on the province information to filter or locate relevant actions.

For example, in the function `actions_for_province`, ordered_province is called to identify all legal actions that involve a specific province. Similarly, in `find_action_with_area`, ordered_province helps find the first action for a unit within a given area by determining the province associated with each action.

**Note**: Ensure that the input action is valid and properly formatted as an integer or numpy array. The bit positions specified should be within the expected range to avoid errors. Incorrect input can lead to unexpected behavior or incorrect results.

**Output Example**: If `action = 259` (binary: `100000011`) and `ACTION_ORDERED_PROVINCE_START = 2`, `ACTION_PROVINCE_BITS = 3`, then `ordered_province(action)` would return the province ID represented by bits at positions 2 through 4, which is `1` in this case.
## FunctionDef shrink_actions(actions)
**shrink_actions**: The function of shrink_actions is to retain the top and bottom byte pairs of actions, which contain the index, order, and ordered unit's area.

**Parameters**:
· parameter1: actions (Union[Action, Sequence[Action], np.ndarray])
    - This parameter accepts actions in various formats: a single Action object, a sequence of Action objects, or an array. The input can be either a scalar or an array-like structure.

**Code Description**: 
The function `shrink_actions` processes the provided actions and returns shrunk versions of them by retaining specific byte pairs from each action. Here is a detailed analysis:

1. **Input Handling**: The first line converts the input `actions` to a NumPy array using `np.asarray(actions)`. This ensures that the function can handle both single Action objects and sequences/arrays of such objects uniformly.
2. **Empty Array Check**: If the size of the actions array is zero, meaning no valid actions are provided, the function returns an empty array of type `int32` with `return actions.astype(np.int32)`.
3. **Bitwise Operations for Shrinking Actions**:
    - The second line performs a bitwise right shift operation on each element in the actions array by 32 bits: `(actions >> 32)`. This effectively moves the lower 32 bits (bottom byte pairs) to the higher positions.
    - The third line applies a bitmask `~0xffff` to the result of the previous step, which masks out all but the bottom 16 bits. This retains only the least significant bytes that contain relevant information such as the order and area.
    - The fourth line performs another bitwise AND operation with the original actions array (`actions & 0xffff`), masking out everything except the top 16 bits (top byte pairs).
    - Finally, these two results are added together: `((actions >> 32) & ~0xffff) + (actions & 0xffff)` to produce the shrunk action.

**Note**: 
- Ensure that the input actions are compatible with the expected format and type.
- The function assumes that the top and bottom byte pairs contain meaningful information for the index, order, and area. Any other data in these bytes will be lost during the shrinking process.

**Output Example**: 
If the input `actions` is an array `[1234567890, 2345678901]`, the output might look like:
```python
array([12345678, 23456789], dtype=int32)
```
This example shows that the function retains the top and bottom byte pairs from each action, resulting in a shrunk array.
## FunctionDef find_action_with_area(actions, area)
**find_action_with_area**: The function of find_action_with_area is to locate the first action associated with a unit within a specified area.
**parameters**:
· parameter1: actions (Sequence[Union[Action, ActionNoIndex]]) - A sequence of actions or action objects from which the relevant action for the given area is selected.
· parameter2: area (utils.AreaID) - The identifier for the area in which to find the first action.

**Code Description**: This function iterates through a list of actions to identify and return the first action that pertains to a unit within a specified area. It uses the `ordered_province` function from the same module to extract province-related information from each action, comparing it with the given area's province identifier. The process involves:
1. Extracting the province ID for each action using `ordered_province`.
2. Comparing the extracted province ID against the province of the specified area.
3. Returning the first action that matches the province of the specified area.

If no matching action is found, it returns 0 as an integer value indicating the absence of a relevant action.

The function plays a crucial role in filtering actions based on their geographical relevance, allowing for precise targeting within specific areas. This functionality is essential for operations where actions need to be localized or prioritized according to administrative divisions like provinces.

**Note**: Ensure that the input `actions` sequence contains valid and properly formatted action objects or arrays. The province information should be correctly encoded in these actions to avoid unexpected behavior or incorrect results.

**Output Example**: If the list of actions is `[Action1, Action2, Action3]` and the specified area has a province ID of 5, where `Action2`'s province ID matches this value, then `find_action_with_area(actions, AreaID(5))` would return the index of `Action2`. If no action matches the province ID, it returns 0.
