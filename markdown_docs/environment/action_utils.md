## FunctionDef bits_between(number, start, end)
**bits_between**: The function of bits_between is to extract a specific range of bits from an integer.

parameters: 
· number: An integer from which bits are to be extracted.
· start: The starting position (inclusive) of the bit range to extract.
· end: The ending position (exclusive) of the bit range to extract.

Code Description: The function `bits_between` is designed to isolate and return a specific sequence of bits from an integer. It achieves this by first using the modulo operation with `(1 << end)` to mask out all bits beyond the specified end position, effectively keeping only the lower `end` bits of the number. Then, it performs an integer division by `(1 << start)` to shift these remaining bits right by `start` positions, thus aligning the desired bit range at the least significant bit (LSB) position. This process ensures that the function returns only the bits between the specified `start` and `end` positions.

In the context of the project, this function is utilized in several other functions to extract specific parts of an action integer. For instance, in `action_breakdown`, it is used multiple times to extract different components such as order type and province identifiers from a 32-bit or 64-bit action integer. Similarly, in `is_waive`, it extracts the order component to determine if the action represents a waive command. Lastly, in `ordered_province`, it isolates the bits corresponding to the ordered province identifier.

Note: The function assumes that the input number is non-negative and that the start and end parameters are valid (i.e., 0 <= start < end <= bit length of the integer).

Output Example: If `number` is 29 (binary '11101'), `start` is 1, and `end` is 4, the function will return 7 (binary '111') because it extracts bits from position 1 to 3 ('111').
## FunctionDef actions_for_province(legal_actions, province)
**actions_for_province**: The function of actions_for_province is to filter and return all actions from a given list that have their main unit located in a specified province.

parameters: 
· legal_actions: A sequence (e.g., list or tuple) of Action objects representing the set of possible actions.
· province: An instance of ProvinceID indicating the specific province for which relevant actions are to be identified.

Code Description: The function `actions_for_province` iterates over each action in the provided `legal_actions` sequence. For each action, it uses the `ordered_province` helper function to extract the province identifier associated with the main unit of that action. If the extracted province matches the specified `province`, the action is added to a new list called `actions`. After processing all actions, the function returns this list containing only those actions whose main units are located in the specified province.

In the context of the project, this function is crucial for scenarios where actions need to be filtered based on their geographical location within a specific province. It leverages the `ordered_province` function to accurately determine the province associated with each action, ensuring that only relevant actions are included in the final output.

Note: The function assumes that all actions in the `legal_actions` sequence are valid and that the `province` parameter is a correctly formatted ProvinceID object. Additionally, it relies on the proper implementation of the `ordered_province` function to accurately extract province identifiers from actions.

Output Example: If `legal_actions` contains three actions with main units located in provinces 10, 20, and 30 respectively, and if `province` is set to 20, the function will return a list containing only the action associated with province 20.
## FunctionDef construct_action(order, ordering_province, target_province, third_province)
**construct_action**: The function of construct_action is to build an action representation based on the given order and provinces involved.

parameters:
· order: An instance of Order representing the type of action (e.g., BUILD_FLEET, MOVE_TO, RETREAT_TO).
· ordering_province: An instance of ProvinceWithFlag indicating the province where the order originates from.
· target_province: An instance of ProvinceWithFlag specifying the destination province for actions like move or retreat.
· third_province: An instance of ProvinceWithFlag used in certain scenarios to specify an additional province (e.g., convoy routes).

Code Description:
The function constructs a bitwise representation of an action by combining different components into a single integer. It starts with initializing order_rep to 0 and then shifts the order value left by ACTION_ORDER_START bits before OR-ing it with order_rep. If ordering_province is provided, its province ID is shifted left by ACTION_ORDERED_PROVINCE_START bits and combined with order_rep. For build fleet orders, an additional bit representing the coast flag of the ordering province is included. Similarly, if a target_province is specified, its province ID is shifted left by ACTION_TARGET_PROVINCE_START bits and added to order_rep. For move or retreat actions, the coast flag of the target province is also considered. If a third_province is given, it is shifted left by ACTION_THIRD_PROVINCE_START bits and included in the final action representation.

Note: The function assumes that the constants ACTION_ORDER_START, ACTION_ORDERED_PROVINCE_START, ACTION_ORDERED_PROVINCE_COAST, ACTION_TARGET_PROVINCE_START, ACTION_TARGET_PROVINCE_COAST, and ACTION_THIRD_PROVINCE_START are defined elsewhere in the codebase. Additionally, it is important to ensure that the provided provinces have valid IDs and coast flags.

Output Example: A possible return value of this function could be 123456789, which represents a specific combination of order type, originating province, target province, and any additional province involved in the action. The exact numerical value would depend on the specific values of the parameters passed to the function.
## FunctionDef action_breakdown(action)
**action_breakdown**: The function of action_breakdown is to decompose an action into its constituent parts, specifically extracting the order type and three province identifiers along with their respective coast indicator bits.

parameters: 
· action: A 32-bit or 64-bit integer representing the action to be broken down.

Code Description: The function `action_breakdown` is designed to dissect a given action integer into its fundamental components. It utilizes the helper function `bits_between` to extract specific segments of bits from the action integer. The process begins by extracting the order type, which is an integer between 1 and 13, using predefined constants `ACTION_ORDER_START` and `ACTION_ORDER_BITS`. Following this, it extracts three pairs of province identifiers and their corresponding coast indicator bits. Each pair consists of a province identifier extracted using `ACTION_ORDERED_PROVINCE_START`, `ACTION_TARGET_PROVINCE_START`, or `ACTION_THIRD_PROVINCE_START` along with the respective bit lengths defined by `ACTION_PROVINCE_BITS`. The coast indicator bit for each province is obtained using `ACTION_ORDERED_PROVINCE_COAST`, `ACTION_TARGET_PROVINCE_COAST`, and `ACTION_THIRD_PROVINCE_COAST` constants. These bits are then returned as a tuple containing the order type and three tuples, each representing a province identifier and its coast indicator.

Note: The coast indicator bits extracted by this function are not area_ids as returned by province_id and area functions. It is crucial to understand that these bits serve a different purpose within the context of the action breakdown process.

Output Example: If the input `action` integer is such that it represents an order type of 5, with ordered province identifier 123 and coast indicator bit 0, target province identifier 456 and coast indicator bit 1, and third province identifier 789 and coast indicator bit 0, the function will return (5, (123, 0), (456, 1), (789, 0)).
## FunctionDef action_index(action)
**action_index**: The function of action_index is to return the index of an action among all possible unit actions.
parameters: 
· action: This parameter can be either an instance of the Action class or a numpy ndarray. It represents the action for which the index is being sought.

Code Description: The function action_index takes an input 'action' which can be either an object of type Action or a numpy array. It performs a bitwise right shift operation on this input by ACTION_INDEX_START bits and returns the result. This operation effectively extracts the index of the action from the provided action representation, assuming that the index is encoded in the higher bits of the action value.

Note: The function assumes that the constant ACTION_INDEX_START is defined elsewhere in the codebase and represents the number of bits to shift right to isolate the action index. Users must ensure that the input 'action' is correctly formatted as either an Action object or a numpy array, and that it contains the necessary information encoded in a way compatible with the bitwise operation.

Output Example: If ACTION_INDEX_START is 4 and the input action value (in binary) is 101000 (which is 40 in decimal), the function will return 2 (binary 10), as the right shift by 4 bits removes the lower 4 bits, leaving only the index.
## FunctionDef is_waive(action)
**is_waive**: The function of is_waive is to determine if a given action represents a waive command by examining its order component.
parameters: 
· action: An instance of either Action or ActionNoIndex, representing the action to be evaluated.

Code Description: The function `is_waive` is designed to ascertain whether a specific action corresponds to a waive command. It achieves this by utilizing the `bits_between` function to extract a particular segment of bits from the action integer that represents the order type. This bit range is defined by the constants `ACTION_ORDER_START` and `ACTION_ORDER_BITS`, which specify the starting position and the number of bits in the order component, respectively. The extracted order value is then compared against the constant `WAIVE`. If they match, the function returns `True`, indicating that the action is a waive command; otherwise, it returns `False`.

In the context of the project, this function plays a crucial role in interpreting actions by focusing on their order type. It leverages the `bits_between` function to isolate and evaluate the relevant portion of the action integer, ensuring accurate determination of whether an action constitutes a waive command.

Note: The function assumes that the input action is valid and that the constants `ACTION_ORDER_START`, `ACTION_ORDER_BITS`, and `WAIVE` are correctly defined in the project's configuration. It is essential to ensure these constants align with the bit structure used for encoding actions within the system.

Output Example: If the action integer has its order component set to the value of `WAIVE`, the function will return `True`. For instance, if `ACTION_ORDER_START` is 0 and `ACTION_ORDER_BITS` is 3, and the binary representation of the action integer's relevant bits is '101', which matches the binary representation of `WAIVE`, then `is_waive(action)` will return `True`. Conversely, if the order component does not match `WAIVE`, the function will return `False`.
## FunctionDef ordered_province(action)
**ordered_province**: The function of ordered_province is to extract the province identifier from an action.

parameters: 
· action: An instance of Action, ActionNoIndex, or np.ndarray representing the action from which the province identifier needs to be extracted.

Code Description: The function `ordered_province` utilizes the helper function `bits_between` to isolate and return a specific sequence of bits from the provided action. This bit range corresponds to the ordered province identifier embedded within the action. The extraction process is defined by two constants, `ACTION_ORDERED_PROVINCE_START` and `ACTION_PROVINCE_BITS`, which specify the starting position and the number of bits that represent the province identifier, respectively.

In the context of the project, this function plays a crucial role in several other functions where actions need to be filtered or analyzed based on their associated provinces. For example, in `actions_for_province`, it is used to identify all actions within a list that are relevant to a specific province by comparing the extracted province identifier with the target province. Similarly, in `find_action_with_area`, it helps locate the first action in a list that pertains to a unit in a given area by matching the extracted province identifier with the province derived from the area.

Note: The function assumes that the input action is valid and that the constants `ACTION_ORDERED_PROVINCE_START` and `ACTION_PROVINCE_BITS` are correctly defined to reflect the structure of the action integer.

Output Example: If the action integer is 123456 (binary '11110001110010000'), and the constants are set such that `ACTION_ORDERED_PROVINCE_START` is 8 and `ACTION_PROVINCE_BITS` is 5, the function will return 19 (binary '10011') because it extracts bits from position 8 to 12 ('10011').
## FunctionDef shrink_actions(actions)
**shrink_actions**: The function of shrink_actions is to retain only the top and bottom byte pairs of actions.

parameters:
· actions: action(s) provided in the format described at the top of this file. This can be a single Action, a sequence of Actions, or a numpy ndarray.

Code Description: 
The shrink_actions function processes input actions by first converting them into a numpy array if they are not already in that format. It then checks if the size of the actions array is zero, and if so, returns an empty array cast to int32 type. If there are actions present, it performs bitwise operations on each action element to extract specific byte pairs. The operation `(actions >> 32) & ~0xffff` shifts the bits of each action 32 places to the right and then masks out the lower 16 bits, effectively retaining only the top byte pair. The operation `actions & 0xffff` retains only the bottom byte pair of each action. These two results are added together to form the "shrunk" actions, which are then cast to int32 type before being returned.

Note: This function assumes that the input actions are encoded in a specific way where the top and bottom byte pairs contain meaningful information such as index, order, and area of an ordered unit. The bitwise operations used here are crucial for extracting this information correctly.

Output Example: 
If the input actions were `[0x123456789ABCDEF0, 0xFEDCBA9876543210]`, the output would be a numpy array of `[-2128831032, -2128831032]` after performing the bitwise operations and casting to int32. This example assumes that the top and bottom byte pairs are combined in such a way that they result in these specific integer values when processed by the function.
## FunctionDef find_action_with_area(actions, area)
**find_action_with_area**: The function of find_action_with_area is to locate the first action in a list that pertains to a unit in a specified area.

parameters: 
· actions: A sequence (list or tuple) of Action or ActionNoIndex objects representing the actions to be searched.
· area: An AreaID object representing the area for which an associated action is sought.

Code Description: The function `find_action_with_area` iterates over a list of actions and checks each one to determine if it corresponds to a unit in the specified area. It achieves this by extracting the province identifier from each action using the `ordered_province` function and comparing it with the province derived from the given area. If a match is found, the corresponding action is returned immediately. If no matching action is found after examining all actions in the list, the function returns 0.

The relationship with its callees in the project from a functional perspective is that `find_action_with_area` relies on `ordered_province` to extract the province identifier from each action. This extracted identifier is then used to compare against the province derived from the area parameter, allowing the function to identify actions relevant to the specified area.

Note: The function assumes that the input actions are valid and that the `ordered_province` function correctly extracts the province identifier from each action. Additionally, it returns 0 if no matching action is found, which may need to be handled appropriately by the caller of this function.

Output Example: If the list of actions contains an action with a province identifier that matches the province derived from the area parameter, that action will be returned. For example, if the actions list includes an action with a province identifier of 5 and the area corresponds to province 5, then that action would be returned. If no such action exists in the list, the function returns 0.
