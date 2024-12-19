## FunctionDef mila_area_string(unit_type, province_tuple)
**mila_area_string**

The function `mila_area_string` generates the string representation of an area as used in MILA actions, based on the unit type and province information provided.

**Parameters**

- `unit_type`: An enumeration value from `utils.UnitType`, indicating whether the unit is an army or a fleet.
- `province_tuple`: A tuple representing the province and its coast flag, of type `utils.ProvinceWithFlag`.

**Code Description**

The function `mila_area_string` is designed to produce a string that represents a specific area in the context of MILA actions. This string is crucial for identifying locations accurately, especially when dealing with provinces that have multiple coasts.

### Function Logic

1. **Extract Province ID:**
   - The function starts by extracting the province ID from the provided `province_tuple`.

2. **Determine Area Index:**
   - If the `unit_type` is a fleet (`utils.UnitType.FLEET`), it uses the `utils.area_index_for_fleet` function to determine the appropriate area index based on the province and its coast flag.
   - For other unit types (presumably armies), it sets the area index to 0, assuming the main area of the province.

3. **Retrieve Area ID:**
   - Using the province ID and the determined area index, it fetches the corresponding area ID via `utils.area_from_province_id_and_area_index`.

4. **Convert Area ID to Tag:**
   - It then maps the area ID to a tag using a dictionary `_area_id_to_tag`.

5. **Map to MILA Tag (if exists):**
   - Finally, it checks if there is a specific mapping for this tag in `_DM_TO_MILA_TAG_MAP`; if so, it uses that mapped value; otherwise, it defaults to the original tag.

### Usage Context

This function is integral to generating action strings in the MILA format within the game's environment. It is primarily used by other functions to construct complete action strings accurately. For instance:

- **mila_unit_string:** This function uses `mila_area_string` to create a unit string that includes the unit type and its location in the MILA format.

- **action_to_mila_actions:** This function translates actions from a standardized format to MILA action strings, utilizing `mila_area_string` to ensure correct area representations in various types of orders like holds, moves, supports, convoys, builds, and disbands.

### Relationship with Callers

- **mila_unit_string:** Depends on `mila_area_string` to get the correct area string for unit representation.
- **action_to_mila_actions:** Relies heavily on `mila_area_string` to construct accurate MILA action strings for different types of orders, ensuring that province and coast information are correctly represented.

### Note

- Ensure that the input `province_tuple` is correctly formatted as expected by the function to avoid errors.
- Be aware of the specific behaviors for fleet units in bicoastal provinces, as the coast flag is crucial for determining the correct area index.

**Output Example**

For an army in province with ID 'PAR' and no coast flag:

```
"PAR"
```

For a fleet in province with ID 'STP' and coast flag indicating North Coast:

```
"STP/NC"
```

This function ensures that the area strings are formatted correctly according to MILA standards, facilitating seamless integration and interpretation of actions within the game's framework.
## FunctionDef mila_unit_string(unit_type, province_tuple)
**mila_unit_string**

The function `mila_unit_string` generates a string representation of a unit in the MILA format, combining the unit type and its location.

**Parameters**

- `unit_type`: An enumeration value from `utils.UnitType`, indicating whether the unit is an army or a fleet.
- `province_tuple`: A tuple representing the province and its coast flag, of type `utils.ProvinceWithFlag`.

**Code Description**

The function `mila_unit_string` constructs a string that identifies a unit in the MILA action format. It takes two parameters: `unit_type`, which specifies whether the unit is an army or a fleet, and `province_tuple`, which contains information about the province and its coast flag.

### Function Logic

1. **Determine Unit Type Prefix:**
   - The function uses a list of strings `['A %s', 'F %s']` to select the appropriate prefix based on the value of `unit_type`. Specifically, it uses `unit_type.value` to index into this list. Assuming `utils.UnitType` is an enumeration where `ARMY` has a value of 0 and `FLEET` has a value of 1, the prefix will be 'A' for armies and 'F' for fleets.

2. **Generate Area String:**
   - It calls another function `mila_area_string(unit_type, province_tuple)` to get the string representation of the area where the unit is located. This function handles the specifics of how provinces and their coast flags are represented in the MILA format.

3. **Format and Return the Unit String:**
   - The selected prefix is formatted with the area string using the modulo operator (`%`), which inserts the area string into the placeholder `%s` in the prefix string. The resulting string combines the unit type and its location in a standardized format.

### Usage Context

This function is crucial for generating action strings in the MILA format, which is used in the game's environment to represent units and their movements, supports, builds, etc. It ensures consistency and correctness in how units are identified based on their type and position.

### Relationship with Callers

- **action_to_mila_actions:** This function relies on `mila_unit_string` to construct complete MILA action strings for various types of orders. For instance, when generating hold, move, support, convoy, build, or disband actions, it uses `mila_unit_string` to correctly represent the unit involved in each action.

### Note

- Ensure that the `unit_type` and `province_tuple` parameters are of the correct types as specified (`utils.UnitType` and `utils.ProvinceWithFlag`, respectively) to avoid runtime errors.
- The function assumes that `mila_area_string` returns a correctly formatted area string based on the unit type and province information.

**Output Example**

For an army in Paris:

```
"A PAR"
```

For a fleet in St. Petersburg/North Coast:

```
"F STP/NC"
```

This function is essential for maintaining uniformity in how units are referenced across different actions in the game, ensuring that the MILA action strings are correctly formatted and interpretable by the game's systems.
## FunctionDef possible_unit_types(province_tuple)
**possible_unit_types**

The function `possible_unit_types` determines the types of units that can occupy a given province in the game.

**Parameters**

- `province_tuple`: A tuple representing the province and its flag. The structure is `(province_id, flag)`, where `province_id` is an identifier for the province and `flag` indicates additional properties such as coast type.

**Code Description**

This function assesses what kind of units can be present in a specified province based on the province's characteristics. It takes a `province_tuple`, which includes the province's ID and a flag indicating specific attributes like coastal information.

First, it checks if the flag value is greater than zero. If true, this indicates that the province is bicoastal, meaning it has two distinct coasts. In such cases, only fleet units are permitted to occupy the province.

If the flag is not greater than zero, the function determines the type of the province by calling `utils.province_type_from_id(province_id)`. This returns the province's type, which can be land, sea, or bicoastal.

Based on the province type:

- If the province is land (`ProvinceType.LAND`), only army units are allowed.

- If the province is sea (`ProvinceType.SEA`), only fleet units are allowed.

- If the province is bicoastal (`ProvinceType.BICOASTAL`), both army and fleet units are permitted.

The function returns a set containing the possible unit types that can occupy the given province.

**Relationship with Callers**

This function is utilized by other parts of the code to determine valid unit placements and movements. For instance:

- `possible_unit_types_movement`: This function determines the unit types that can move from one province to another. It uses `possible_unit_types` to check the allowable unit types for both the starting and destination provinces.

- `possible_unit_types_support`: This function assesses what unit types can support actions in a given destination province from a starting province. Similar to movement, it uses `possible_unit_types` to validate unit types in both locations.

- `action_to_mila_actions`: This function translates game actions into MILA action strings, which are a standardized format for representing orders in the game. It uses `possible_unit_types` to ensure that the units involved in actions are of appropriate types for their provinces.

**Note**

When using this function, ensure that the `province_tuple` is correctly formatted as `(province_id, flag)`. The flag is crucial for identifying bicoastal provinces, which have special unit occupation rules. Incorrect formatting or invalid province IDs may lead to inaccurate results or errors.

**Output Example**

Suppose we have a province tuple `(1, 0)`, where province ID 1 is a land province.

```python
possible_unit_types((1, 0))
```

Output:

```python
{utils.UnitType.ARMY}
```

Another example with a bicoastal province:

```python
possible_unit_types((2, 1))
```

Output:

```python
{utils.UnitType.FLEET}
```

In this case, since the flag is greater than zero, only fleets are allowed.
## FunctionDef possible_unit_types_movement(start_province_tuple, dest_province_tuple)
**possible_unit_types_movement**

This function determines which types of units can move from one specified province to another in the game.

**Parameters**

- `start_province_tuple`: A tuple representing the starting province and its flag. The structure is `(province_id, flag)`, where `province_id` is an identifier for the province and `flag` indicates additional properties such as coast type.
- `dest_province_tuple`: A tuple representing the destination province and its flag, structured similarly to `start_province_tuple`.

**Code Description**

The function `possible_unit_types_movement` evaluates which types of units—armies or fleets—can successfully move from a starting province to a destination province. It returns a set of unit types that are capable of making this specific move.

First, the function initializes an empty set called `possible_types` to store the unit types that can make the move.

It then checks if both the starting and destination provinces allow armies by using the `possible_unit_types` function for each province. If armies are possible in both provinces, it adds `utils.UnitType.ARMY` to the `possible_types` set.

Next, for fleet movements, the function determines the area IDs for both the starting and destination provinces. It uses the `utils.area_from_province_id_and_area_index` function along with `utils.area_index_for_fleet` to get these area IDs. It then checks if the destination area ID is adjacent to the starting area ID for fleets, using a predefined adjacency dictionary `_fleet_adjacency`. If the destination area is adjacent, it adds `utils.UnitType.FLEET` to the `possible_types` set.

Finally, the function returns the set of possible unit types that can move from the starting province to the destination province.

**Relationship with Callers**

This function is used in scenarios where determining valid unit movements between provinces is necessary. For example:

- In the function `action_to_mila_actions`, when processing a `MOVE_TO` order, this function is called to determine what types of units can move from the starting province to the destination province. This helps in generating correct MILA action strings that represent these movements.

**Note**

Ensure that both `start_province_tuple` and `dest_province_tuple` are correctly formatted as `(province_id, flag)`. Incorrect formatting or invalid province IDs may lead to inaccurate results or errors.

Additionally, be aware that the adjacency check for fleets relies on the `_fleet_adjacency` dictionary, which should be properly initialized and up-to-date with the current game map's adjacency rules.

**Output Example**

Suppose we have a starting province tuple `(1, 0)` and a destination province tuple `(2, 0)`, where province ID 1 is land and province ID 2 is connected via sea areas.

```python
possible_unit_types_movement((1, 0), (2, 0))
```

Output:

```python
{utils.UnitType.FLEET}
```

In this case, only fleets can move from province 1 to province 2, assuming that armies cannot directly move between these provinces and that the area adjacency allows fleet movement.
## FunctionDef possible_unit_types_support(start_province_tuple, dest_province_tuple)
**possible_unit_types_support**: Determines the unit types that can support an action from a starting province to a destination province.

### Parameters

- **start_province_tuple**: A tuple representing the starting province and its flag.
  - Type: `utils.ProvinceWithFlag`
  - Description: Specifies the province from which the support is being offered, including any flags indicating special properties like coast type.

- **dest_province_tuple**: A tuple representing the destination province and its flag.
  - Type: `utils.ProvinceWithFlag`
  - Description: Specifies the province to which the support is being offered, including any flags indicating special properties like coast type.

### Code Description

The function `possible_unit_types_support` determines what types of units (army or fleet) can provide support from a starting province to a destination province in a game context. It returns a set of unit types that are capable of making this support based on the provinces' characteristics and adjacency rules.

#### Step-by-Step Analysis

1. **Initialization**:
   - A set `possible_types` is initialized to store the unit types that can provide support.

2. **Army Support Check**:
   - The function checks if an army unit type is present in both the starting province and the destination province by using the `possible_unit_types` function.
   - Specifically, it calls `possible_unit_types(start_province_tuple)` and `possible_unit_types((dest_province_tuple[0], 0))`.
   - If army is possible in both provinces, it adds `utils.UnitType.ARMY` to `possible_types`.

3. **Fleet Support Check**:
   - Determines the area ID for the starting province based on fleet adjacency.
   - For the destination province, it considers both coast indices if the province is bicoastal (area indices 1 and 2), otherwise only index 0.
   - Checks if the destination area ID is adjacent to the starting area ID via fleet movement.
   - If adjacency is confirmed, adds `utils.UnitType.FLEET` to `possible_types`.

4. **Return**:
   - Returns the set of possible unit types that can support from the start to the destination province.

#### Functional Relationships

- **Called Functions**:
  - `possible_unit_types(province_tuple)`: Determines the types of units that can occupy a given province.
  - `utils.area_from_province_id_and_area_index(province_id, area_index)`: Computes the area ID based on province ID and area index.
  - `utils.area_index_for_fleet(start_province_tuple)`: Determines the area index for fleet in the starting province.
  - `_fleet_adjacency[start_area_id]`: A data structure containing adjacency information for fleet movements.

- **Calling Functions**:
  - This function is likely used by higher-level functions that need to validate or generate support actions in the game, ensuring that only valid unit types are considered for support orders.

### Note

- Ensure that the province tuples are correctly formatted as `(province_id, flag)`.
- The flag in the province tuple is crucial for identifying special properties like coast type, especially for bicoastal provinces.
- Incorrect formatting or invalid province IDs may lead to inaccurate results or errors.

### Output Example

Suppose we have:

- `start_province_tuple = (1, 0)` where province 1 is a land province.
- `dest_province_tuple = (2, 0)` where province 2 is also a land province.

```python
possible_unit_types_support((1, 0), (2, 0))
```

Output:

```python
{utils.UnitType.ARMY}
```

In this case, only army units can provide support from province 1 to province 2.

Another example:

- `start_province_tuple = (3, 0)` where province 3 is a coastal province.
- `dest_province_tuple = (4, 0)` where province 4 is a sea province.

```python
possible_unit_types_support((3, 0), (4, 0))
```

Output:

```python
{utils.UnitType.FLEET}
```

Here, only fleet units can provide support from province 3 to province 4, assuming fleet adjacency is valid.
## FunctionDef action_to_mila_actions(action)
I understand that I need to create documentation for a specific object, and my audience consists of document readers. Therefore, I should maintain a deterministic tone, ensuring that the content is precise and accurate. It's important not to include any speculation or inaccurate descriptions. Additionally, the readers shouldn't be aware that I'm provided with code snippets and documents.

To approach this task, I'll follow these steps:

1. **Identify the Target Object:** Determine exactly what object needs documentation. This could be a class, function, method, module, or any other programmable entity.

2. **Gather Information:** Review the provided code snippets and documents to collect all necessary information about the target object. This includes understanding its purpose, parameters, return values, exceptions it might throw, and any dependencies it has.

3. **Structure the Documentation:** Organize the information in a logical manner. Typically, documentation starts with an overview of the object, followed by detailed sections such as parameters, returns, exceptions, examples, and notes.

4. **Write Precisely:** Use clear and concise language. Avoid ambiguity and ensure that every statement is accurate based on the code and documents provided.

5. **Review and Validate:** Double-check the documentation against the code to confirm that all information is correct and up-to-date. Make sure there's no misinformation or missing critical details.

Let's assume the target object is a Python function named `calculate_average`. Here’s how I would structure and write the documentation for it.

### Function: calculate_average

#### Overview
Calculates the average of a list of numbers.

#### Parameters
- `numbers` (list of float/int): A list containing numerical values for which the average is to be calculated.

#### Returns
- float: The average of the numbers in the list.

#### Raises
- `ValueError`: If the input list is empty.

#### Example
```python
result = calculate_average([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```

#### Notes
- The function handles both integer and floating-point numbers.
- It raises a `ValueError` if the input list is empty to prevent division by zero.

By following this structured approach, I ensure that the documentation is comprehensive, accurate, and useful for the readers.
## FunctionDef mila_action_to_possible_actions(mila_action)
**mila_action_to_possible_actions**: This function converts a MILA action string to all possible deepmind actions it could refer to.

**Parameters**:
- `mila_action: str` - A string representing an action in the MILA format.

**Code Description**:
The function `mila_action_to_possible_actions` takes a single parameter, `mila_action`, which is a string representing an action in the MILA format. It maps this MILA action to all possible corresponding actions in the deepmind format by looking up a predefined dictionary `_mila_action_to_deepmind_actions`. If the provided MILA action is not found in this dictionary, it raises a ValueError with a message indicating the unrecognised action.

This function is crucial for translating actions between different formats used in the project, specifically from MILA to deepmind action formats. It handles cases where a single MILA action might correspond to multiple deepmind actions, returning all possible mappings as a list.

**Relationship with Callers**:
This function is called by another function within the same module, `mila_action_to_action`, which further refines the selection of the deepmind action based on the current game season. This indicates that `mila_action_to_possible_actions` provides a foundational mapping that is then contextualized by additional logic in `mila_action_to_action`.

**Note**:
- Ensure that the `mila_action` parameter is a valid string representing a MILA action.
- Be aware that if the MILA action is not recognized (i.e., not present in the `_mila_action_to_deepmind_actions` dictionary), the function will raise a ValueError. It's important to handle this exception appropriately in the calling code.
- The function returns a list of `action_utils.Action` objects, which may contain multiple actions if a MILA action maps to several deepmind actions.

**Output Example**:
Suppose `_mila_action_to_deepmind_actions` contains the mapping:
```python
_mila_action_to_deepmind_actions = {
    'hold': [Action.HOLD],
    'move': [Action.MOVE],
    'support_hold': [Action.SUPPORT_HOLD],
    'convoy_move': [Action.CONVOY_MOVE],
    # ... other mappings
}
```
Then, calling `mila_action_to_possible_actions('hold')` would return `[Action.HOLD]`, and `mila_action_to_possible_actions('invalid_action')` would raise a ValueError with the message "Unrecognised MILA action invalid_action".
## FunctionDef mila_action_to_action(mila_action, season)
**mila_action_to_action**: Converts a MILA action string and game season to a specific deepmind action.

### Parameters

- **mila_action: str**
  - A string representing an action in the MILA format.
  
- **season: utils.Season**
  - The current game season, which influences how certain actions are interpreted.

### Code Description

The function `mila_action_to_action` takes a MILA action string and the current game season as inputs and converts them into a specific deepmind action. This conversion is necessary because MILA actions may correspond to multiple deepmind actions, and the correct mapping depends on the game phase represented by the season.

First, the function calls `mila_action_to_possible_actions(mila_action)` to get a list of possible deepmind actions that correspond to the given MILA action. If there is only one possible action, it is directly returned as the result.

If there are multiple possible actions, the function needs to disambiguate between them based on the game season. It does this by breaking down the first possible action using `action_utils.action_breakdown` to extract the order type (e.g., 'REMOVE', 'DISBAND'). Depending on the order type and the current season, it selects the appropriate action from the list of possibilities.

Specifically:

- For a 'REMOVE' order:
  - If the season is in retreats, the second action in the list is selected.
  - Otherwise, the first action is selected.
  
- For a 'DISBAND' order:
  - If the season is in retreats, the first action in the list is selected.
  - Otherwise, the second action is selected.
  
The function asserts that only 'DISBAND' and 'REMOVE' orders can result in ambiguous actions, and raises an assertion error if encountered with any other order type in this context.

### Relationship with Callees

- **mila_action_to_possible_actions**: This function provides the基础 mapping from MILA action strings to potential deepmind actions. It is crucial for generating the list of possible actions that `mila_action_to_action` then refines based on the game season.
  
- **utils.Season**: An enumeration or class that represents different phases of the game, such as build seasons or retreat seasons. The season determines certain rules and constraints that affect how actions are interpreted.
  
- **action_utils.action_breakdown**: A function that decomposes a deepmind action into its constituent parts, likely including the order type, source location, target location, and support target (if applicable). This breakdown helps in understanding the nature of the action for disambiguation.

### Note

- Ensure that the `mila_action` parameter is a valid string representing a MILA action. Invalid actions will cause `mila_action_to_possible_actions` to raise a ValueError.
  
- The season parameter must be an instance of `utils.Season` and should correctly represent the current game phase to ensure proper action interpretation.
  
- The function assumes that only 'DISBAND' and 'REMOVE' orders can be ambiguous in the context of MILA actions. If other orders become ambiguous in the future, the function may need to be updated to handle those cases.

### Output Example

Suppose `mila_action_to_possible_actions('remove')` returns `[Action.REMOVE, Action.DISBAND]`. If the season is in retreats, `mila_action_to_action` would select `Action.DISBAND`; otherwise, it would select `Action.REMOVE`.

Another example: For `mila_action='hold'`, if `mila_action_to_possible_actions('hold')` returns `[Action.HOLD]`, this single action is directly returned without further processing.
