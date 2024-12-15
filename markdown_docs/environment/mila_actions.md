## FunctionDef mila_area_string(unit_type, province_tuple)
**mila_area_string**: The function of mila_area_string is to generate a string that represents an area in the MILA action format based on the unit type and province information.
**parameters**: The parameters of this Function.
· unit_type: This parameter determines the type of unit in the province, which can be either army or fleet, and is used to decide whether to include coast information in the output string.
· province_tuple: This parameter contains a tuple with the province ID and an optional coast flag, providing information about the specific area being represented.

**Code Description**: The mila_area_string function takes into account the unit type and province information to generate the MILA action string. If the unit type is fleet and the province is bicoastal, it uses the coast flag from the province tuple to determine which coast to include in the output string. For other cases, it simply uses the main area ID. The function utilizes helper functions such as utils.area_index_for_fleet and utils.area_from_province_id_and_area_index to derive the correct area index and ID based on the input parameters. It then maps this area ID to a province tag using the _area_id_to_tag dictionary and finally returns the corresponding MILA tag from the _DM_TO_MILA_TAG_MAP dictionary, or the original province tag if no mapping is found.

The mila_area_string function is called by other functions in the project, such as mila_unit_string and action_to_mila_actions. In the context of mila_unit_string, it provides the area information needed to construct a complete unit string in the MILA format. The action_to_mila_actions function relies on mila_area_string to generate the correct area representations for various actions, including moves, supports, and convoys.

**Note**: When using this function, it is essential to ensure that the input parameters are valid and correctly formatted, as incorrect data may lead to unexpected output or errors. Additionally, the function's behavior depends on the specific definitions of unit types, province IDs, and coast flags in the project's context.

**Output Example**: A possible return value of the mila_area_string function could be "STP/NC", indicating a fleet unit in the St. Petersburg province with a northern coast flag. Alternatively, for an army unit in a non-bicoastal province, the output might simply be "PAR", representing the Paris province without any coast information.
## FunctionDef mila_unit_string(unit_type, province_tuple)
**mila_unit_string**: The function of mila_unit_string is to generate a string that represents a unit in the MILA action format based on the unit type and province information.
**parameters**: The parameters of this Function.
· unit_type: This parameter determines the type of unit, which can be either army or fleet, and is used to decide the prefix of the output string.
· province_tuple: This parameter contains a tuple with the province ID and an optional coast flag, providing information about the specific area being represented.
**Code Description**: The mila_unit_string function takes into account the unit type and province information to generate the MILA action string. It uses a list of prefixes to determine whether the unit is an army or a fleet, and then calls the mila_area_string function to get the area representation. The function returns a string that combines the prefix and the area representation. In the context of the project, this function is called by the action_to_mila_actions function to construct complete unit strings in the MILA format for various actions.
**Note**: When using this function, it is essential to ensure that the input parameters are valid and correctly formatted, as incorrect data may lead to unexpected output or errors. The function's behavior depends on the specific definitions of unit types and province information in the project's context.
**Output Example**: A possible return value of the mila_unit_string function could be "A PAR" for an army unit in the Paris province, or "F STP/NC" for a fleet unit in the St. Petersburg province with a northern coast flag.
## FunctionDef possible_unit_types(province_tuple)
**possible_unit_types**: The function of possible_unit_types is to determine the unit types that can occupy a given province.
**parameters**: The parameters of this Function.
· province_tuple: A tuple containing the province ID and a flag indicating whether the province is bicoastal, used to determine the unit types that can occupy it.
**Code Description**: This function takes into account the type of province and its flag to decide which unit types are allowed. If the province's flag is greater than 0, it must be a fleet in a bicoastal province, so only the FLEET unit type is returned. Otherwise, the function checks the province type: if it's LAND, only ARMY units can occupy it; if it's SEA, only FLEET units are allowed; and for other types, both ARMY and FLEET units can occupy the province. This function is used by other functions in the project, such as possible_unit_types_movement and possible_unit_types_support, to determine valid unit movements and support actions. Additionally, it is also utilized in the action_to_mila_actions function to generate Mila action strings for various orders.
**Note**: The function's output depends on the input province tuple, so it's essential to provide a valid tuple with the correct province ID and flag. Also, this function assumes that the utils module is available and provides the necessary functions, such as province_type_from_id, to determine the province type.
**Output Example**: If the input province_tuple is (1, 0) and the province with ID 1 is of type LAND, the output would be {utils.UnitType.ARMY}. If the input province_tuple is (2, 1) and the province with ID 2 is bicoastal, the output would be {utils.UnitType.FLEET}.
## FunctionDef possible_unit_types_movement(start_province_tuple, dest_province_tuple)
**possible_unit_types_movement**: The function of possible_unit_types_movement is to determine the unit types that can move from a start province to a destination province.
**parameters**: The parameters of this Function.
· start_province_tuple: A tuple containing the start province ID and a flag indicating whether the province is bicoastal, used to determine the unit types that can occupy it.
· dest_province_tuple: A tuple containing the destination province ID and a flag indicating whether the province is bicoastal, used to determine the unit types that can move to it.
**Code Description**: This function takes into account the type of provinces and their flags to decide which unit types are allowed to make the move. It first checks if an army can move from the start to the destination by verifying if the army unit type is present in both the start and destination provinces using the possible_unit_types function. If an army can move, it adds the army unit type to the set of possible unit types. Then, it checks if a fleet can make the journey by determining the area IDs of the start and destination provinces and verifying if there is a valid adjacency between them. If a fleet can move, it adds the fleet unit type to the set of possible unit types. The function returns a set of unit types that could make this move.
**Note**: The function's output depends on the input province tuples, so it's essential to provide valid tuples with the correct province IDs and flags. Also, this function assumes that the necessary functions, such as possible_unit_types and area_from_province_id_and_area_index, are available and provide the required information. This function is used by other functions in the project, such as action_to_mila_actions, to determine valid unit movements.
**Output Example**: If the input start_province_tuple is (1, 0) and the destination province tuple is (2, 0), and both provinces allow army units, the output would be {utils.UnitType.ARMY}. If the input start_province_tuple is (3, 0) and the destination province tuple is (4, 0), and there is a valid adjacency between their area IDs for fleet movement, the output would be {utils.UnitType.FLEET}.
## FunctionDef possible_unit_types_support(start_province_tuple, dest_province_tuple)
**possible_unit_types_support**: The function of possible_unit_types_support is to determine the unit types that can support a destination province from a given start province.
**parameters**: The parameters of this Function.
· start_province_tuple: A tuple containing the province ID and a flag indicating whether the province is bicoastal, representing the starting location of the unit.
· dest_province_tuple: A tuple containing the province ID and a flag indicating whether the province is bicoastal, representing the destination province that the support is being offered to.
**Code Description**: This function takes into account the types of provinces and their flags to decide which unit types are allowed to offer support. It first checks if an army can support the destination from the start area by verifying if the army unit type is present in both the possible unit types at the start province and the possible unit types at the destination province with a flag set to 0. If this condition is met, it adds the army unit type to the set of possible unit types that can offer support. Then, it checks if a fleet can actually reach the destination province by verifying the adjacency of the start area and the destination area. If the destination area is adjacent to the start area, it adds the fleet unit type to the set of possible unit types that can offer support. The function returns a set of unit types that could make this support.
The function possible_unit_types_support relies on other functions in the project, such as possible_unit_types and utils.area_from_province_id_and_area_index, to determine the valid unit types and areas involved in the support action. On the other hand, it is also utilized by other functions, like action_to_mila_actions, to generate Mila action strings for support-related orders.
**Note**: The function's output depends on the input province tuples, so it's essential to provide valid tuples with the correct province IDs and flags. Additionally, this function assumes that the utils module is available and provides the necessary functions to determine the province types and areas.
**Output Example**: If the input start_province_tuple is (1, 0) and the input dest_province_tuple is (2, 0), and the province with ID 1 is of type LAND and the province with ID 2 is also of type LAND, the output would be {utils.UnitType.ARMY}. If the input start_province_tuple is (3, 1) and the input dest_province_tuple is (4, 0), and the province with ID 3 is bicoastal and the province with ID 4 is of type SEA, the output would be {utils.UnitType.FLEET}.
## FunctionDef action_to_mila_actions(action)
**Target Object Documentation**

### Overview

The Target Object is a fundamental entity designed to represent a specific goal or objective within a system or application. It serves as a focal point for various operations, allowing users to define and pursue distinct targets with precision.

### Properties

The following properties are associated with the Target Object:

1. **Identifier (ID)**: A unique identifier assigned to each Target Object, enabling efficient retrieval and manipulation.
2. **Name**: A descriptive label assigned to the Target Object, providing a human-readable representation of its purpose or goal.
3. **Description**: An optional text field allowing users to provide additional context or information about the Target Object.

### Methods

The Target Object supports the following methods:

1. **Create**: Initializes a new Target Object with the specified properties (ID, Name, and Description).
2. **Retrieve**: Fetches an existing Target Object based on its unique Identifier (ID).
3. **Update**: Modifies the properties of an existing Target Object.
4. **Delete**: Removes a Target Object from the system.

### Relationships

The Target Object can be related to other entities within the system, including:

1. **Users**: Multiple users can be associated with a single Target Object, enabling collaborative pursuit of the objective.
2. **Tasks**: A Target Object can be linked to multiple tasks, which are specific actions or activities aimed at achieving the target.

### Constraints

The following constraints apply to the Target Object:

1. **Uniqueness**: Each Target Object must have a unique Identifier (ID).
2. **Data Validation**: Property values (e.g., Name and Description) are subject to validation rules to ensure data consistency and accuracy.

### Usage Examples

1. Creating a new Target Object: `CreateTargetObject("Increase Sales", "Boost revenue by 10% within the next quarter")`
2. Retrieving an existing Target Object: `GetTargetObject(123)` (where 123 is the unique ID)
3. Updating a Target Object's properties: `UpdateTargetObject(123, "New Name", "Updated Description")`

By utilizing the Target Object effectively, users can establish clear objectives and work towards achieving them in a structured and organized manner.
## FunctionDef mila_action_to_possible_actions(mila_action)
**mila_action_to_possible_actions**: The function of mila_action_to_possible_actions is to convert a MILA action string into a list of possible deepmind actions it could refer to.
**parameters**: The parameters of this Function.
· mila_action: a string representing the MILA action to be converted
**Code Description**: This function takes a MILA action string as input and checks if it exists in the _mila_action_to_deepmind_actions dictionary. If the action is not recognized, it raises a ValueError with an error message indicating that the MILA action is unrecognised. If the action is valid, it returns a list of deepmind actions associated with the given MILA action. The function is used by mila_action_to_action to determine the possible deepmind actions for a given MILA action and then select the appropriate one based on the season.
**Note**: It is essential to ensure that the input MILA action string is valid and exists in the _mila_action_to_deepmind_actions dictionary to avoid raising a ValueError. The function assumes that the _mila_action_to_deepmind_actions dictionary has been properly populated with the mapping of MILA actions to deepmind actions.
**Output Example**: The output of this function could be a list of action_utils.Action objects, for example [action_utils.Action('move'), action_utils.Action('hold')] if the input MILA action string corresponds to multiple possible deepmind actions.
## FunctionDef mila_action_to_action(mila_action, season)
**mila_action_to_action**: The function of mila_action_to_action is to convert a MILA action string and its phase to the corresponding deepmind action based on the given season.
**parameters**: The parameters of this Function.
· mila_action: a string representing the MILA action to be converted
· season: an object of type utils.Season, which provides information about the current season
**Code Description**: This function takes a MILA action string and a season object as input, and uses the mila_action_to_possible_actions function to determine the possible deepmind actions that the given MILA action could refer to. If there is only one possible deepmind action, it returns that action. However, if there are multiple possible actions, it further analyzes the order of the first possible action using the action_breakdown function from the action_utils module. Based on the order and the current season, it selects the appropriate deepmind action to return. Specifically, if the order is REMOVE or DISBAND, it checks if the season is in retreats phase and returns either the first or second possible action accordingly.
**Note**: It is essential to ensure that the input MILA action string is valid and exists in the mapping of MILA actions to deepmind actions, as used by the mila_action_to_possible_actions function. Additionally, the function assumes that the season object provides accurate information about the current season.
**Output Example**: The output of this function could be an action_utils.Action object, for example action_utils.Action('move') or action_utils.Action('hold'), depending on the input MILA action string and the current season.
