## FunctionDef area_string(area_tuple)
**area_string**: The function of area_string is to convert an area tuple into its corresponding human-readable string representation based on province tags.

parameters: 
· area_tuple: This parameter is expected to be of type `utils.ProvinceWithFlag`, which is essentially a tuple containing the province ID and a flag (coast number).

Code Description: The function takes an input tuple, `area_tuple`, which includes a province ID and a coast number. It then retrieves the corresponding string tag for the province from the `_province_id_to_tag` dictionary using the province ID as the key. This function is primarily used to convert internal identifiers (province IDs) into more human-readable format (province tags). The function does not consider the coast number in its output, which makes it suitable for scenarios where the specific coast information is not necessary.

The `area_string` function is called by other functions within the same module to generate parts of human-readable action strings. Specifically, it is used in `action_string` when the board information is not available or when the unit type does not require coast specification (e.g., armies). Additionally, it is utilized in `area_string_with_coast_if_fleet` when the unit type is an army or when the province is single-coasted.

Note: The function assumes that `_province_id_to_tag` is a predefined dictionary mapping province IDs to their respective string tags. It does not handle cases where the province ID might be missing from this dictionary, which could lead to a KeyError if such a scenario occurs.

Output Example: If `area_tuple` is `(1034, 0)`, and `_province_id_to_tag[1034]` is `'STP'`, then the function will return `'STP'`.
## FunctionDef area_string_with_coast_if_fleet(area_tuple, unit_type)
**area_string_with_coast_if_fleet**: The function of area_string_with_coast_if_fleet is to generate a human-readable string representation of an area tuple, including coast information when the unit type is a fleet and the province is bicoastal.

parameters: 
· area_tuple: This parameter is expected to be of type `utils.ProvinceWithFlag`, which is essentially a tuple containing the province ID and a flag (coast number).
· unit_type: This parameter is an optional argument that specifies the type of the unit, either `utils.UnitType.ARMY` or `utils.UnitType.FLEET`. If not provided, it defaults to `None`.

Code Description: The function takes two parameters: `area_tuple`, which includes a province ID and a coast number, and `unit_type`, which indicates whether the unit is an army or a fleet. It first checks if the province is single-coasted (province_id less than `utils.SINGLE_COASTED_PROVINCES`) or if the unit type is an army. In either case, it calls the `area_string` function to return the human-readable string representation of the area without coast information.

If the unit type is a fleet and the province is bicoastal (province_id greater than or equal to `utils.SINGLE_COASTED_PROVINCES`), the function retrieves the corresponding province tag from the `_province_id_to_tag` dictionary using the province ID as the key. It then appends 'NC' for North Coast or 'SC' for South Coast based on the coast number (0 for North Coast, non-zero for South Coast) to the province tag and returns this string.

If the unit type is unknown (`None`), indicating that the caller does not have information about the unit type, the function still retrieves the province tag from `_province_id_to_tag`. It appends 'maybe_NC' or 'SC' based on the coast number to indicate possible North Coast or South Coast and returns this string.

In all other cases, where the unit type is neither an army nor a fleet (or invalid), the function raises a `ValueError`.

This function is primarily used in generating human-readable action strings when the board information is available and the unit type requires coast specification. It is called by the `action_string` function for actions that involve movement (`MOVE_TO`, `RETREAT_TO`) or building fleets (`BUILD_FLEET`), where specifying the correct coast is crucial.

Note: The function assumes that `_province_id_to_tag` is a predefined dictionary mapping province IDs to their respective string tags. It does not handle cases where the province ID might be missing from this dictionary, which could lead to a KeyError if such a scenario occurs.

Output Example: If `area_tuple` is `(1034, 0)` and `_province_id_to_tag[1034]` is `'STP'`, and `unit_type` is `utils.UnitType.FLEET`, then the function will return `'STPMC'`. If `unit_type` is `None`, it will return `'STPmaybe_NC'`. If `area_tuple` is `(102, 0)` (a single-coasted province) and `unit_type` is `utils.UnitType.FLEET`, it will return `'STP'`.
## FunctionDef action_string(action, board)
**action_string**: The function of action_string is to convert an action into its human-readable string representation based on the game's rules.

parameters: 
· action: This parameter represents the action to be converted and is expected to be of type `Union[action_utils.Action, action_utils.ActionNoIndex]`. It contains details about the order, starting position, target position, and additional information as needed.
· board: This optional parameter is a numpy array representing the game board. It provides context about the units' types (army or fleet) to ensure accurate coast annotations in certain actions.

Code Description: The function `action_string` takes an action and optionally a board to generate a human-readable string that describes the action in a concise format. It first breaks down the action into its components using `action_utils.action_breakdown(action)` which returns the order type and positions involved (p1, p2, p3). 

The function then determines the unit's location by calling `area_string(p1)`. If a board is provided, it also identifies whether the unit at position p1 is an army or fleet using `utils.unit_type(p1[0], board)`.

Depending on the order type (e.g., HOLD, CONVOY, MOVE_TO), the function constructs the human-readable string by combining the unit's location with the appropriate action notation. For example:
- For a HOLD order, it returns '{unit_string} H'.
- For a CONVOY order, it returns '{unit_string} C {area_string(p3)} - {area_string(p2)}'.
- For a MOVE_TO order, it uses `area_string_with_coast_if_fleet` to ensure the correct coast is specified if the unit is a fleet and the province is bicoastal.

The function handles various action types including support actions (SUPPORT_HOLD, SUPPORT_MOVE_TO), retreats (RETREAT_TO), disbands (DISBAND), builds (BUILD_ARMY, BUILD_FLEET), removals (REMOVE), and waives (WAIVE). If an unrecognized order type is encountered, it raises a `ValueError`.

Note: The function relies on the correctness of the input action format and optionally the board to provide accurate human-readable strings. It assumes that the necessary utility functions (`action_utils.action_breakdown`, `area_string`, `utils.unit_type`, `area_string_with_coast_if_fleet`) are correctly implemented and available.

Output Example: If the action is a move order for a fleet from province (1034, 0) to province (1035, 1), and the board indicates that the unit at (1034, 0) is a fleet, then `action_string` will return 'STPMC - STPSC'. Here, 'STPMC' represents the starting location with coast specification for a fleet, and 'STPSC' represents the target location with coast specification.
