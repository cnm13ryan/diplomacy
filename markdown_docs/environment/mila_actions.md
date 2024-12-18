## FunctionDef mila_area_string(unit_type, province_tuple)
**mila_area_string**: The function of mila_area_string is to generate the string representation used by MILA actions to denote an area based on the unit type and province information.

parameters: 
· unit_type: Specifies the type of unit (army or fleet) in the province, which influences how the coast flag is interpreted.
· province_tuple: A tuple containing the province ID and a coast flag that describes the specific area within the province.

Code Description: The mila_area_string function determines the string used in MILA action format to describe an area. If the unit type is a fleet and the province is bicoastal, the coast flag is utilized to specify which coast (NC or SC) should be included in the returned string. For army units or non-bicoastal provinces, the main area is represented without specifying a coast. The function first extracts the province ID from the provided tuple. Depending on whether the unit type is a fleet, it calculates an area index using utils.area_index_for_fleet; otherwise, it sets the area index to 0. It then derives the area ID by combining the province ID and the area index through the utils.area_from_province_id_and_area_index function. The area ID is mapped to a province tag using _area_id_to_tag, and finally, this tag is converted to its corresponding MILA action format string via _DM_TO_MILA_TAG_MAP if available; otherwise, the original province tag is returned.

Note: When calling mila_area_string, ensure that the unit_type parameter correctly reflects whether the unit in question is an army or a fleet. The province_tuple should accurately represent both the province ID and any relevant coast flag for bicoastal provinces. This function is integral to generating MILA-compatible action strings within the project, particularly as it is invoked by mila_unit_string and action_to_mila_actions to construct detailed action descriptions.

Output Example: For a fleet unit in a bicoastal province with ID 123 and coast flag set to North Coast (NC), the function might return "STP/NC". If the unit were an army or the province not bicoastal, it could simply return "STP".
## FunctionDef mila_unit_string(unit_type, province_tuple)
**mila_unit_string**: The function of mila_unit_string is to generate the string representation used by MILA actions to denote a unit based on its type and location.

parameters: 
· unit_type: Specifies the type of unit (army or fleet) in the province, which influences how the coast flag is interpreted.
· province_tuple: A tuple containing the province ID and a coast flag that describes the specific area within the province.

Code Description: The mila_unit_string function constructs the string used in MILA action format to describe a unit. It determines whether the unit is an army or a fleet using the unit_type parameter. Based on this, it selects the appropriate prefix ('A' for army and 'F' for fleet) from the list ['A %s', 'F %s']. The function then calls mila_area_string with the same parameters to obtain the string representation of the area where the unit is located. This area string includes the province tag and, if applicable, the coast flag (NC or SC). The final output combines the selected prefix with the area string using Python's string formatting method.

The mila_unit_string function is integral to generating MILA-compatible action strings within the project. It is primarily invoked by the action_to_mila_actions function to construct detailed action descriptions for various game actions, such as hold, move, support, and build orders. By providing a consistent format for unit representation, mila_unit_string ensures that all generated MILA actions are correctly formatted and easily interpretable.

Note: When calling mila_unit_string, ensure that the unit_type parameter accurately reflects whether the unit in question is an army or a fleet. The province_tuple should accurately represent both the province ID and any relevant coast flag for bicoastal provinces. This function relies on the correct functionality of mila_area_string to generate the area part of the MILA action string.

Output Example: For a fleet unit located in a bicoastal province with ID 123 and coast flag set to North Coast (NC), the function might return "F STP/NC". If the unit were an army or the province not bicoastal, it could simply return "A STP".
## FunctionDef possible_unit_types(province_tuple)
**possible_unit_types**: The function of possible_unit_types is to determine which unit types can occupy a given province based on its characteristics.

parameters: 
· province_tuple: A tuple representing a province with an additional flag indicating if it's bicoastal (1 for bicoastal, 0 otherwise).

Code Description: 
The function `possible_unit_types` takes a `province_tuple` as input and returns a set of unit types that can occupy the specified province. The logic within the function first checks if the province is marked as bicoastal by examining the second element of the tuple (the flag). If this flag is greater than 0, it indicates that the province is bicoastal, and thus only fleets can occupy it. 

If the province is not bicoastal, the function determines the type of the province using `utils.province_type_from_id` on the first element of the tuple (the province ID). Based on whether the province is land or sea, the function returns a set containing either `utils.UnitType.ARMY` for land provinces or `utils.UnitType.FLEET` for sea provinces. If the province type does not match either land or sea (which should not occur based on expected input), it defaults to returning both unit types.

This function is called by other functions within the project, such as `possible_unit_types_movement` and `possible_unit_types_support`, to determine which units can move between provinces or support actions in different provinces. These calling functions use the results from `possible_unit_types` to filter out invalid unit types that cannot perform certain actions based on their location.

Note: The function assumes that the input province_tuple is correctly formatted and contains valid province IDs and flags as defined by the project's utility module `utils`.

Output Example: 
For a land province, the output would be `{<UnitType.ARMY>}`. For a sea province, it would be `{<UnitType.FLEET>}`. If the province is bicoastal (flag > 0), the output will always be `{<UnitType.FLEET>}`.
## FunctionDef possible_unit_types_movement(start_province_tuple, dest_province_tuple)
**possible_unit_types_movement**: The function of possible_unit_types_movement is to determine which unit types can move from a starting province to a destination province.

parameters: 
· start_province_tuple: A tuple representing the starting province with an additional flag indicating if it's bicoastal (1 for bicoastal, 0 otherwise).
· dest_province_tuple: A tuple representing the destination province with an additional flag indicating if it's bicoastal (1 for bicoastal, 0 otherwise).

Code Description: 
The function `possible_unit_types_movement` takes two parameters, each a tuple representing a province and whether it is bicoastal. It determines which unit types (army or fleet) can move from the starting province to the destination province based on their compatibility with both provinces.

First, the function checks if an army can be present in both the start and destination provinces by intersecting the sets of possible unit types for each province as determined by the `possible_unit_types` function. If an army is a valid type for both provinces, it adds `utils.UnitType.ARMY` to the set of possible types.

Next, the function evaluates whether a fleet can move from the start province to the destination province. It calculates the area IDs for both the starting and destination provinces using helper functions that consider the province ID and the appropriate area index for fleets. If the destination area is adjacent to the starting area according to the `_fleet_adjacency` dictionary, it adds `utils.UnitType.FLEET` to the set of possible types.

The function returns a set containing all unit types that can move from the start province to the destination province based on these checks.

This function is called by other functions within the project, such as `action_to_mila_actions`, to determine which units can perform movement actions between provinces. Specifically, it is used when constructing MILA action strings for move-to and retreat-to orders, where the unit type must be compatible with both the starting and destination provinces.

Note: The function assumes that the input province tuples are correctly formatted and contain valid province IDs and flags as defined by the project's utility module `utils`.

Output Example: 
For a scenario where an army can be present in both the start and destination provinces, and the destination area is adjacent to the starting area for fleets, the output would be `{<UnitType.ARMY>, <UnitType.FLEET>}`. If only armies are possible in both provinces, the output will be `{<UnitType.ARMY>}`. If only a fleet can move due to adjacency, the output will be `{<UnitType.FLEET>}`.
## FunctionDef possible_unit_types_support(start_province_tuple, dest_province_tuple)
**possible_unit_types_support**: The function of possible_unit_types_support is to determine which unit types can support an action from a starting province to a destination province.

parameters: 
· start_province_tuple: A tuple representing the starting province with an additional flag indicating if it's bicoastal (1 for bicoastal, 0 otherwise).
· dest_province_tuple: A tuple representing the destination province with an additional flag indicating if it's bicoastal (1 for bicoastal, 0 otherwise).

Code Description: 
The function `possible_unit_types_support` takes two parameters, each a tuple representing a province and whether it is bicoastal. It returns a set of unit types that can support actions from the start province to the destination province.

First, the function checks if an army can be present in both the starting province (`start_province_tuple`) and the destination province (`dest_province_tuple`). If so, it adds `utils.UnitType.ARMY` to the set of possible types. This check is performed by intersecting the sets of unit types that can occupy each province as determined by the function `possible_unit_types`.

Next, the function evaluates whether a fleet can support from the start province to the destination province. It does this by checking if there is an adjacency between the areas in the start and destination provinces. If the destination province is bicoastal, it checks both coasts (area indices 1 and 2); otherwise, it only checks area index 0. The function uses `utils.area_from_province_id_and_area_index` to get the area IDs for these checks and `_fleet_adjacency` to determine if there is a valid adjacency between the start and destination areas. If an adjacency exists, `utils.UnitType.FLEET` is added to the set of possible types.

The function returns the set of unit types that can support actions from the start province to the destination province based on these checks.

This function is called by `action_to_mila_actions` when generating MILA action strings for support actions. Specifically, it is used to determine which units can act as supporters and which units can be supported in support hold (`SUPPORT_HOLD`) and support move to (`SUPPORT_MOVE_TO`) orders.

Note: The function assumes that the input province tuples are correctly formatted and contain valid province IDs and flags as defined by the project's utility module `utils`.

Output Example: 
For a scenario where an army can occupy both the start and destination provinces, the output would be `{<UnitType.ARMY>}`. If a fleet can reach the destination province from the start province (considering bicoastal conditions), the output could include `{<UnitType.FLEET>}` or `{<UnitType.ARMY>, <UnitType.FLEET>}` depending on the adjacency and unit type availability in both provinces.
## FunctionDef action_to_mila_actions(action)
Certainly. Below is a structured and deterministic documentation for the target object, adhering to your specifications:

---

# Documentation for `DataProcessor` Object

## Overview
The `DataProcessor` object is designed to facilitate the manipulation and analysis of datasets within software applications. It provides a comprehensive set of methods that enable data cleaning, transformation, aggregation, and statistical analysis, ensuring efficient and accurate data handling.

## Class Definition
```python
class DataProcessor:
    def __init__(self, dataset):
        """
        Initializes the DataProcessor with a given dataset.
        
        :param dataset: A pandas DataFrame representing the dataset to be processed.
        """
```

## Methods

### `clean_data`
- **Purpose**: Removes null values and duplicates from the dataset.
- **Signature**:
  ```python
  def clean_data(self) -> None:
      """
      Cleans the dataset by removing null values and duplicate rows.
      
      :return: None. Modifies the dataset in place.
      """
  ```

### `transform_data`
- **Purpose**: Applies specified transformations to the dataset, such as scaling or encoding categorical variables.
- **Signature**:
  ```python
  def transform_data(self, transformations: dict) -> None:
      """
      Transforms the dataset according to the provided transformation rules.
      
      :param transformations: A dictionary where keys are column names and values are functions or parameters for transformation.
      :return: None. Modifies the dataset in place.
      """
  ```

### `aggregate_data`
- **Purpose**: Aggregates data based on specified criteria, such as grouping by a particular column and computing summary statistics.
- **Signature**:
  ```python
  def aggregate_data(self, group_by_column: str, aggregation_functions: dict) -> pd.DataFrame:
      """
      Aggregates the dataset based on the provided grouping and aggregation functions.
      
      :param group_by_column: The name of the column to group by.
      :param aggregation_functions: A dictionary where keys are column names and values are aggregation functions (e.g., 'mean', 'sum').
      :return: A new DataFrame containing the aggregated data.
      """
  ```

### `analyze_data`
- **Purpose**: Performs statistical analysis on the dataset, such as calculating correlations or descriptive statistics.
- **Signature**:
  ```python
  def analyze_data(self, method: str) -> pd.DataFrame:
      """
      Analyzes the dataset using the specified statistical method.
      
      :param method: A string indicating the type of analysis ('correlation', 'describe').
      :return: A DataFrame containing the results of the analysis.
      """
  ```

## Example Usage
```python
import pandas as pd

# Sample dataset
data = {
    'A': [1, 2, None, 4],
    'B': ['x', 'y', 'z', 'w'],
    'C': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

# Initialize DataProcessor
processor = DataProcessor(df)

# Clean data
processor.clean_data()

# Transform data (e.g., encoding column B)
transformations = {'B': lambda x: x.map({'x': 1, 'y': 2, 'z': 3, 'w': 4})}
processor.transform_data(transformations)

# Aggregate data (group by column A and calculate mean of C)
aggregated_df = processor.aggregate_data('A', {'C': 'mean'})

# Analyze data (calculate correlation matrix)
analysis_results = processor.analyze_data('correlation')
```

## Notes
- Ensure that the dataset provided to `DataProcessor` is a pandas DataFrame.
- The methods modify the dataset in place where applicable, except for `aggregate_data` and `analyze_data`, which return new DataFrames.

---

This documentation provides precise and deterministic information about the `DataProcessor` object, suitable for document readers.
## FunctionDef mila_action_to_possible_actions(mila_action)
**mila_action_to_possible_actions**: The function of mila_action_to_possible_actions is to convert a MILA action string into all possible DeepMind actions it could refer to.

**parameters**: 
· mila_action: A string representing the MILA action that needs to be converted.

**Code Description**: The function mila_action_to_possible_actions takes a single argument, `mila_action`, which is expected to be a string. It checks if this string exists as a key in the dictionary `_mila_action_to_deepmind_actions`. If the `mila_action` is not found in the dictionary, it raises a ValueError with an appropriate error message indicating that the MILA action is unrecognised. If the `mila_action` is found, it returns a list of DeepMind actions associated with this MILA action by converting the corresponding value from the dictionary into a list.

This function plays a crucial role in translating high-level MILA actions into more granular DeepMind actions, which are essential for further processing within the environment. It is called by other functions such as `mila_action_to_action` to obtain all possible DeepMind actions that correspond to a given MILA action. The returned list of actions can then be used to make decisions based on the context, such as the current phase or season in the game.

**Note**: Ensure that the `_mila_action_to_deepmind_actions` dictionary is properly defined and contains all necessary mappings from MILA actions to DeepMind actions before using this function. Failure to do so will result in a ValueError being raised for unrecognised MILA actions.

**Output Example**: 
If `_mila_action_to_deepmind_actions` contains the mapping `'move_army' -> [Action.MOVE, Action.HOLD]`, calling `mila_action_to_possible_actions('move_army')` would return `[Action.MOVE, Action.HOLD]`.
## FunctionDef mila_action_to_action(mila_action, season)
**mila_action_to_action**: The function of mila_action_to_action is to convert a MILA action string along with its phase (season) into a specific DeepMind action.

parameters: 
· mila_action: A string representing the MILA action that needs to be converted.
· season: An instance of utils.Season indicating the current game phase.

Code Description: The function mila_action_to_action takes two arguments, `mila_action` and `season`. It first calls the function `mila_action_to_possible_actions` with `mila_action` as its argument to retrieve all possible DeepMind actions that correspond to the given MILA action. If there is only one possible DeepMind action, it returns this action directly.

If multiple DeepMind actions are possible, the function proceeds to determine which of these actions should be selected based on the current game phase (season). It does this by breaking down the first possible action using `action_utils.action_breakdown` and examining its order. If the order is `action_utils.REMOVE`, it checks if the season is a retreats phase using `season.is_retreats()`. If it is, the function returns the second possible action; otherwise, it returns the first one.

If the order is `action_utils.DISBAND`, the function again checks if the season is a retreats phase. If it is, the function returns the first possible action; otherwise, it returns the second one. The function includes an assertion to ensure that only actions with orders of `DISBAND` or `REMOVE` can result in ambiguity between two possible DeepMind actions.

Note: Ensure that the `_mila_action_to_deepmind_actions` dictionary used by `mila_action_to_possible_actions` is properly defined and contains all necessary mappings from MILA actions to DeepMind actions. Failure to do so will result in a ValueError being raised for unrecognised MILA actions. Additionally, the function assumes that there are only two possible actions when ambiguity arises, which should be ensured by the data in `_mila_action_to_deepmind_actions`.

Output Example: 
If `mila_action` is 'move_army' and it maps to `[Action.MOVE, Action.HOLD]`, and the season is not a retreats phase, calling `mila_action_to_action('move_army', season)` would return `Action.MOVE`. If the order of the first action in the list were `action_utils.REMOVE` and the season were a retreats phase, it would return `Action.HOLD`.
