## FunctionDef area_string(area_tuple)
**Function Overview**

The `area_string` function returns a human-readable string representation of a province based on its ID.

**Parameters**

- **area_tuple**: A tuple representing a province with a flag. The first element is the province ID (`int`) and the second element is a boolean flag indicating some property (not used within this function).

**Return Values**

- Returns a `str` which is the human-readable tag corresponding to the provided province ID.

**Detailed Explanation**

The `area_string` function takes an input tuple, `area_tuple`, which contains two elements: a province ID and a flag. The function uses the province ID to look up its corresponding human-readable string representation in the `_province_id_to_tag` dictionary. It then returns this string.

Here is a breakdown of the function's logic:

1. **Input Handling**: The function expects `area_tuple` to be an instance of `utils.ProvinceWithFlag`, which is a tuple containing a province ID and a flag.
2. **Dictionary Lookup**: Using the province ID from `area_tuple[0]`, the function retrieves the corresponding human-readable tag from the `_province_id_to_tag` dictionary.
3. **Return Statement**: The retrieved string is returned as the output of the function.

**Usage Notes**

- **Limitations**: This function assumes that the `_province_id_to_tag` dictionary is pre-populated with correct mappings between province IDs and their tags. If the dictionary does not contain a mapping for the given province ID, this will result in a `KeyError`.
- **Edge Cases**: The function does not handle cases where the input tuple is malformed (e.g., missing elements or incorrect types). It assumes that the input adheres to the expected format.
- **Performance Considerations**: The performance of this function is primarily determined by the efficiency of dictionary lookups in `_province_id_to_tag`. If the dictionary is large, lookup times may increase. However, given typical use cases, this should not be a significant concern.

This function is used within other functions such as `area_string_with_coast_if_fleet` and `action_string` to generate human-readable strings for actions involving provinces.
## FunctionDef area_string_with_coast_if_fleet(area_tuple, unit_type)
**Function Overview**

The `area_string_with_coast_if_fleet` function returns a human-readable string representation of an area, specifically indicating coasts when a fleet is present in a bicoastal province.

**Parameters**

- **area_tuple**: A tuple representing a province with a flag. The first element is the province ID (`int`) and the second element is a boolean flag (not used within this function).
- **unit_type**: An optional parameter of type `utils.UnitType` that specifies the type of unit present in the area. It can be one of `utils.UnitType.ARMY`, `utils.UnitType.FLEET`, or `None`.

**Return Values**

- Returns a `str` which is the human-readable tag corresponding to the provided province ID, with additional coast information if applicable.

**Detailed Explanation**

The `area_string_with_coast_if_fleet` function determines how to represent an area based on whether it contains a fleet and whether it is bicoastal. The function's logic can be broken down into several steps:

1. **Input Handling**: The function expects `area_tuple` to be an instance of `utils.ProvinceWithFlag`, which is a tuple containing a province ID and a flag. It also accepts an optional `unit_type` parameter that specifies the type of unit present in the area.

2. **Single-Coasted Provinces or Army Units**:
   - If the province ID indicates a single-coasted province (`province_id < utils.SINGLE_COASTED_PROVINCES`) or if the unit type is an army (`unit_type == utils.UnitType.ARMY`), the function calls `area_string(area_tuple)` to get the human-readable string representation of the area without additional coast information.

3. **Fleet in a Bicoastal Province**:
   - If the unit type is a fleet (`unit_type == utils.UnitType.FLEET`) and the province is bicoastal, the function retrieves the province tag using `_province_string(province_id)` and appends coast information based on the second element of `area_tuple`. The result is a string indicating the specific coast where the fleet is located.

4. **Unknown Unit Type**:
   - If the unit type is unknown (`unit_type == None`), the function behaves similarly to the case where the unit type is an army, calling `area_string(area_tuple)` and returning the human-readable string representation of the area without additional coast information.

**Usage Notes**

- **Limitations**: The function assumes that the province ID in `area_tuple` is valid and corresponds to a known province. It also relies on the correct mapping between province IDs and their respective tags.
  
- **Edge Cases**: 
  - If the province ID does not correspond to any known province, the behavior of `_province_string(province_id)` is undefined and may result in an error or unexpected output.
  - The function does not handle cases where `area_tuple` is malformed (e.g., missing elements) or contains invalid data types.

- **Performance Considerations**: 
  - The function's performance is primarily dependent on the efficiency of `_province_string(province_id)` and the operations performed within the conditional statements. Since these operations are generally simple, the function should perform well for typical use cases.
  
- **Integration with Other Functions**: This function is used by `action_string` to generate human-readable descriptions of actions involving units in specific areas. It is crucial that the output format remains consistent to ensure proper integration and readability of game logs or user interfaces.

By following these guidelines, developers can effectively utilize `area_string_with_coast_if_fleet` to accurately represent areas with fleet-specific coast information in their applications.
## FunctionDef action_string(action, board)
```json
{
  "module": "core",
  "class": "DataProcessor",
  "description": "The DataProcessor class is designed to handle and process large datasets efficiently. It provides methods for data cleaning, transformation, and analysis.",
  "methods": [
    {
      "name": "__init__",
      "parameters": [
        {"name": "data_source", "type": "str", "description": "Path or URL to the dataset."}
      ],
      "description": "Initializes a new instance of DataProcessor with the specified data source."
    },
    {
      "name": "load_data",
      "parameters": [],
      "description": "Loads the dataset from the provided source into memory for processing."
    },
    {
      "name": "clean_data",
      "parameters": [
        {"name": "remove_nulls", "type": "bool", "default": "True", "description": "Flag to indicate whether null values should be removed."}
      ],
      "description": "Cleans the loaded dataset by removing or handling specified anomalies."
    },
    {
      "name": "transform_data",
      "parameters": [
        {"name": "transformation_type", "type": "str", "options": ["normalize", "standardize"], "description": "Type of transformation to apply to the data."}
      ],
      "description": "Applies a specified type of transformation to the dataset for better analysis."
    },
    {
      "name": "analyze_data",
      "parameters": [],
      "returns": {"type": "dict", "description": "A dictionary containing statistical summaries and insights from the dataset."},
      "description": "Performs an analysis on the processed data, returning key findings."
    }
  ]
}
```
