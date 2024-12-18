## ClassDef MapMDF
**MapMDF**: The function of MapMDF is to define constants representing different types of map configurations used within the project.

attributes: The attributes of this Class.
· STANDARD_MAP: Represents the standard map configuration with an integer value of 0.
· BICOASTAL_MAP: Represents a bicoastal map configuration with an integer value of 1.

Code Description: The MapMDF class is an enumeration that defines two constants, STANDARD_MAP and BICOASTAL_MAP. These constants are used to specify which map configuration should be utilized in various functions throughout the project. The use of an enumeration ensures that only valid map configurations can be passed as arguments to functions expecting a map type.

In the provided codebase, MapMDF is primarily used in the following functions:
- get_mdf_content: This function takes a MapMDF argument and returns the corresponding MDF content string based on the specified map configuration. If an unknown MapMDF value is provided, it raises a ValueError.
- province_name_to_id: This function also accepts a MapMDF argument to determine which map's MDF content should be used when generating a dictionary that maps province names to their respective IDs.
- province_id_to_home_sc_power: Although this function does not explicitly accept a MapMDF argument, it internally calls get_mdf_content with the STANDARD_MAP configuration. This indicates that the home superpower mapping is based on the standard map configuration.
- fleet_adjacency_map: Similar to province_name_to_id, this function uses the BICOASTAL_MAP configuration when calling get_mdf_content to build a mapping of valid fleet movements between areas.

Note: When using MapMDF in your code, ensure that you are passing one of the predefined constants (STANDARD_MAP or BICOASTAL_MAP) to avoid runtime errors. The choice of map configuration should align with the specific requirements of the function being called and the data it processes.
## FunctionDef get_mdf_content(map_mdf)
**get_mdf_content**: The function of get_mdf_content is to retrieve the MDF content string based on the specified map configuration.

parameters: 
· map_mdf: Represents the type of map configuration for which the MDF content should be returned. It accepts values from the MapMDF enumeration, specifically MapMDF.STANDARD_MAP or MapMDF.BICOASTAL_MAP. The default value is MapMDF.STANDARD_MAP.

Code Description: The get_mdf_content function takes a single parameter, map_mdf, which specifies the desired map configuration. If map_mdf is set to MapMDF.STANDARD_MAP, the function returns the content of _STANDARD_MAP_MDF_CONTENT. If it is set to MapMDF.BICOASTAL_MAP, it returns the content of _BICOASTAL_MAP_MDF_CONTENT. If an invalid value is provided for map_mdf, a ValueError is raised with a message indicating the unknown map configuration.

In the project, get_mdf_content serves as a utility function that provides MDF content based on the specified map type. It is utilized by several other functions to fetch the appropriate MDF data:
- province_name_to_id: This function calls get_mdf_content to obtain the MDF content for generating a dictionary that maps province names to their respective IDs.
- province_id_to_home_sc_power: Although this function does not directly accept a map configuration, it internally uses get_mdf_content with MapMDF.STANDARD_MAP to determine which power is the home superpower for each province based on the standard map configuration.
- fleet_adjacency_map: This function calls get_mdf_content with MapMDF.BICOASTAL_MAP to build a mapping of valid fleet movements between areas, using the bicoastal map configuration.

Note: When calling get_mdf_content, ensure that you pass one of the predefined constants from the MapMDF enumeration (STANDARD_MAP or BICOASTAL_MAP) to avoid runtime errors. The choice of map configuration should align with the specific requirements of the function being called and the data it processes.

Output Example: 
If map_mdf is set to MapMDF.STANDARD_MAP, the output might be a string like:
```
PROVINCES
(
  ENG England
  SPA Spain
  FRA France
)
HOME_SC_POWERS (ENG SPA FRA)
...
```

If map_mdf is set to MapMDF.BICOASTAL_MAP, the output might be a string like:
```
AREAS
(
  ENG England
  SPA Spain
  FRA France
)
FLEET_ADJACENCY (
  ENG (SPA FLT)
  SPA (ENG FLT)
  FRA (SPA FLT)
)
...
```
## FunctionDef _province_tag(l)
**_province_tag**: The function of _province_tag is to extract the province name from a given line string by removing any parentheses.

parameters: 
· l: A string representing a line that may contain a province tag enclosed in parentheses or as a standalone word.

Code Description: The description of this Function.
The function _province_tag takes a single argument, `l`, which is expected to be a string. It splits the string into words using space as a delimiter and iterates through each word. If a word is not an opening or closing parenthesis ('(' or ')'), it returns that word immediately as the province tag. The function assumes that the first non-parenthesis word in the line is the province name. If no such word is found, it raises a ValueError indicating that no province was found for the given line.

In the context of the project, this function is called by `_tag_to_id` within the same file `province_order.py`. The `_tag_to_id` function processes lines from an mdf content string to map each province tag (extracted using _province_tag) to a unique identifier. This mapping is stored in a dictionary where keys are province tags and values are integers representing their order of appearance.

Note: Points to note about the use of the code
The input line `l` should contain at least one word that does not represent an opening or closing parenthesis for the function to return a valid province tag. If the line only contains parentheses or is empty, the function will raise a ValueError.

Output Example: Mock up a possible appearance of the code's return value.
If the input string `l` is "(province) name", the function will return "name". Similarly, if `l` is "another_name (description)", it will return "another_name". If `l` is "() ()", a ValueError will be raised.
## FunctionDef province_name_to_id(map_mdf)
**province_name_to_id**: The function of province_name_to_id is to retrieve a dictionary mapping province names to their respective IDs based on the specified map configuration.

parameters: 
· map_mdf: Represents the type of map configuration for which the province name to ID mapping should be generated. It accepts values from the MapMDF enumeration, specifically MapMDF.STANDARD_MAP or MapMDF.BICOASTAL_MAP. The default value is MapMDF.STANDARD_MAP.

Code Description: The function province_name_to_id takes a single parameter, map_mdf, which specifies the desired map configuration. It calls the get_mdf_content function with the provided map_mdf argument to obtain the MDF content string corresponding to the specified map type. This MDF content is then passed to the _tag_to_id function, which processes the content to generate a dictionary mapping province names (tags) to unique integer identifiers. The resulting dictionary is returned by province_name_to_id.

In the context of the project, province_name_to_id serves as a utility function that provides a consistent way to refer to provinces using unique IDs based on the specified map configuration. This function relies on get_mdf_content to fetch the appropriate MDF data and _tag_to_id to parse this data into a usable dictionary format.

Note: When calling province_name_to_id, ensure that you pass one of the predefined constants from the MapMDF enumeration (STANDARD_MAP or BICOASTAL_MAP) to avoid runtime errors. The choice of map configuration should align with the specific requirements of the function being called and the data it processes.

Output Example: Mock up a possible appearance of the code's return value.
If map_mdf is set to MapMDF.STANDARD_MAP, and the MDF content contains lines such as:
```
(header information)
(province) England
Spain (description)
(another_province) France
(trailing information)
```
The function province_name_to_id will return a dictionary like:
```python
{
    'England': 0,
    'Spain': 1,
    'France': 2
}
```
## FunctionDef province_id_to_home_sc_power
**province_id_to_home_sc_power**: The function of province_id_to_home_sc_power is to map each province ID to its corresponding home superpower based on the standard map configuration.

parameters: This function does not accept any parameters.

Code Description: The function retrieves the MDF content for the standard map configuration using the get_mdf_content function with MapMDF.STANDARD_MAP as the argument. It then extracts the line containing information about home superpowers from this content. A dictionary mapping province tags to their respective IDs is created using the _tag_to_id function, which processes the same MDF content.

The function iterates over each word in the extracted home superpower line. If a word represents a province (i.e., it exists in the tag_to_id dictionary), it assigns the current power index to this province ID in the id_to_power dictionary. The power index is incremented whenever a new power tag is encountered, which indicates the start of a new set of provinces associated with that power.

In summary, province_id_to_home_sc_power constructs a mapping from each province's unique identifier to the home superpower it belongs to, based on the standard map configuration.

Note: This function relies on the correct format and content of the MDF file for accurate results. The MDF content must include a line specifying home superpowers in a format that can be parsed by this function.

Output Example: A possible return value of the function could be:
```python
{
    utils.ProvinceID(0): 0,
    utils.ProvinceID(1): 0,
    utils.ProvinceID(2): 1,
    utils.ProvinceID(3): 1,
    utils.ProvinceID(4): 2,
}
```
In this example, provinces with IDs 0 and 1 are associated with home superpower 0, provinces with IDs 2 and 3 are associated with home superpower 1, and province with ID 4 is associated with home superpower 2.
## FunctionDef _tag_to_id(mdf_content)
**_tag_to_id**: The function of _tag_to_id is to map province tags extracted from lines of MDF content to unique integer identifiers.

parameters: 
· mdf_content: A string representing the content of an MDF file, which includes lines with province information enclosed in parentheses or as standalone words.

Code Description: The function _tag_to_id processes the input MDF content by splitting it into individual lines and iterating over all lines except for the first four and the last one. For each line, it calls the helper function `_province_tag` to extract the province tag. It then assigns a unique integer identifier to this tag, starting from 0 and incrementing by 1 for each new tag found. The mapping of tags to identifiers is stored in a dictionary where keys are the province tags and values are their corresponding integer identifiers.

In the context of the project, _tag_to_id serves as a foundational function that provides a consistent way to refer to provinces using unique IDs. This function is called by several other functions within the same file `province_order.py`, including `province_name_to_id`, `province_id_to_home_sc_power`, `build_adjacency`, `topological_index`, and `fleet_adjacency_map`. These functions rely on the mapping generated by _tag_to_id to perform their specific tasks, such as building adjacency matrices or determining topological indices.

Note: Points to note about the use of the code
The input string mdf_content should be a valid MDF file content with lines that contain province tags. The function assumes that each line relevant for tag extraction follows a consistent format where the first non-parenthesis word is the province name. If the format is not adhered to, the function `_province_tag` may raise a ValueError indicating that no province was found for a given line.

Output Example: Mock up a possible appearance of the code's return value.
If the input string mdf_content contains lines such as:
```
(header information)
(province) name1
name2 (description)
(another_province) name3
(trailing information)
```
The function _tag_to_id will return a dictionary like:
```python
{
    'name1': 0,
    'name2': 1,
    'name3': 2
}
```
## FunctionDef build_adjacency(mdf_content)
**build_adjacency**: The function of build_adjacency is to construct an adjacency matrix from MDF content that represents connections between provinces.

parameters: 
· mdf_content: A string representing the content of an MDF file, which includes lines with province information and their connectivity details.

Code Description: The function build_adjacency processes the input MDF content by first mapping province tags to unique integer identifiers using the _tag_to_id function. It then initializes a zero matrix of size num_provinces-by-num_provinces, where num_provinces is determined from the maximum value in the tag-to-id mapping plus one. The function iterates over each line of the MDF content (excluding the first four and last lines) to extract province connectivity information. For each line, it identifies the sender province and any receiver provinces that are reachable by either an army or a fleet, excluding keywords 'AMY' and 'FLT'. It updates the adjacency matrix to reflect these connections by setting the corresponding entries to 1.0. Additionally, if the sender province has multiple coasts (indicated by a tag longer than three characters), it establishes bidirectional connectivity between the coast-specific tag and its base land province.

Note: Points to note about the use of the code
The input string mdf_content should be a valid MDF file content with lines that contain province tags and their connections. The function assumes that each line relevant for connectivity extraction follows a consistent format where the first word is the sender province, followed by receiver provinces or keywords 'AMY' and 'FLT'. If the format is not adhered to, the function may incorrectly interpret the data, leading to an inaccurate adjacency matrix.

Output Example: Mock up a possible appearance of the code's return value.
If the input string mdf_content contains lines such as:
```
(header information)
(province) SPA
SPA (Spain) ENG FRA
SPA.NW ENG
SPA.NE FRA
SPA.S EA
ENG SPA NTH
FRA SPA NW
(trailing information)
```
The function build_adjacency will return an adjacency matrix like:
```python
array([[0., 1., 1., 0.],
       [1., 0., 1., 0.],
       [1., 1., 0., 0.],
       [0., 0., 0., 0.]])
```
In this example, the provinces are mapped as follows: SPA (Spain) -> 0, ENG (England) -> 1, FRA (France) -> 2, and EA (East) -> 3. The matrix indicates that Spain is connected to England and France through its coasts, while England is connected to Spain and the NTH province, and France is connected to Spain and its NW coast.
## FunctionDef topological_index(mdf_content, topological_order)
**topological_index**: The function of topological_index is to convert a list of province names into their corresponding unique integer identifiers based on an MDF content string.

parameters: 
· mdf_content: A string representing the content of an MDF file, which includes lines with province information enclosed in parentheses or as standalone words.
· topological_order: A sequence (e.g., list) of strings where each string is a province name that needs to be converted into its unique integer identifier.

Code Description: The function topological_index first calls the helper function _tag_to_id, passing it the mdf_content parameter. This call generates a dictionary mapping each province tag found in the MDF content to a unique integer identifier. The function then iterates over the provided topological_order sequence, using this dictionary to look up and retrieve the corresponding integer identifier for each province name in the order specified. These identifiers are collected into a list, which is returned as the final result.

In the context of the project, topological_index relies on _tag_to_id to establish a consistent mapping between province names and their unique IDs. This mapping is essential for maintaining a standardized reference across various functions that require numerical identifiers for provinces. By leveraging this mapping, topological_index ensures that the sequence of province names provided in topological_order is accurately translated into a sequence of corresponding integer identifiers.

Note: Points to note about the use of the code
The input string mdf_content should be a valid MDF file content with lines that contain province tags formatted consistently. The function assumes that each line relevant for tag extraction follows a format where the first non-parenthesis word is the province name. If this format is not adhered to, the function _province_tag may raise a ValueError indicating that no province was found for a given line.

Output Example: Mock up a possible appearance of the code's return value.
If the input string mdf_content contains lines such as:
```
(header information)
(province) name1
name2 (description)
(another_province) name3
(trailing information)
```
And the topological_order is ['name2', 'name1', 'name3'], the function _tag_to_id will generate a dictionary like:
```python
{
    'name1': 0,
    'name2': 1,
    'name3': 2
}
```
The function topological_index will then return a list of identifiers corresponding to the order specified in topological_order:
```python
[1, 0, 2]
```
## FunctionDef fleet_adjacency_map
**fleet_adjacency_map**: The function of fleet_adjacency_map is to build a mapping for valid fleet movements between areas based on the bicoastal map configuration.

parameters: The parameters of this Function.
· No explicit parameters are required as the function uses a predefined map configuration (MapMDF.BICOASTAL_MAP).

Code Description: The description of this Function.
The fleet_adjacency_map function constructs a dictionary that maps each area to a list of areas where fleets can move. It begins by retrieving the MDF content for the bicoastal map configuration using the get_mdf_content function with MapMDF.BICOASTAL_MAP as the argument. The function then creates a mapping from province tags to unique integer identifiers using the _tag_to_id function.

The MDF content is split into lines, and the function processes each line except for the first four and the last one. Each relevant line represents an edge in the adjacency map, starting with a province tag followed by a list of adjacent provinces that can be reached by fleet movement. The function identifies these tags and converts them to their corresponding integer identifiers using the previously created mapping.

The function iterates through each line, skipping non-relevant characters like parentheses and empty strings. It identifies the start province and initializes an empty list for its adjacency in the fleet_adjacency dictionary. As it processes each subsequent tag on the same line, it checks for the presence of the 'FLT' keyword, which indicates that the following tags represent valid fleet movement targets. Once 'FLT' is found, all subsequent tags are added to the adjacency list for the start province.

Note: Points to note about the use of the code
The function does not accept any parameters and always uses the bicoastal map configuration. Ensure that the MDF content for the bicoastal map is correctly formatted and contains the necessary information for accurate mapping. The output dictionary will only include areas that have valid fleet movement targets as specified in the MDF content.

Output Example: Mock up a possible appearance of the code's return value.
If the MDF content for the bicoastal map includes lines such as:
```
AREAS
(
  ENG England
  SPA Spain
  FRA France
)
FLEET_ADJACENCY (
  ENG (SPA FLT)
  SPA (ENG FLT)
  FRA (SPA FLT)
)
...
```
The function fleet_adjacency_map will return a dictionary like:
```python
{
    0: [1],  # Assuming 'ENG' is mapped to 0 and 'SPA' is mapped to 1
    1: [0, 2],  # Assuming 'SPA' is mapped to 1, 'ENG' is mapped to 0, and 'FRA' is mapped to 2
    2: [1]  # Assuming 'FRA' is mapped to 2 and 'SPA' is mapped to 1
}
```
This dictionary indicates that fleets can move from ENG (ID 0) to SPA (ID 1), from SPA (ID 1) to both ENG (ID 0) and FRA (ID 2), and from FRA (ID 2) to SPA (ID 1).
