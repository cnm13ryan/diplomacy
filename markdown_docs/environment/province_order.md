## ClassDef MapMDF
**MapMDF**: The function of MapMDF is to define an enumeration of map types.
**attributes**: The attributes of this Class.
· STANDARD_MAP: represents a standard map with a value of 0
· BICOASTAL_MAP: represents a bicoastal map with a value of 1
**Code Description**: The description of this Class. MapMDF is an enumeration class that defines two types of maps, namely STANDARD_MAP and BICOASTAL_MAP. This class is used in various functions throughout the project to determine which type of map to use for different operations. For instance, the get_mdf_content function uses the MapMDF enum to return the content of either a standard map or a bicoastal map based on the input parameter. Similarly, the province_name_to_id and fleet_adjacency_map functions also utilize the MapMDF enum to determine which type of map to use for their respective operations. The use of an enumeration class provides a clear and concise way to define and work with different map types, making the code more readable and maintainable.
**Note**: Points to note about the use of the code. When using the MapMDF enum, it is essential to ensure that the correct map type is used for each operation, as this can affect the outcome of the functions that utilize it. Additionally, any new map types added to the enum should be thoroughly tested to ensure compatibility with existing functions and operations.
## FunctionDef get_mdf_content(map_mdf)
**get_mdf_content**: The function of get_mdf_content is to retrieve the content of a map based on the specified map type.
**parameters**: The parameters of this Function.
· map_mdf: This parameter is an instance of the MapMDF enumeration class, which defines the type of map to retrieve content for. It defaults to MapMDF.STANDARD_MAP if not provided.
**Code Description**: The get_mdf_content function takes a map_mdf parameter and uses it to determine which type of map content to return. If the map_mdf is set to MapMDF.STANDARD_MAP, it returns the content of the standard map. If the map_mdf is set to MapMDF.BICOASTAL_MAP, it returns the content of the bicoastal map. If an unknown map_mdf value is provided, it raises a ValueError with a message indicating that the map_mdf is unknown. This function is used by other functions in the project, such as province_name_to_id, province_id_to_home_sc_power, and fleet_adjacency_map, to retrieve the necessary map content for their operations.
**Note**: Points to note about the use of the code. When using the get_mdf_content function, it is essential to ensure that a valid MapMDF enum value is provided as the map_mdf parameter. The function's return value should be handled accordingly based on the specific requirements of the calling function. Additionally, any modifications to the MapMDF enum or the get_mdf_content function itself should be thoroughly tested to ensure compatibility with existing functions and operations in the project.
**Output Example**: The output of this function will be a string representing the content of the specified map type, such as "_STANDARD_MAP_MDF_CONTENT" or "_BICOASTAL_MAP_MDF_CONTENT".
## FunctionDef _province_tag(l)
**_province_tag**: The function of _province_tag is to extract the province tag from a given line of text.
**parameters**: The parameters of this Function.
· l: a string representing the line of text from which the province tag will be extracted
**Code Description**: This function takes a string as input, splits it into words, and then iterates over each word. It checks if the word is not a parenthesis and returns the first word that meets this condition. If no such word is found after checking all words, it raises a ValueError with a message indicating that no province was found for the given line. The function is used by _tag_to_id to extract province tags from lines of text and map them to unique identifiers.
**Note**: The function assumes that the input string contains at least one word that is not a parenthesis, and that the first such word is the province tag. It also assumes that the input string does not contain any leading or trailing whitespace. Users should be aware that this function will raise an exception if it cannot find a province tag in the input string.
**Output Example**: If the input string is "ProvinceA (some additional text)", the output would be "ProvinceA".
## FunctionDef province_name_to_id(map_mdf)
**province_name_to_id**: The function of province_name_to_id is to retrieve a dictionary that maps province names to their corresponding unique identifiers.
**parameters**: The parameters of this Function.
· map_mdf: This parameter is an instance of the MapMDF enumeration class, which defines the type of map to use for retrieving the province name to identifier mappings. It defaults to MapMDF.STANDARD_MAP if not provided.
**Code Description**: The province_name_to_id function utilizes the get_mdf_content function to retrieve the content of a map based on the specified map_mdf parameter. This content is then passed to the _tag_to_id function, which creates a dictionary mapping province names to unique identifiers. The resulting dictionary is returned by the province_name_to_id function. The function's operation relies on the correct functioning of its callees, get_mdf_content and _tag_to_id, to produce the desired output.
**Note**: When using the province_name_to_id function, it is essential to ensure that a valid MapMDF enum value is provided as the map_mdf parameter. Additionally, users should be aware of the assumptions made by the _tag_to_id function regarding the format of the input string and the extraction of province tags.
**Output Example**: The output of this function will be a dictionary where each key represents a province name and its corresponding value is a unique identifier, such as {"ProvinceA": 0, "ProvinceB": 1, ...}.
## FunctionDef province_id_to_home_sc_power
**province_id_to_home_sc_power**: The function of province_id_to_home_sc_power is to determine which power is associated with each province as its home supply center.

**parameters**: None

**Code Description**: This function retrieves the content of a standard map using the get_mdf_content function from the MapMDF enumeration class. It then extracts the third line of the map content, which contains information about the home supply centers for each power. The function splits this line into individual words and iterates over them to identify province tags and power identifiers. Province tags are mapped to their corresponding unique identifiers using the _tag_to_id function, while power identifiers are assigned a numerical value based on their order of appearance. The function returns a dictionary where each key is a province identifier and its corresponding value is the associated power.

The get_mdf_content function is used to retrieve the map content, and the MapMDF enumeration class provides a way to specify the type of map to use. In this case, the STANDARD_MAP is used. The _tag_to_id function plays a crucial role in mapping province tags to unique identifiers, which are then used to create the dictionary returned by the province_id_to_home_sc_power function.

**Note**: It is essential to note that this function assumes the map content has a specific format and that the get_mdf_content and _tag_to_id functions behave as expected. Any changes to these functions or the map content may affect the accuracy of the results produced by the province_id_to_home_sc_power function.

**Output Example**: The output of this function would be a dictionary where each key is a unique province identifier, and its corresponding value is an integer representing the associated power, such as {1: 0, 2: 0, 3: 1, ...}, indicating that provinces with identifiers 1 and 2 are associated with power 0, while province with identifier 3 is associated with power 1.
## FunctionDef _tag_to_id(mdf_content)
**_tag_to_id**: The function of _tag_to_id is to create a dictionary mapping province tags to unique identifiers based on the content of a given string.
**parameters**: The parameters of this Function.
· mdf_content: a string representing the content from which the province tags will be extracted and mapped to unique identifiers
**Code Description**: This function takes the input string, splits it into lines, and then iterates over each line, excluding the first four and last lines. For each line, it extracts the province tag using the _province_tag function and assigns a unique identifier to it based on the order of appearance. The function returns a dictionary containing these mappings. The _tag_to_id function is used by various other functions in the project, including province_name_to_id, province_id_to_home_sc_power, build_adjacency, topological_index, and fleet_adjacency_map, to establish relationships between provinces and their corresponding identifiers.
**Note**: It is essential to note that this function relies on the _province_tag function to correctly extract the province tags from each line. Additionally, the input string should have a specific format, with the first four and last lines being excluded from the processing. Users should be aware of these assumptions when using this function.
**Output Example**: If the input string contains lines such as "ProvinceA (some additional text)", "ProvinceB (more text)", and so on, the output would be a dictionary like {"ProvinceA": 0, "ProvinceB": 1, ...}, where each province tag is mapped to a unique identifier based on its order of appearance.
## FunctionDef build_adjacency(mdf_content)
**build_adjacency**: The function of build_adjacency is to construct an adjacency matrix from the content of a map file, indicating which provinces are adjacent to each other based on possible movements of armies or fleets.

**parameters**: The parameters of this Function.
· mdf_content: a string representing the content of a map file, used as input to determine the adjacency relationships between provinces.

**Code Description**: This function initializes an empty dictionary using the _tag_to_id function to map province tags to unique identifiers. It then determines the total number of provinces based on these identifiers and creates a zero-filled adjacency matrix with dimensions equal to the number of provinces. The function proceeds to parse the input string line by line, excluding the first four and last lines, to extract information about province connections. For each connection found, it updates the corresponding entries in the adjacency matrix to indicate that the provinces are adjacent. Specifically, if a province has multiple coasts, it is considered adjacent to all provinces reachable from any of its coasts. The function ultimately returns this constructed adjacency matrix.

The _tag_to_id function plays a crucial role in this process by providing the necessary mappings between province tags and unique identifiers, which are essential for correctly populating the adjacency matrix. By leveraging this mapping, the build_adjacency function can accurately determine the relationships between different provinces based on their connections as defined in the input map file.

**Note**: It is important to note that the input string is expected to have a specific format, with certain lines being excluded from processing. Additionally, the function's reliance on the _tag_to_id function means that any errors or inconsistencies in the mapping of province tags to identifiers could potentially affect the accuracy of the resulting adjacency matrix.

**Output Example**: The output of this function would be a 2D numpy array representing the adjacency matrix, where each entry [i][j] is 1.0 if province i is adjacent to province j, and 0.0 otherwise. For instance, if there are three provinces A, B, and C, and A is connected to B, while B is connected to C, the output might look like:
```
[[0.0, 1.0, 0.0],
 [1.0, 0.0, 1.0],
 [0.0, 1.0, 0.0]]
```
## FunctionDef topological_index(mdf_content, topological_order)
**topological_index**: The function of topological_index is to generate a sequence of unique province identifiers based on a given topological order and content string.
**parameters**: The parameters of this Function.
· mdf_content: a string representing the content from which the province tags will be extracted and mapped to unique identifiers
· topological_order: a sequence of strings representing the desired order of provinces
**Code Description**: This function works by first creating a dictionary mapping province tags to unique identifiers using the _tag_to_id function, which takes the mdf_content string as input. The _tag_to_id function returns a dictionary where each key is a province tag and its corresponding value is a unique identifier. The topological_index function then uses this dictionary to generate a sequence of unique province identifiers based on the given topological_order. It does this by iterating over each province in the topological_order and looking up its corresponding unique identifier in the dictionary created by _tag_to_id.
**Note**: It is essential to note that the correctness of the output depends on the accuracy of the _tag_to_id function, which relies on a specific format of the input string. Users should be aware of this assumption when using the topological_index function. Additionally, the topological_order sequence should contain valid province tags that can be found in the mdf_content string.
**Output Example**: If the input string contains lines such as "ProvinceA (some additional text)", "ProvinceB (more text)", and so on, and the topological_order is ["ProvinceA", "ProvinceB"], the output would be a sequence of unique identifiers like [0, 1], where each identifier corresponds to a province in the given order.
## FunctionDef fleet_adjacency_map
**fleet_adjacency_map**: The function of fleet_adjacency_map is to build a mapping for valid fleet movements between areas.
**parameters**: None
**Code Description**: This function constructs a dictionary that maps each area to a list of adjacent areas where fleets can move. It starts by retrieving the content of a bicoastal map using the get_mdf_content function with MapMDF.BICOASTAL_MAP as the argument. The retrieved content is then split into lines, and for each line (excluding the first four and last lines), it extracts the province tags and creates a mapping between these tags and their corresponding area identifiers using the _tag_to_id function. The function then iterates over the extracted provinces and checks for the presence of a 'FLT' tag, which indicates fleet movement. If the 'FLT' tag is found, the subsequent provinces are added to the list of adjacent areas for the current province. Finally, the function returns the constructed dictionary, which represents the valid fleet movements between areas.
**Note**: It is essential to note that this function relies on the correct implementation of the get_mdf_content and _tag_to_id functions to produce accurate results. Additionally, the input map content should be in a specific format for the function to work correctly.
**Output Example**: The output of this function will be a dictionary where each key represents an area identifier, and its corresponding value is a list of adjacent area identifiers where fleets can move, such as {AreaID1: [AreaID2, AreaID3], AreaID4: [AreaID5, AreaID6]}.
