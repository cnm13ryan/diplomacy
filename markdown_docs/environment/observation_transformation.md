## ClassDef ObservationTransformState
**ObservationTransformState**: The function of ObservationTransformState is to store and manage the state of observations in a structured format.
**attributes**: The attributes of this Class.
· previous_board_state: A numpy array representing the board state at the last moves phase.
· last_action_board_state: A numpy array representing the most recent board state.
· actions_since_previous_moves_phase: A numpy array recording the actions taken since the last moves phase.
· last_phase_was_moves: A boolean indicating whether the last phase was a moves phase.

**Code Description**: The ObservationTransformState class is designed to hold and manage the state of observations, which includes the previous board state, the current board state, actions taken since the last moves phase, and an indicator of whether the last phase was a moves phase. This class is utilized by other components in the project, such as the update_state function, which updates the ObservationTransformState based on new observations and returns the updated state. The GeneralObservationTransformer class also relies on ObservationTransformState to construct initial network observations and states, and to transform observations into the format required by network policies.

The ObservationTransformState is used in conjunction with other functions to maintain a consistent and accurate representation of the observation state throughout the project. For instance, when the update_state function updates the actions_since_previous_moves_phase attribute, it ensures that the actions are properly recorded and updated based on the new observation. Similarly, when the GeneralObservationTransformer class uses ObservationTransformState to construct initial network observations, it relies on the attributes of this class to provide a comprehensive representation of the current state.

The relationship between ObservationTransformState and its callers is one of dependency, where the callers rely on ObservationTransformState to provide a structured format for storing and managing observation states. The update_state function and GeneralObservationTransformer class use ObservationTransformState to ensure that the observation state is accurately represented and updated throughout the project.

**Note**: When using ObservationTransformState, it is essential to ensure that the attributes are properly initialized and updated to maintain a consistent representation of the observation state. Additionally, the callers of ObservationTransformState should be aware of the dependencies between this class and other components in the project to ensure accurate and reliable functionality.
## FunctionDef update_state(observation, prev_state)
**update_state**: The function of update_state is to return an updated state for alliance features based on the provided observation and previous state.
**parameters**: The parameters of this Function.
· observation: An instance of utils.Observation, which represents the current observation from the environment.
· prev_state: An optional instance of ObservationTransformState, which stores the previous state of observations.

**Code Description**: This function plays a crucial role in maintaining and updating the state of observations. It takes into account the current observation and the previous state to determine the updated state. If no previous state is provided, it initializes the necessary variables, including last_phase_was_moves, last_board_state, previous_board_state, and actions_since_previous_moves_phase. The function then updates these variables based on the current observation, considering factors such as the type of actions taken and whether the last phase was a moves phase. It utilizes other functions like action_breakdown from action_utils to process the actions in the observation. The updated state is returned as an instance of ObservationTransformState.

The update_state function is closely related to its callers, such as GeneralObservationTransformer's initial_observation_transform method, which relies on update_state to obtain the next state based on the current observation and previous state. This relationship highlights the importance of update_state in maintaining a consistent and accurate representation of the observation state throughout the project.

**Note**: When using the update_state function, it is essential to ensure that the provided observation and previous state are valid and properly initialized. Additionally, the caller should be aware of the dependencies between this function and other components in the project to guarantee accurate and reliable functionality.

**Output Example**: The return value of the update_state function will be an instance of ObservationTransformState, containing the updated previous_board_state, last_action_board_state, actions_since_previous_moves_phase, and last_phase_was_moves. For example, it might look like:
ObservationTransformState(
  previous_board_state=np.array([...]), 
  last_action_board_state=np.array([...]), 
  actions_since_previous_moves_phase=np.array([...]), 
  last_phase_was_moves=True
)
## ClassDef TopologicalIndexing
**TopologicalIndexing**: The function of TopologicalIndexing is to define an enumeration of possible topological indexing methods used in observation transformations.
**attributes**: The attributes of this Class.
· NONE: Represents no topological indexing, where areas are ordered according to their index in the observation.
· MILA: Represents a specific topological indexing method, similar to the one used in Pacquette et al.

**Code Description**: This Class is an enumeration that provides two possible values for topological indexing: NONE and MILA. The NONE value indicates that no topological indexing should be applied, and areas should be ordered according to their index in the observation. The MILA value represents a specific topological indexing method, which is used to determine the order of areas when producing orders from different areas. This enumeration is used by the GeneralObservationTransformer class, specifically in its constructor and _topological_index method. In the constructor, the topological_indexing parameter is set to one of the values defined in this Class, determining how areas will be ordered. The _topological_index method returns the order of areas based on the chosen topological indexing method.

**Note**: When using this Class, it is essential to understand the implications of each topological indexing method on the observation transformation process. The NONE value provides a straightforward ordering based on area indices, while the MILA value applies a specific ordering that may be more suitable for certain applications or models, such as those described in Pacquette et al. Incorrectly choosing a topological indexing method may lead to unexpected behavior or incorrect results in downstream processing or modeling steps.
## ClassDef GeneralObservationTransformer
**GeneralObservationTransformer**: The function of GeneralObservationTransformer is to transform observations from an environment into a format required by network policies.

**attributes**: The attributes of this Class.
· rng_key: A Jax random number generator key, for use if an observation transformation is ever stochastic.
· board_state: Flag for whether to include the current board state in the observation.
· last_moves_phase_board_state: Flag for whether to include the board state at the start of the last moves phase in the observation.
· actions_since_last_moves_phase: Flag for whether to include the actions since the last moves phase in the observation.
· season: Flag for whether to include the current season in the observation.
· build_numbers: Flag for whether to include the number of builds/disbands each player has in the observation.
· topological_indexing: When choosing unit actions in sequence, the order they are chosen is determined by this config.
· areas: Flag for whether to include a vector of length NUM_AREAS, which is True in the area that the next unit-action will be chosen for.
· last_action: Flag for whether to include the action chosen in the previous unit-action selection in the input.
· legal_actions_mask: Flag for whether to include a mask of which actions are legal in the observation.
· temperature: Flag for whether to include a sampling temperature in the neural network input.

**Code Description**: The GeneralObservationTransformer class is designed to transform observations from an environment into a format required by network policies. It has several attributes that determine what information is included in the transformed observation. The class has several methods, including initial_observation_spec, initial_observation_transform, step_observation_spec, and step_observation_transform, which are used to construct the transformed observation. The initial_observation_transform method constructs the initial network observations and state, while the step_observation_transform method converts raw step observations from the environment into network inputs. The class also has an observation_transform method that transforms the observation into the format required by network policies.

The GeneralObservationTransformer class is initialized with several parameters, including rng_key, board_state, last_moves_phase_board_state, actions_since_last_moves_phase, season, build_numbers, topological_indexing, areas, last_action, legal_actions_mask, and temperature. These parameters determine what information is included in the transformed observation.

The initial_observation_spec method returns a spec for the output of the initial observation transform, while the step_observation_spec method returns a spec for the output of the step observation transform. The observation_spec method returns a spec for the output of the observation transform.

The class also has several other methods, including zero_observation and _topological_index, which are used to generate a zero observation and determine the order in which to produce orders from different areas, respectively.

**Note**: When using the GeneralObservationTransformer class, it is important to note that the topological_indexing attribute determines the order in which unit actions are chosen. If this attribute is set to TopologicalIndexing.MILA, the order will be determined by the mila_topological_index. Otherwise, the order will be determined by the observation.

**Output Example**: The output of the GeneralObservationTransformer class will depend on the specific parameters used to initialize it. However, in general, the output will include several dictionaries and arrays that contain information about the current state of the environment, such as the board state, actions since the last moves phase, season, build numbers, areas, last action, legal actions mask, and temperature. For example:
```python
{
    'initial_observation': {
        'board_state': [...],
        'actions_since_last_moves_phase': [...],
        'season': ...,
        'build_numbers': [...]
    },
    'step_observations': [
        {
            'areas': [...],
            'last_action': ...,
            'legal_actions_mask': [...],
            'temperature': ...
        },
        ...
    ],
    'sequence_lengths': [...]
}
```
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to configure the fields that the GeneralObservationTransformer will return as part of the observation transformation process.
**parameters**: The parameters of this Function.
· rng_key: A Jax random number generator key, used if an observation transformation is stochastic.
· board_state: A flag indicating whether to include the current board state in the observation.
· last_moves_phase_board_state: A flag indicating whether to include the board state at the start of the last moves phase in the observation.
· actions_since_last_moves_phase: A flag indicating whether to include the actions since the last moves phase in the observation.
· season: A flag indicating whether to include the current season in the observation.
· build_numbers: A flag indicating whether to include the number of builds or disbands each player has in the observation.
· topological_indexing: An enumeration value determining the ordering of areas when producing orders from different areas, with possible values being NONE or MILA.
· areas: A flag indicating whether to include a vector of length NUM_AREAS, which is True in the area that the next unit-action will be chosen for.
· last_action: A flag indicating whether to include the action chosen in the previous unit-action selection in the input.
· legal_actions_mask: A flag indicating whether to include a mask of which actions are legal in the observation.
· temperature: A flag indicating whether to include a sampling temperature in the neural network input.
**Code Description**: This function initializes the GeneralObservationTransformer by setting the specified parameters, which determine the fields that will be included in the observation. The topological_indexing parameter is an enumeration value that determines how areas will be ordered when producing orders from different areas. The other parameters are flags that indicate whether specific information should be included in the observation, such as the current board state, actions since the last moves phase, and the current season. The function stores these parameters as instance variables, which can then be used to configure the observation transformation process.
**Note**: It is essential to carefully consider the implications of each parameter on the observation transformation process when using this function. Incorrectly choosing a topological indexing method or including unnecessary information in the observation may lead to unexpected behavior or incorrect results in downstream processing or modeling steps.
**Output Example**: This function does not return any value, as it is an initializer that configures the instance variables of the GeneralObservationTransformer class. However, the configured instance can be used to transform observations based on the specified parameters.
***
### FunctionDef initial_observation_spec(self, num_players)
**initial_observation_spec**: The function of initial_observation_spec is to return a spec for the output of initial_observation_transform.
**parameters**: The parameters of this Function.
· num_players: an integer representing the number of players, which is used to determine the shape of certain arrays in the returned spec
**Code Description**: This function generates a spec for the initial observation transform by creating an ordered dictionary and conditionally adding specs for various components based on the presence of specific attributes. The attributes that may be included are board_state, last_moves_phase_board_state, actions_since_last_moves_phase, season, and build_numbers. Each of these components has a corresponding spec with a specific shape and data type. For example, if self.board_state is present, the function adds a spec for 'board_state' with a shape of (utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH) and a dtype of np.float32. The function returns this spec as an ordered dictionary. In the context of the project, this function is called by observation_spec to generate the initial observation spec, which is then combined with other specs to form the complete observation spec.
**Note**: It's essential to note that the returned spec is dependent on the presence of specific attributes, so the actual structure and content may vary. Additionally, understanding the context in which this function is called, specifically by observation_spec, can provide insight into its purpose and usage.
**Output Example**: A possible appearance of the code's return value could be an ordered dictionary with specs for 'board_state', 'last_moves_phase_board_state', 'actions_since_last_moves_phase', 'season', and 'build_numbers', where each spec has a specific shape and data type, such as:
{
    'board_state': specs.Array((utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH), dtype=np.float32),
    'last_moves_phase_board_state': specs.Array((utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH), dtype=np.float32),
    'actions_since_last_moves_phase': specs.Array((utils.NUM_AREAS, 3), dtype=np.int32),
    'season': specs.Array((), dtype=np.int32),
    'build_numbers': specs.Array((num_players,), dtype=np.int32)
}
***
### FunctionDef initial_observation_transform(self, observation, prev_state)
**initial_observation_transform**: The function of initial_observation_transform is to construct initial network observations and state based on the provided observation and previous state.
**parameters**: The parameters of this Function.
· observation: Parsed observation from environment, which is an instance of utils.Observation.
· prev_state: Previous ObservationTransformState, which stores the state of observations in a structured format.

**Code Description**: This function plays a crucial role in initializing the network observations and state. It takes into account the current observation and the previous state to determine the initial observations and next state. The function first updates the state by calling the update_state function, which returns an updated ObservationTransformState based on the provided observation and previous state. Then, it constructs the initial observations by selectively including certain fields from the observation, such as board state, last moves phase board state, actions since last moves phase, season, and build numbers, depending on the configuration flags set in the class instance. These fields are then stored in an ordered dictionary, which is returned along with the next state.

The initial_observation_transform function is closely related to its caller, GeneralObservationTransformer's observation_transform method, which relies on this function to obtain the initial observations and next state based on the current observation and previous state. This relationship highlights the importance of initial_observation_transform in maintaining a consistent and accurate representation of the observation state throughout the project.

The function also depends on the update_state function, which updates the ObservationTransformState based on new observations and returns the updated state. The ObservationTransformState class is designed to hold and manage the state of observations, which includes the previous board state, the current board state, actions taken since the last moves phase, and an indicator of whether the last phase was a moves phase.

**Note**: When using the initial_observation_transform function, it is essential to ensure that the provided observation and previous state are valid and properly initialized. Additionally, the caller should be aware of the dependencies between this function and other components in the project to guarantee accurate and reliable functionality.

**Output Example**: The return value of the initial_observation_transform function will be a tuple containing an ordered dictionary representing the initial observations and an instance of ObservationTransformState representing the next state. For example, it might look like:
(initial_observation={'board_state': np.array([...]), 'last_moves_phase_board_state': np.array([...]), ...}, next_state=ObservationTransformState(previous_board_state=np.array([...]), last_action_board_state=np.array([...]), actions_since_previous_moves_phase=np.array([...]), last_phase_was_moves=True))
***
### FunctionDef step_observation_spec(self)
**step_observation_spec**: The function of step_observation_spec is to return a specification for the output of step_observation_transform.
**parameters**: The parameters of this Function.
· self: A reference to the current instance of the class
**Code Description**: This function generates a specification for the output of step_observation_transform based on the attributes of the class instance. It checks if certain attributes are set, such as areas, last_action, legal_actions_mask, and temperature, and adds corresponding specifications to an ordered dictionary. The specifications are defined using specs.Array with specific shapes and data types. For example, if self.areas is True, it adds a specification for 'areas' with shape (utils.NUM_AREAS,) and dtype bool. The function returns the constructed specification dictionary. In the context of the project, this function is called by observation_spec to generate the step observation specification, which is then used in observation_transform to transform the observation into the required format.
**Note**: The attributes self.areas, self.last_action, self.legal_actions_mask, and self.temperature determine the structure of the output specification. If any of these attributes are not set, the corresponding specifications will not be included in the output dictionary.
**Output Example**: A possible appearance of the code's return value could be {'areas': specs.Array(shape=(10,), dtype=bool), 'last_action': specs.Array(shape=(), dtype=np.int32), 'legal_actions_mask': specs.Array(shape=(100,), dtype=np.uint8), 'temperature': specs.Array(shape=(1,), dtype=np.float32)}, depending on the values of the class instance attributes.
***
### FunctionDef step_observation_transform(self, transformed_initial_observation, legal_actions, slot, last_action, area, step_count, previous_area, temperature)
**step_observation_transform**: The function of step_observation_transform is to convert raw step observations from the diplomacy environment into network inputs.

**parameters**: The parameters of this Function.
· transformed_initial_observation: A dictionary containing the initial observation made with the same configuration, where each key maps to a jnp.ndarray value.
· legal_actions: A sequence of jnp.ndarray values representing the legal actions for all players in the current turn.
· slot: An integer indicating the player ID or slot for which the observation is being created.
· last_action: An integer representing the player's last action, used for teacher forcing.
· area: An integer specifying the area to create an action for.
· step_count: An integer indicating how many unit actions have been created so far.
· previous_area: An optional integer representing the area of the previous unit action.
· temperature: A float value representing the sampling temperature for unit actions.

**Code Description**: The step_observation_transform function takes in the specified parameters and returns a dictionary containing the step observation. It first determines the areas to sum over based on the provided area and player ID. If the area is set to BUILD_PHASE_AREA_FLAG, it calculates the build numbers and board state from the transformed initial observation and identifies the relevant areas for the player. Otherwise, it uses the province ID and area index to determine the legal actions for the specified area. The function then constructs a mask of legal actions and creates an ordered dictionary containing the step observation, which may include the areas, last action, legal actions mask, and temperature.

The step_observation_transform function is called by the observation_transform method, which transforms the observation into the format required by network policies. In this context, the step_observation_transform function is used to generate step observations for each player in the game, taking into account their individual areas and actions.

**Note**: The previous_area parameter is not used in the function and is set to None by default. Additionally, if no legal actions are found for the specified area, a ValueError is raised. It is also important to note that the network requires an area ordering to be specified, and attempting to use an invalid area flag will result in a NotImplementedError.

**Output Example**: The output of the step_observation_transform function may resemble the following dictionary:
{
    'areas': np.array([True, False, True, ...], dtype=bool),
    'last_action': np.array([-1], dtype=np.int32),
    'legal_actions_mask': np.array([False, True, False, ...], dtype=bool),
    'temperature': np.array([0.5], dtype=np.float32)
}
***
### FunctionDef observation_spec(self, num_players)
**observation_spec**: The function of observation_spec is to return a spec for the output of observation_transform.
**parameters**: The parameters of this Function.
· num_players: an integer representing the number of players, which is used to determine the shape of certain arrays in the returned spec
**Code Description**: This function generates a specification for the output of observation_transform by calling two other functions, initial_observation_spec and step_observation_spec, and combining their results with an additional sequence lengths array. The initial_observation_spec function returns a dictionary containing specifications for various components such as board state, last moves phase board state, actions since last moves phase, season, and build numbers. The step_observation_spec function returns a dictionary containing specifications for areas, last action, legal actions mask, and temperature. The observation_spec function then uses the tree.map_structure function to modify the shape of the arrays in the step observation spec by adding a num_players dimension. Finally, it returns a tuple containing the initial observation spec, the modified step observation spec, and an array representing sequence lengths.
**Note**: It's essential to note that the returned spec is dependent on the presence of specific attributes in the class instance, so the actual structure and content may vary. Additionally, understanding the context in which this function is called, specifically by zero_observation, can provide insight into its purpose and usage.
**Output Example**: A possible appearance of the code's return value could be a tuple containing three elements: an ordered dictionary with specs for 'board_state', 'last_moves_phase_board_state', 'actions_since_last_moves_phase', 'season', and 'build_numbers', where each spec has a specific shape and data type; an ordered dictionary with specs for 'areas', 'last_action', 'legal_actions_mask', and 'temperature', where each spec has a modified shape to include the num_players dimension; and an array representing sequence lengths with shape (num_players,) and dtype np.int32.
***
### FunctionDef zero_observation(self, num_players)
**zero_observation**: The function of zero_observation is to return a zero-filled observation by generating values based on the observation specification for a given number of players.
**parameters**: The parameters of this Function.
· num_players: an integer representing the number of players, which is used to determine the shape of certain arrays in the returned observation
**Code Description**: This function utilizes the tree.map_structure function to apply a lambda function to each element of the observation specification generated by the observation_spec method. The lambda function calls the generate_value method on each specification, effectively generating zero-filled values for the entire observation structure. The observation_spec method is responsible for defining the shape and structure of the observation, taking into account the number of players. By leveraging this method, zero_observation can produce a zero-filled observation that conforms to the expected format.
**Note**: It is essential to understand the context in which zero_observation is used, as its purpose is closely tied to the generation of observations in a specific environment. The observation_spec method plays a crucial role in defining the structure of the observation, and its output directly influences the result produced by zero_observation.
**Output Example**: A possible appearance of the code's return value could be a nested structure containing zero-filled arrays and values, where each element corresponds to a specific component of the observation specification, such as board state, last moves phase board state, actions since last moves phase, season, build numbers, areas, last action, legal actions mask, and temperature. The exact shape and content of the output will depend on the specifics of the observation_spec method and the number of players provided as input.
***
### FunctionDef observation_transform(self)
**Target Object Documentation**

## Overview

The Target Object is a fundamental entity designed to represent a specific goal or objective within a system or application. It serves as a focal point for various operations, allowing users to define and pursue distinct outcomes.

## Properties

The following properties are associated with the Target Object:

* **ID**: A unique identifier assigned to each Target Object instance, enabling efficient referencing and management.
* **Name**: A descriptive label assigned to the Target Object, providing context and clarity regarding its purpose or objective.
* **Description**: An optional text field allowing users to provide additional information about the Target Object, facilitating understanding and collaboration.

## Methods

The Target Object supports the following methods:

* **Create**: Initializes a new instance of the Target Object, assigning a unique ID and allowing users to specify its name and description.
* **Update**: Modifies existing properties of the Target Object, enabling users to refine or adjust its definition as needed.
* **Delete**: Removes the Target Object instance, eliminating it from the system or application.

## Relationships

The Target Object may establish relationships with other entities within the system or application, including:

* **Users**: Multiple users can be associated with a single Target Object, facilitating collaboration and shared ownership.
* **Tasks**: The Target Object can be linked to one or more tasks, representing specific actions or activities aimed at achieving the defined objective.

## Constraints

The following constraints apply to the Target Object:

* **Uniqueness**: Each Target Object instance must have a unique ID, ensuring distinct identification and management.
* **Data Validation**: Property values, such as name and description, are subject to validation rules to ensure data consistency and accuracy.

## Usage

To utilize the Target Object effectively, follow these guidelines:

1. Create a new Target Object instance by providing a unique name and optional description.
2. Associate relevant users and tasks with the Target Object to establish relationships and facilitate collaboration.
3. Update or modify the Target Object properties as needed to reflect changes in objectives or requirements.
4. Delete the Target Object instance when it is no longer required or relevant.

By adhering to these guidelines and understanding the properties, methods, and relationships of the Target Object, users can effectively leverage this entity to achieve their goals within the system or application.
***
### FunctionDef _topological_index(self)
**_topological_index**: The function of _topological_index is to determine the order in which areas should be processed based on the chosen topological indexing method.
**parameters**: None
**Code Description**: This function is used to return the order of areas according to the specified topological indexing method. It first checks the value of self._topological_indexing, which can be either TopologicalIndexing.NONE or TopologicalIndexing.MILA. If it is set to TopologicalIndexing.NONE, the function returns None, indicating that the order in the observation will be used. If it is set to TopologicalIndexing.MILA, the function returns mila_topological_index, which represents a specific topological indexing method. If self._topological_indexing has any other value, the function raises a RuntimeError, indicating an unexpected branch. The returned order of areas is then used in the observation transformation process, such as in the GeneralObservationTransformer class, to determine how areas will be ordered when producing orders from different areas.
**Note**: It is essential to understand the implications of each topological indexing method on the observation transformation process and choose the correct method according to the specific application or model requirements. Incorrectly choosing a topological indexing method may lead to unexpected behavior or incorrect results in downstream processing or modeling steps.
**Output Example**: The return value of this function can be either None, indicating no topological indexing, or a list representing the order of areas according to the chosen topological indexing method, such as mila_topological_index. For example, if self._topological_indexing is set to TopologicalIndexing.MILA, the function may return [area1, area2, area3], indicating that area1 should be processed first, followed by area2 and then area3.
***
