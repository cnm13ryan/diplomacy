## FunctionDef normalize_adjacency(adjacency)
**normalize_adjacency**: The function of normalize_adjacency is to compute the symmetric normalized Laplacian of an adjacency matrix.
**parameters**: The parameters of this Function.
· adjacency: a numpy array representing the map adjacency matrix without self-connections, which is used as input to calculate the symmetric normalized Laplacian.

**Code Description**: This function takes an adjacency matrix as input and returns its symmetric normalized Laplacian. It first adds the identity matrix to the input adjacency matrix using np.eye and then calculates the inverse of the square root of the sum of each row in the resulting matrix, which is used as a normalization factor. The function then applies this normalization factor to the adjacency matrix by multiplying it with the normalization factor from both sides, resulting in the symmetric normalized Laplacian. This process is crucial for graph representation and is utilized in GraphConvNets. In the context of the project, this function is called by the Network class's constructor to normalize the area and province adjacency matrices, which are then used in the BoardEncoder.

**Note**: The input adjacency matrix should not contain self-connections, and the output will be a symmetric normalized Laplacian matrix. It is also important to note that the calculation of the inverse of the square root of the sum of each row may result in division by zero if there are rows with zero sum, but this is handled implicitly by the np.power function.

**Output Example**: The output of this function will be a numpy array representing the symmetric normalized Laplacian matrix, where each element represents the normalized connection between two nodes in the graph. For instance, given an input adjacency matrix [[0, 1, 1], [1, 0, 1], [1, 1, 0]], the output might look like [[0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]].
## ClassDef EncoderCore
**EncoderCore**: The function of EncoderCore is to implement a Graph Network with non-shared weights across nodes, processing input organized by area and topology described by the symmetric normalized Laplacian of an adjacency matrix.

**attributes**: The attributes of this Class.
· `adjacency`: a symmetric normalized Laplacian of the adjacency matrix, describing the topology of the graph.
· `filter_size`: the output size of the per-node linear layer, defaulting to 32.
· `batch_norm_config`: a configuration dictionary for batch normalization, defaulting to None.
· `name`: a name for the module, defaulting to "encoder_core".
· `_bn`: an instance of `hk.BatchNorm` for batch normalization.

**Code Description**: The EncoderCore class is designed to perform one round of message passing in a graph network. It takes in a tensor representing the input nodes and returns a tensor representing the output nodes after message passing. The class constructor initializes the module with the given adjacency matrix, filter size, batch normalization configuration, and name. The `__call__` method performs the actual message passing operation, which involves computing messages from the input tensors, applying the adjacency matrix to the messages, concatenating the resulting tensors with the original messages, and then applying batch normalization and a ReLU activation function.

In the context of the project, EncoderCore is used by other classes such as BoardEncoder and RelationalOrderDecoder. In BoardEncoder, multiple instances of EncoderCore are created with different filter sizes to process shared and player-specific layers. Similarly, in RelationalOrderDecoder, multiple instances of EncoderCore are created to form a series of relational cores.

**Note**: When using the EncoderCore class, it is essential to provide a valid adjacency matrix that represents the topology of the graph. Additionally, the filter size and batch normalization configuration should be carefully chosen based on the specific requirements of the application.

**Output Example**: The output of the EncoderCore class will be a tensor with shape `[B, NUM_AREAS, 2 * self._filter_size]`, where `B` is the batch size, `NUM_AREAS` is the number of areas in the graph, and `self._filter_size` is the output size of the per-node linear layer. For instance, if the input tensor has shape `[32, 10, 64]` and the filter size is 32, the output tensor will have shape `[32, 10, 64]`.
### FunctionDef __init__(self, adjacency)
**__init__**: The function of __init__ is to initialize the EncoderCore module with the given parameters.
**parameters**: The parameters of this Function.
· adjacency: a symmetric normalized Laplacian of the adjacency matrix, represented as a jnp.ndarray with shape [NUM_AREAS, NUM_AREAS].
· filter_size: the output size of the per-node linear layer, defaults to 32 if not provided.
· batch_norm_config: an optional dictionary containing configuration for hk.BatchNorm, defaults to None.
· name: a string representing the name of the module, defaults to "encoder_core".
**Code Description**: The description of this Function. 
The __init__ function is responsible for setting up the EncoderCore module by storing the provided adjacency matrix, filter size, and batch normalization configuration. It first calls the parent class's constructor with the provided name. Then, it stores the adjacency matrix and filter size as instance variables. If a batch normalization configuration is provided, it updates the default configuration with the given values and creates an hk.BatchNorm instance with the updated configuration. The default batch normalization configuration includes a decay rate of 0.9, an epsilon value of 1e-5, and flags to create scale and offset.
**Note**: Points to note about the use of the code. 
When using this function, it is essential to provide a valid adjacency matrix as the first argument. The filter size and batch normalization configuration are optional but can significantly impact the behavior of the EncoderCore module. Additionally, the name parameter can be used to identify the module in the network architecture.
***
### FunctionDef __call__(self, tensors)
**__call__**: The function of __call__ is to perform one round of message passing in a neural network.
**parameters**: The parameters of this Function.
· tensors: a JAX numpy array with shape [B, NUM_AREAS, REP_SIZE] representing the input tensors
· is_training: a boolean indicating whether the function is being called during training, defaulting to False
**Code Description**: This function implements a single round of message passing in a neural network. It first retrieves a parameter "w" with shape [REP_SIZE, self._filter_size] using the Haiku library's get_parameter function, initialized with a variance scaling initializer. The input tensors are then transformed by multiplying them with the retrieved parameter "w" using an einsum operation, resulting in messages. These messages are then propagated through the network by multiplying them with the adjacency matrix self._adjacency. The output is obtained by concatenating the propagated messages and the original messages along the last axis, followed by batch normalization using self._bn and a ReLU activation function.
**Note**: The function uses JAX's numpy library for array operations and Haiku library for parameter management. The batch normalization is performed conditionally based on the is_training parameter. It is essential to ensure that the input tensors have the correct shape and that the filter size is compatible with the input representation size.
**Output Example**: The output of this function will be a JAX numpy array with shape [B, NUM_AREAS, 2 * self._filter_size], where B is the batch size, NUM_AREAS is the number of areas, and self._filter_size is the filter size used in the message passing process. For instance, if B = 32, NUM_AREAS = 10, and self._filter_size = 128, the output shape would be [32, 10, 256].
***
## ClassDef BoardEncoder
**BoardEncoder**: The function of BoardEncoder is to encode board state into a representation that can be used by other components in the system.

**attributes**: The attributes of this Class.
· adjacency: a symmetric normalized Laplacian of the adjacency matrix representing the connections between areas on the board
· shared_filter_size: the filter size of each EncoderCore for shared layers
· player_filter_size: the filter size of each EncoderCore for player-specific layers
· num_shared_cores: the number of shared layers, or rounds of message passing
· num_player_cores: the number of player-specific layers, or rounds of message passing
· num_players: the number of players in the game
· num_seasons: the number of seasons in the game
· player_embedding_size: the size of the player embedding
· season_embedding_size: the size of the season embedding
· min_init_embedding: the minimum value for the random uniform initializer for player and season embeddings
· max_init_embedding: the maximum value for the random uniform initializer for player and season embeddings
· batch_norm_config: a configuration dictionary for batch normalization
· name: a name for this module

**Code Description**: The BoardEncoder class is designed to construct a representation of the board state, taking into account the season, player, and number of builds. It first creates a shared representation that does not depend on the specific player, and then includes player-specific information in later layers. The class uses EncoderCore modules to perform message passing between areas on the board. The BoardEncoder is used by the Network class to encode the board state, which is then used as input to other components in the system. In particular, the Network class creates two instances of the BoardEncoder: one for the current board state and one for the last moves.

The BoardEncoder class has several key components. The `_season_embedding` and `_player_embedding` attributes are embeddings that map season and player indices to dense vectors. The `make_encoder` function is a partial application of the EncoderCore constructor, which creates an EncoderCore module with the given adjacency matrix and batch normalization configuration. The `_shared_encode`, `_shared_core`, `_player_encode`, and `_player_core` attributes are EncoderCore modules that perform message passing between areas on the board.

The `__call__` method of the BoardEncoder class takes in several inputs, including the state representation, season, build numbers, and a flag indicating whether the model is being trained. It first constructs a season context by embedding the season index and tiling it to match the shape of the state representation. It then concatenates the state representation, season context, and build numbers along the last axis. The method applies several rounds of message passing using the shared EncoderCore modules, followed by several rounds of message passing using the player-specific EncoderCore modules.

**Note**: When using the BoardEncoder class, it is important to ensure that the input shapes are correct and that the batch normalization configuration is properly set up. Additionally, the number of players and seasons should be consistent with the game being modeled.

**Output Example**: The output of the BoardEncoder class will be a tensor representing the encoded board state, with shape `[batch_size, num_areas, 2 * player_filter_size]`. For example, if `batch_size` is 32, `num_areas` is 56, and `player_filter_size` is 32, the output might look like:
```python
array([[[ 0.1,  0.2, ...,  0.9],
        [ 1.1,  1.2, ...,  1.9],
        ...,
        [55.1, 55.2, ..., 55.9]],

       [[ 0.11,  0.21, ...,  0.91],
        [ 1.11,  1.21, ...,  1.91],
        ...,
        [55.11, 55.21, ..., 55.91]],

       ...,

       [[ 0.31,  0.41, ...,  0.91],
        [ 1.31,  1.41, ...,  1.91],
        ...,
        [55.31, 55.41, ..., 55.91]]], dtype=float32)
```
### FunctionDef __init__(self, adjacency)
**__init__**: The function of __init__ is to initialize the BoardEncoder module with the given parameters.
**parameters**: The parameters of this Function.
· adjacency: a symmetric normalized Laplacian of the adjacency matrix, describing the topology of the graph.
· shared_filter_size: the filter size of each EncoderCore for shared layers, defaulting to 32.
· player_filter_size: the filter size of each EncoderCore for player-specific layers, defaulting to 32.
· num_shared_cores: the number of shared layers, or rounds of message passing, defaulting to 8.
· num_player_cores: the number of player-specific layers, or rounds of message passing, defaulting to 8.
· num_players: the number of players, defaulting to 7.
· num_seasons: the number of seasons, defaulting to utils.NUM_SEASONS.
· player_embedding_size: the size of player embedding, defaulting to 16.
· season_embedding_size: the size of season embedding, defaulting to 16.
· min_init_embedding: the minimum value for hk.initializers.RandomUniform for player and season embedding, defaulting to -1.0.
· max_init_embedding: the maximum value for hk.initializers.RandomUniform for player and season embedding, defaulting to 1.0.
· batch_norm_config: a configuration dictionary for batch normalization, defaulting to None.
· name: a name for this module, defaulting to "board_encoder".
**Code Description**: The __init__ function initializes the BoardEncoder module by setting up the necessary components, including the season and player embeddings, shared and player-specific EncoderCore instances, and batch normalization. It first calls the superclass's constructor with the given name. Then, it sets up the season and player embeddings using hk.Embed, with the specified embedding sizes and initialization ranges. The function also creates a partial function make_encoder, which is used to create EncoderCore instances with the given adjacency matrix and batch normalization configuration. The shared and player-specific EncoderCore instances are created using this partial function, with the specified filter sizes and numbers of cores. Finally, the function sets up the batch normalization module with the given configuration.
The BoardEncoder module relies on the EncoderCore class to perform message passing in the graph network. The EncoderCore class is responsible for processing input tensors organized by area and topology described by the symmetric normalized Laplacian of an adjacency matrix. By creating multiple instances of EncoderCore with different filter sizes, the BoardEncoder module can process shared and player-specific layers separately.
**Note**: When using the BoardEncoder module, it is essential to provide a valid adjacency matrix that represents the topology of the graph. Additionally, the filter sizes, batch normalization configuration, and embedding sizes should be carefully chosen based on the specific requirements of the application. The number of players, seasons, and cores should also be set according to the problem's needs.
***
### FunctionDef __call__(self, state_representation, season, build_numbers, is_training)
**__call__**: The function of __call__ is to encode board state into a numerical representation.
**parameters**: The parameters of this Function.
· state_representation: a jnp.ndarray representing the current state of the board, with shape [B, NUM_AREAS, REP_SIZE].
· season: a jnp.ndarray representing the current season, with shape [B, 1].
· build_numbers: a jnp.ndarray representing the current build numbers, with shape [B, 1].
· is_training: a boolean indicating whether this is during training, defaulting to False.

**Code Description**: The __call__ function first creates a season context by tiling the season embedding across all areas of the board. It then creates a build number context by tiling the build numbers across all areas of the board. These contexts are concatenated with the state representation to create an enhanced representation of the board state. This representation is then passed through a series of encoding layers, including shared and player-specific layers, to produce a final encoded representation of the board state. The function uses various techniques such as batch application and concatenation to efficiently process the input data.

**Note**: The use of jnp.ndarray as input parameters suggests that this function is designed to work with JAX numerical computing library. The function also relies on various instance variables, such as _season_embedding, _player_embedding, _shared_encode, _shared_core, _player_encode, and _player_core, which are not defined in this documentation. It is assumed that these variables are properly initialized and configured before calling the __call__ function.

**Output Example**: The output of the __call__ function will be a jnp.ndarray with shape [B, NUM_AREAS, 2 * self._player_filter_size], representing the encoded board state. For example, if B = 32, NUM_AREAS = 10, and self._player_filter_size = 128, the output would be a jnp.ndarray with shape [32, 10, 256].
***
## ClassDef RecurrentOrderNetworkInput
**RecurrentOrderNetworkInput**: The function of RecurrentOrderNetworkInput is to represent the input data structure for a recurrent neural network designed to process sequential order-related information.

**attributes**: The attributes of this Class.
· average_area_representation: a jnp.ndarray representing the averaged area representation
· legal_actions_mask: a jnp.ndarray indicating the mask for legal actions
· teacher_forcing: a jnp.ndarray representing the teacher forcing input
· previous_teacher_forcing_action: a jnp.ndarray representing the previous teacher forcing action
· temperature: a jnp.ndarray representing the temperature value

**Code Description**: The RecurrentOrderNetworkInput class is a NamedTuple that encapsulates the necessary input data for a recurrent neural network. This data structure is specifically designed to handle sequential order-related information, where each attribute plays a crucial role in the decision-making process of the network. The average_area_representation attribute provides a condensed representation of the area, while the legal_actions_mask attribute filters out illegal actions. The teacher_forcing and previous_teacher_forcing_action attributes are used to guide the network during training, and the temperature attribute controls the exploration-exploitation trade-off.

In the context of the project, the RecurrentOrderNetworkInput class is utilized by the RelationalOrderDecoder and Network classes. Specifically, in the RelationalOrderDecoder's __call__ method, an instance of RecurrentOrderNetworkInput is passed as an argument to provide the necessary input data for issuing orders based on board representation and previous decisions. Similarly, in the Network's step_inference method, a RecurrentOrderNetworkInput instance is created to compute logits for units that require orders.

The use of RecurrentOrderNetworkInput enables the recurrent neural network to effectively process sequential order-related information, making it an essential component of the project's architecture.

**Note**: When using the RecurrentOrderNetworkInput class, it is essential to ensure that all attributes are properly initialized and passed to the relevant methods. Additionally, the specific requirements for each attribute, such as shape and data type, must be adhered to in order to maintain the integrity of the network's functionality.
## FunctionDef previous_action_from_teacher_or_sample(teacher_forcing, previous_teacher_forcing_action, previous_sampled_action_index)
**previous_action_from_teacher_or_sample**: The function of previous_action_from_teacher_or_sample is to determine the previous action based on teacher forcing or sampled actions.
**parameters**: The parameters of this Function.
· teacher_forcing: a jnp.ndarray indicating whether teacher forcing is being used
· previous_teacher_forcing_action: a jnp.ndarray representing the previous action when teacher forcing is used
· previous_sampled_action_index: a jnp.ndarray representing the index of the previously sampled action
**Code Description**: This function uses the jnp.where function to conditionally return either the previous teacher forcing action or the action corresponding to the previous sampled action index. The action corresponding to the previous sampled action index is obtained by using the shrink_actions function from the action_utils module and then indexing into the resulting array with the previous sampled action index. This function is called by the RelationalOrderDecoder's __call__ method, where it is used to determine the previous action based on whether teacher forcing is being used or not. The result of this function is then used to update the blocked provinces and construct a representation of the previous order.
**Note**: The use of teacher forcing allows for the model to be trained using supervised learning, where the correct actions are provided as input. When teacher forcing is not used, the model samples actions from its output distribution. This function provides a way to handle both cases and ensure that the model can learn from its past actions.
**Output Example**: The output of this function will be a jnp.ndarray representing the previous action, which can take on values corresponding to the possible actions in the environment. For example, if there are 10 possible actions, the output might be an array with shape [batch_size] containing integers between 0 and 9, where each integer represents one of the possible actions.
## FunctionDef one_hot_provinces_for_all_actions
**one_hot_provinces_for_all_actions**: The function of one_hot_provinces_for_all_actions is to generate a one-hot representation of provinces for all possible actions.
**parameters**: There are no parameters for this Function.
**Code Description**: This function utilizes the jax.nn.one_hot function to create a one-hot encoding of provinces for all possible actions. It first converts the ordered province indices for all possible actions into a JAX numpy array using jnp.asarray, and then applies the one-hot encoding with the number of provinces defined in utils.NUM_PROVINCES. The resulting one-hot representation is used by other functions in the project, such as blocked_provinces_and_actions and RelationalOrderDecoder, to determine which provinces are associated with each action and to calculate the legality of actions based on previous decisions.
**Note**: It's essential to note that this function relies on external constants and functions, including action_utils.ordered_province, utils.NUM_PROVINCES, and jax.nn.one_hot, which must be defined and accessible for this function to work correctly. Additionally, the output of this function is used in various calculations throughout the project, so any changes to its implementation may have far-reaching effects on the overall functionality.
**Output Example**: The return value of one_hot_provinces_for_all_actions will be a 2D array where each row represents an action and each column represents a province, with a value of 1 indicating that the province is associated with the action and 0 otherwise. For example, if there are 10 provinces and 20 possible actions, the output might look like a 20x10 array with binary values indicating which provinces are relevant for each action.
## FunctionDef blocked_provinces_and_actions(previous_action, previous_blocked_provinces)
**blocked_provinces_and_actions**: The function of blocked_provinces_and_actions is to calculate which provinces and actions are illegal based on previous decisions.
**parameters**: The parameters of this Function.
· previous_action: a jnp.ndarray representing the previous action taken, used to determine the updated blocked provinces and actions.
· previous_blocked_provinces: a jnp.ndarray representing the previously blocked provinces, used as input to calculate the updated blocked provinces.
**Code Description**: This function takes into account the previous action and blocked provinces to compute the updated blocked provinces by applying the maximum operation between the previous blocked provinces and the one-hot encoded provinces obtained from the previous action. It then calculates the blocked actions by multiplying the one-hot representation of provinces for all possible actions with the updated blocked provinces, and filters out waive actions using the is_waive function. The result is a tuple containing the updated blocked provinces and blocked actions. This function is used in conjunction with other functions such as RelationalOrderDecoder to determine the legality of actions based on previous decisions.
**Note**: It's essential to note that this function relies on external constants and functions, including action_utils and utils, which must be defined and accessible for this function to work correctly. Additionally, the output of this function is used in various calculations throughout the project, so any changes to its implementation may have far-reaching effects on the overall functionality.
**Output Example**: The return value of blocked_provinces_and_actions will be a tuple containing two arrays: updated_blocked_provinces and blocked_actions. For instance, if there are 10 provinces and 20 possible actions, the output might look like a tuple where the first element is a 1D array representing the updated blocked provinces with binary values indicating which provinces are blocked, and the second element is a 1D array representing the blocked actions with binary values indicating which actions are illegal.
## FunctionDef sample_from_logits(logits, legal_action_mask, temperature)
**sample_from_logits**: The function of sample_from_logits is to sample an action from a given set of logits while respecting a legal actions mask and considering a temperature parameter.
**parameters**: The parameters of this Function.
· logits: a JAX numpy array representing the logits of the actions
· legal_action_mask: a JAX numpy array indicating which actions are legal
· temperature: a JAX numpy array controlling the level of randomness in the sampling process
**Code Description**: This function takes in the logits, legal action mask, and temperature as inputs. It first constructs deterministic logits by setting all non-maximum values to negative infinity, and stochastic logits by scaling the original logits with the temperature. The function then combines these two sets of logits based on the temperature value: if the temperature is zero, it uses the deterministic logits; otherwise, it uses the stochastic logits. Finally, the function samples an action from the resulting logits using JAX's categorical sampling function.
The sample_from_logits function is used by the RelationalOrderDecoder class in the network module to issue orders based on board representation and previous decisions. Specifically, it is called in the __call__ method of RelationalOrderDecoder to sample an action index after constructing order logits conditional on province representation and previous orders.
**Note**: The temperature parameter controls the level of randomness in the sampling process: a temperature of zero results in deterministic sampling, while a non-zero temperature introduces randomness. Additionally, the legal action mask ensures that only valid actions are considered during sampling.
**Output Example**: The output of sample_from_logits is a JAX numpy array representing the sampled action index, which can be used to update the state and make subsequent decisions. For instance, if the input logits have shape [batch_size, max_action_index], the output might look like an array of shape [batch_size] containing integer values between 0 and max_action_index-1, each representing a sampled action.
## ClassDef RelationalOrderDecoderState
**RelationalOrderDecoderState**: The function of RelationalOrderDecoderState is to represent the state of a relational order decoder, encapsulating previous orders, blocked provinces, and sampled action indices.

**attributes**: The attributes of this Class.
· prev_orders: a jnp.ndarray representing the previous orders, with shape [B*PLAYERS, NUM_PROVINCES, 2 * self._filter_size].
· blocked_provinces: a jnp.ndarray indicating the blocked provinces, with shape [B*PLAYERS, NUM_PROVINCES].
· sampled_action_index: a jnp.ndarray representing the sampled action index, with shape [B*PLAYER].

**Code Description**: The RelationalOrderDecoderState class is a NamedTuple that stores the state of a relational order decoder. It contains three attributes: prev_orders, blocked_provinces, and sampled_action_index. This class is used in conjunction with the RelationalOrderDecoder class, where it serves as both an input and output to the __call__ method. The initial_state method of the RelationalOrderDecoder class returns an instance of RelationalOrderDecoderState, which is then passed to the __call__ method along with other inputs. The __call__ method updates the state based on the previous orders, blocked provinces, and sampled action indices, and returns a new instance of RelationalOrderDecoderState. This updated state can be used for subsequent calls to the __call__ method.

The attributes of RelationalOrderDecoderState are critical in maintaining the context of the relational order decoder. The prev_orders attribute keeps track of the previous orders, which is essential in constructing the representation of the province currently under consideration. The blocked_provinces attribute indicates which provinces are blocked, and this information is used to eliminate illegal actions. The sampled_action_index attribute represents the sampled action index, which is used to update the state.

**Note**: When using RelationalOrderDecoderState, it is essential to ensure that the attributes are correctly initialized and updated. The initial_state method of the RelationalOrderDecoder class provides a convenient way to initialize an instance of RelationalOrderDecoderState with zeros. Additionally, the __call__ method of the RelationalOrderDecoder class updates the state based on the previous orders, blocked provinces, and sampled action indices, ensuring that the state remains consistent throughout the decoding process.
## ClassDef RelationalOrderDecoder
**RelationalOrderDecoder**: The function of RelationalOrderDecoder is to output order logits for a unit based on the current board representation and the orders selected for other units so far.

**attributes**: The attributes of this Class.
· `adjacency`: a symmetric normalized Laplacian of the per-province adjacency matrix, with shape [NUM_PROVINCES, NUM_PROVINCES].
· `filter_size`: the filter size for relational cores, default value is 32.
· `num_cores`: the number of relational cores, default value is 4.
· `batch_norm_config`: a configuration dictionary for batch normalization, default value is None.
· `name`: the name of the module, default value is "relational_order_decoder".

**Code Description**: The RelationalOrderDecoder class is designed to issue orders based on the current board representation and previous decisions. It uses a relational core to process the input information and generate order logits. The class has several key methods: `__init__` for initialization, `_scatter_to_province` and `_gather_province` for scattering and gathering province information, `_relational_core` for applying the relational core, `__call__` for issuing an order, and `initial_state` for generating the initial state.

The `__init__` method initializes the RelationalOrderDecoder object with the given adjacency matrix, filter size, number of cores, batch normalization configuration, and name. It also sets up the relational cores and batch normalization layer.

The `_scatter_to_province` method scatters a vector to its province location in the inputs, while the `_gather_province` method gathers specific province information from the inputs.

The `_relational_core` method applies the relational core to the current province and previous decisions. It concatenates the previous orders and board representation, encodes the input using the `EncoderCore`, and then applies multiple relational cores to the encoded representation.

The `__call__` method issues an order based on the board representation and previous decisions. It takes in the recurrent order network input, previous state, and a flag indicating whether it is during training. It constructs the representation of the previous order, updates the blocked provinces, and then applies the relational core to generate the order logits.

The `initial_state` method generates the initial state of the RelationalOrderDecoder object, which includes the previous orders, blocked provinces, and sampled action index.

**Note**: The RelationalOrderDecoder class assumes that the input data is properly formatted and that the adjacency matrix is symmetric and normalized. It also assumes that the batch normalization configuration is valid. Users should ensure that these assumptions are met when using this class.

**Output Example**: The output of the `__call__` method will be a tuple containing the order logits with shape [B*PLAYERS, MAX_ACTION_INDEX] and the updated RelationalOrderDecoderState object, which includes the previous orders, blocked provinces, and sampled action index. For example:
```python
order_logits = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
updated_state = RelationalOrderDecoderState(
    prev_orders=jnp.array([[[0.7, 0.8], [0.9, 1.0]], [[1.1, 1.2], [1.3, 1.4]]]),
    blocked_provinces=jnp.array([[0, 1], [1, 0]]),
    sampled_action_index=jnp.array([1, 2])
)
```
### FunctionDef __init__(self, adjacency)
**__init__**: The function of __init__ is to initialize the RelationalOrderDecoder module with the given parameters.
**parameters**: The parameters of this Function.
· adjacency: a symmetric normalized Laplacian of the per-province adjacency matrix, describing the topology of the graph.
· filter_size: the filter size for relational cores, defaulting to 32.
· num_cores: the number of relational cores, defaulting to 4.
· batch_norm_config: a configuration dictionary for batch normalization, defaulting to None.
· name: a name for the module, defaulting to "relational_order_decoder".
**Code Description**: The __init__ function initializes the RelationalOrderDecoder module by calling the constructor of its parent class with the provided name. It then sets the filter size and creates an instance of EncoderCore with the given adjacency matrix, filter size, and batch normalization configuration. Additionally, it creates a list of relational cores by appending multiple instances of EncoderCore to the list. The function also calculates the projection size based on the filter size and initializes an instance of hk.BatchNorm with the provided batch normalization configuration.
The RelationalOrderDecoder module utilizes the EncoderCore class to perform graph network operations. The EncoderCore class is designed to process input tensors organized by area and topology described by the symmetric normalized Laplacian of an adjacency matrix. In the context of the RelationalOrderDecoder module, multiple instances of EncoderCore are created to form a series of relational cores, which enables the module to perform complex graph network operations.
**Note**: When using the RelationalOrderDecoder module, it is essential to provide a valid adjacency matrix that represents the topology of the graph. Additionally, the filter size, number of relational cores, and batch normalization configuration should be carefully chosen based on the specific requirements of the application. The batch normalization configuration dictionary should contain valid parameters for hk.BatchNorm, such as decay rate, epsilon, create scale, and create offset.
***
### FunctionDef _scatter_to_province(self, vector, scatter)
**_scatter_to_province**: The function of _scatter_to_province is to scatter a given vector to its corresponding province location based on a provided one-hot encoding scatter array.
**parameters**: The parameters of this Function.
· vector: a JAX numpy array with shape [B*PLAYERS, REP_SIZE] representing the input vector to be scattered.
· scatter: a JAX numpy array with shape [B*PLAYER, NUM_PROVINCES] representing the one-hot encoding scatter array that determines the province location for each element in the input vector.

**Code Description**: The _scatter_to_province function takes two parameters, vector and scatter, and returns a new array where the input vector has been scattered to its corresponding province location based on the provided one-hot encoding scatter array. This is achieved by performing an element-wise multiplication between the input vector and the scatter array, with the scatter array broadcasted to match the shape of the input vector. The resulting array has a shape of [B*PLAYERS, NUM_AREAS, REP_SIZE], where each element in the input vector has been added to its corresponding province location.

In the context of the RelationalOrderDecoder class, this function is used to scatter the average area representation and previous order representation to their respective province locations. The scattered representations are then used as inputs to the relational core and gather province functions to construct order logits conditional on province representation and previous orders.

The _scatter_to_province function is called by the __call__ method of the RelationalOrderDecoder class, which issues an order based on the board representation and previous decisions. Specifically, it is used to scatter the average area representation and previous order representation to their respective province locations, which are then used to construct the order logits.

**Note**: The _scatter_to_province function assumes that the input vector and scatter array have compatible shapes for broadcasting. Additionally, the function does not perform any error checking on the inputs, so it is the responsibility of the caller to ensure that the inputs are valid.

**Output Example**: The output of the _scatter_to_province function will be a JAX numpy array with shape [B*PLAYERS, NUM_AREAS, REP_SIZE], where each element in the input vector has been added to its corresponding province location. For example, if the input vector has a shape of [10, 128] and the scatter array has a shape of [10, 20], the output array will have a shape of [10, 20, 128].
***
### FunctionDef _gather_province(self, inputs, gather)
**_gather_province**: The function of _gather_province is to gather specific province location from inputs based on a given one-hot encoding.
**parameters**: The parameters of this Function.
· inputs: a 3D array with shape [B*PLAYERS, NUM_PROVINCES, REP_SIZE] representing the input data
· gather: a 2D array with shape [B*PLAYERS, NUM_PROVINCES] representing the one-hot encoding for gathering specific provinces
**Code Description**: The _gather_province function takes in two parameters, inputs and gather. It uses the gather parameter to select specific province locations from the inputs array. This is achieved by multiplying the inputs array with the gather array, which has a shape of [B*PLAYERS, NUM_PROVINCES], and then summing along the axis representing the provinces (axis=1). The result is a 2D array with shape [B*PLAYERS, REP_SIZE] where each row represents the gathered province location for each player. This function is used in the RelationalOrderDecoder to construct order logits conditional on province representation and previous orders.
**Note**: The _gather_province function assumes that the inputs array has a shape of [B*PLAYERS, NUM_PROVINCES, REP_SIZE] and the gather array has a shape of [B*PLAYERS, NUM_PROVINCES]. It also assumes that the gather array is a one-hot encoding where each row represents a specific province. The function is used in conjunction with other functions in the RelationalOrderDecoder to generate order logits.
**Output Example**: The output of the _gather_province function could be a 2D array like [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], where each row represents the gathered province location for each player, with the values representing the REP_SIZE dimension.
***
### FunctionDef _relational_core(self, previous_orders, board_representation, is_training)
**_relational_core**: The function of _relational_core is to apply relational core to current province and previous decisions.
**parameters**: The parameters of this Function.
· previous_orders: a jnp.ndarray representing previous orders
· board_representation: the representation of the game board
· is_training: a boolean indicating whether the model is in training mode, defaulting to False
**Code Description**: This function takes in previous orders and the current board representation, concatenates them, and then applies an encoding operation. The encoded representation is then passed through multiple relational cores, with each core's output being added to the previous one. Finally, the result is normalized using a batch normalization layer. The _relational_core function is called by the __call__ method of the RelationalOrderDecoder class, where it plays a crucial role in constructing order logits conditional on province representation and previous orders.
**Note**: The _relational_core function assumes that the input previous_orders and board_representation are properly formatted and have the correct shapes. Additionally, the function's behavior may vary depending on whether the model is in training mode or not.
**Output Example**: The output of this function will be a jnp.ndarray representing the relational core applied to the current province and previous decisions, which can then be used for further processing, such as constructing order logits. For instance, the output might have a shape similar to [batch_size, num_provinces, representation_size], although the actual shape may vary depending on the specific implementation and input shapes.
***
### FunctionDef __call__(self, inputs, prev_state)
**Target Object Documentation**

### Overview

The Target Object is a fundamental entity designed to represent a specific goal or objective within a system or process. It serves as a focal point for various operations, allowing for precise control and manipulation.

### Properties

The following properties are associated with the Target Object:

* **Identifier**: A unique identifier assigned to the Target Object, enabling distinction from other objects.
* **Description**: A concise description of the Target Object's purpose or function.
* **Status**: The current state of the Target Object, which can be one of the following:
	+ Active: The Target Object is currently in use or being processed.
	+ Inactive: The Target Object is not currently in use or being processed.
	+ Pending: The Target Object is awaiting processing or activation.

### Methods

The Target Object supports the following methods:

* **Create**: Initializes a new instance of the Target Object with specified properties.
* **Update**: Modifies the properties of an existing Target Object.
* **Delete**: Removes the Target Object from the system or process.
* **Activate**: Sets the status of the Target Object to Active.
* **Deactivate**: Sets the status of the Target Object to Inactive.

### Relationships

The Target Object can be related to other objects within the system or process, including:

* **Parent Object**: The object that contains or owns the Target Object.
* **Child Objects**: Objects that are contained or owned by the Target Object.
* **Peer Objects**: Objects that have a similar relationship or function as the Target Object.

### Constraints

The following constraints apply to the Target Object:

* Each Target Object must have a unique Identifier.
* The Description property is optional but recommended for clarity and understanding.
* The Status property can only be one of the specified values (Active, Inactive, or Pending).

By adhering to these properties, methods, relationships, and constraints, the Target Object provides a structured and consistent representation of goals or objectives within a system or process.
***
### FunctionDef initial_state(self, batch_size, dtype)
**initial_state**: The function of initial_state is to return an instance of RelationalOrderDecoderState, representing the initial state of a relational order decoder.

**parameters**: The parameters of this Function.
· batch_size: an integer specifying the size of the batch
· dtype: the data type of the returned state, defaulting to jnp.float32

**Code Description**: This function generates an instance of RelationalOrderDecoderState with its attributes initialized to zero. The prev_orders attribute is a jnp.ndarray with shape (batch_size, utils.NUM_PROVINCES, 2 * self._filter_size), the blocked_provinces attribute is a jnp.ndarray with shape (batch_size, utils.NUM_PROVINCES), and the sampled_action_index attribute is a jnp.ndarray with shape (batch_size,) and data type jnp.int32. The function utilizes the RelationalOrderDecoderState class to create this initial state, which can then be used as input for subsequent operations.

**Note**: When using the initial_state function, it is essential to ensure that the batch_size parameter is correctly specified, as it determines the size of the returned state. Additionally, the data type of the returned state can be customized by passing a different dtype parameter, but it defaults to jnp.float32 if not provided.

**Output Example**: The return value of this function would be an instance of RelationalOrderDecoderState, with its attributes initialized as follows: 
prev_orders = jnp.zeros((batch_size, utils.NUM_PROVINCES, 2 * self._filter_size)), 
blocked_provinces = jnp.zeros((batch_size, utils.NUM_PROVINCES)), 
sampled_action_index = jnp.zeros((batch_size,), dtype=jnp.int32).
***
## FunctionDef ordered_provinces(actions)
**ordered_provinces**: The function of ordered_provinces is to extract the province information from a given set of actions.
**parameters**: The parameters of this Function.
· actions: a jnp.ndarray representing the actions from which the province information will be extracted
**Code Description**: This function takes in an array of actions and applies bitwise operations to extract the province information. It first right-shifts the actions by a specified number of bits, defined by action_utils.ACTION_ORDERED_PROVINCE_START, and then performs a bitwise AND operation with a mask created from action_utils.ACTION_PROVINCE_BITS. The result is the extracted province information.
The ordered_provinces function is used in the RelationalOrderDecoder class, specifically in the __call__ method, to process previous actions and update the state of the decoder. In this context, the function helps to determine the provinces that are relevant to the current decision-making process.
**Note**: The correct usage of this function relies on the proper definition of action_utils.ACTION_ORDERED_PROVINCE_START and action_utils.ACTION_PROVINCE_BITS, which are assumed to be defined elsewhere in the codebase. Additionally, the input actions should be a jnp.ndarray with the correct shape and data type.
**Output Example**: The output of the ordered_provinces function will be a jnp.ndarray representing the extracted province information, where each element corresponds to a specific action in the input array. For instance, if the input actions are [1024, 2048, 4096], the output might be [1, 2, 3], indicating that the corresponding provinces are 1, 2, and 3, respectively.
## FunctionDef is_waive(actions)
**is_waive**: The function of is_waive is to determine whether a given set of actions represents a waive action.
**parameters**: The parameters of this Function.
· actions: a jnp.ndarray representing the set of actions to be evaluated
**Code Description**: This function takes in an array of actions and applies bitwise operations to extract specific information. It first shifts the bits of each action to the right by ACTION_ORDER_START places, then performs a bitwise AND operation with a mask created from ACTION_ORDER_BITS. The result is compared to WAIVE using jnp.equal, which returns a boolean value indicating whether the actions represent a waive. In the context of the project, this function is used by blocked_provinces_and_actions to filter out waive actions when determining which provinces and actions are illegal.
**Note**: The use of bitwise operations and specific constants such as ACTION_ORDER_START, ACTION_ORDER_BITS, and WAIVE suggests that the actions are encoded in a binary format, where certain bits correspond to specific properties or flags. It is essential to understand the meaning of these constants and their relationship to the action encoding to correctly interpret the results of this function.
**Output Example**: The output of is_waive could be a boolean array, where each element corresponds to an action in the input array, indicating whether that action represents a waive. For instance, [True, False, True] would indicate that the first and third actions are waive actions, while the second is not.
## FunctionDef loss_from_logits(logits, actions, discounts)
**loss_from_logits**: The function of loss_from_logits is to calculate the cross-entropy loss or entropy based on the given logits, actions, and discounts.
**parameters**: The parameters of this Function.
· logits: The input logits to be used for calculating the loss.
· actions: The actions taken, which can be None if entropy is to be calculated instead of cross-entropy loss.
· discounts: The discount factors applied to the loss calculation.
**Code Description**: This function calculates the loss based on the provided logits and actions. If actions are not None, it computes the cross-entropy loss by taking the log softmax of the logits and selecting the corresponding action indices. The loss is then filtered to only consider actual actions (i.e., actions greater than 0) and summed along the last axis. If actions are None, the function calculates the entropy of the logits instead. In both cases, the calculated loss is then multiplied by the discounts and finally averaged to obtain the mean loss. This function is used in the network's policy update process, specifically in the loss_info method of the Network class, where it is called to calculate the policy loss and policy entropy.
**Note**: The function expects the input logits to have a shape compatible with the actions and discounts tensors. Additionally, the actions tensor should have values greater than 0 for actual actions, and the discounts tensor should contain the discount factors applied to each action.
**Output Example**: A scalar value representing the calculated mean loss, which can be used as part of the overall loss calculation in the network's policy update process. For instance, if the input logits are [0.5, 0.3, 0.2] and actions are [1, 0, 0], the function might return a value around 0.4, indicating the mean cross-entropy loss for the given inputs.
## FunctionDef ordered_provinces_one_hot(actions, dtype)
**ordered_provinces_one_hot**: The function of ordered_provinces_one_hot is to generate one-hot encoded provinces based on given actions while considering specific conditions.
**parameters**: The parameters of this Function.
· actions: This parameter represents the input actions that will be used to determine the one-hot encoded provinces.
· dtype: This parameter specifies the data type of the output, with a default value of jnp.float32.

**Code Description**: The ordered_provinces_one_hot function utilizes the jax.nn.one_hot function to generate one-hot encoded provinces. It first computes the ordered province indices using the action_utils.ordered_province function and then creates one-hot encodings for these indices. The resulting one-hot encoded provinces are then multiplied by a mask that filters out actions with values less than or equal to 0 and waive actions, as determined by the action_utils.is_waive function. This ensures that only valid actions contribute to the final output.

The ordered_provinces_one_hot function is used in conjunction with other functions within the project, such as blocked_provinces_and_actions and reorder_actions. In the context of blocked_provinces_and_actions, the ordered_provinces_one_hot function helps to calculate which provinces are blocked by past decisions, ultimately influencing the determination of illegal actions. Meanwhile, in the reorder_actions function, ordered_provinces_one_hot plays a crucial role in reordering actions according to area ordering, facilitating the computation of ordered actions.

**Note**: It is essential to note that the output of this function depends on the input actions and the specific conditions applied during the computation process. The data type of the output can be customized using the dtype parameter, which may impact subsequent operations or calculations involving the resulting one-hot encoded provinces.

**Output Example**: A possible appearance of the code's return value could be a 3D array where each element represents a one-hot encoded province, with values being either 0 or 1, depending on whether the corresponding action is valid and not waived. For instance, if there are 10 provinces and 5 actions, the output might resemble a 3D array of shape (5, 10, 1), where each action is associated with a one-hot encoded province vector.
## FunctionDef reorder_actions(actions, areas, season)
**reorder_actions**: The function of reorder_actions is to reorder actions based on area ordering.
**parameters**: The parameters of this Function.
· actions: This parameter represents the input actions that will be used to determine the reordered actions.
· areas: This parameter represents the areas that are used as a reference for reordering the actions.
· season: This parameter represents the current season, which is used to determine whether to skip reordering.

**Code Description**: The reorder_actions function takes in three parameters: actions, areas, and season. It first computes one-hot encoded provinces based on the given areas using the jax.nn.one_hot function. Then, it calculates the action provinces by calling the ordered_provinces_one_hot function with the input actions. The function then computes the ordered actions by summing the product of the actions and action provinces, and the provinces. Additionally, it calculates the number of actions found by summing the product of the action provinces and provinces. If an action is missing, represented by -1, it adds the number of actions found minus 1 to the ordered actions. The function also checks if the current season is a build season, and if so, it skips reordering the actions. Finally, the function returns the reordered actions.

The reorder_actions function is called by the loss_info method in the Network class, where it is used to reorder the actions to match with the legal actions ordering before calculating the policy loss and value loss. The reordered actions are then used as input to the loss_from_logits function to calculate the policy loss.

**Note**: It is essential to note that the output of this function depends on the input actions, areas, and season. The function assumes that the input actions and areas are valid and correctly formatted. If the input actions or areas are invalid, the function may produce incorrect results.

**Output Example**: A possible appearance of the code's return value could be a 3D array where each element represents a reordered action, with values being either the original action value or -1 if the action is missing. For instance, if there are 10 actions and 5 areas, the output might resemble a 3D array of shape (5, 10), where each area is associated with a list of reordered actions.
## ClassDef Network
**Target Object Documentation**

## Overview

The Target Object is a fundamental entity designed to represent a specific goal or objective within a system or process. It serves as a focal point for various operations, allowing for precise control and manipulation.

## Properties

The following properties are associated with the Target Object:

* **Identifier**: A unique identifier assigned to the Target Object, enabling distinction from other objects.
* **Name**: A descriptive label assigned to the Target Object, providing context and clarity.
* **Description**: A detailed explanation of the Target Object's purpose and functionality.

## Methods

The Target Object supports the following methods:

* **Initialize**: Initializes the Target Object with default values or specified parameters.
* **Update**: Modifies the properties of the Target Object, allowing for dynamic changes.
* **Retrieve**: Retrieves the current state or properties of the Target Object.

## Relationships

The Target Object can establish relationships with other objects within the system, including:

* **Parent-Child Relationship**: The Target Object can serve as a parent or child to other objects, enabling hierarchical structures.
* **Peer-to-Peer Relationship**: The Target Object can interact with other objects on an equal level, facilitating collaborative operations.

## Constraints

The following constraints apply to the Target Object:

* **Uniqueness**: Each Target Object must have a unique identifier to prevent duplication.
* **Data Validation**: Properties and methods of the Target Object are subject to validation rules to ensure data integrity.

## Usage

The Target Object is designed to be used in various contexts, including but not limited to:

* **Goal-Oriented Systems**: The Target Object represents a specific objective or goal, guiding system behavior.
* **Process Management**: The Target Object serves as a focal point for process control and manipulation.

By following the guidelines outlined in this documentation, developers can effectively utilize the Target Object to achieve precise control and manipulation within their systems.
### FunctionDef initial_inference_params_and_state(cls, constructor_kwargs, rng, num_players)
**initial_inference_params_and_state**: The function of initial_inference_params_and_state is to initialize the parameters and state of a network for inference purposes.
**parameters**: The parameters of this Function.
· cls: This parameter represents the class itself and is used to access class methods, specifically to create an instance of the network.
· constructor_kwargs: These are keyword arguments that would be used to construct an instance of the network.
· rng: This parameter represents a random number generator, which is used to introduce randomness in the initialization process.
· num_players: This parameter specifies the number of players, which is used to determine the shape of the observations.

**Code Description**: The initial_inference_params_and_state function works by first defining an inner function _inference, which takes observations as input and returns the result of the network's inference method. The _inference function is then transformed into a function with state using the hk.transform_with_state method. This transformed function is then initialized with the provided random number generator and a zero observation, which is obtained by calling the get_observation_transformer method to get an instance of GeneralObservationTransformer, and then using its zero_observation method. The initialization process returns the parameters and state of the network.

The get_observation_transformer method is used within initial_inference_params_and_state to obtain an observation transformer, which is necessary for generating a zero observation. This observation transformer is created based on the provided constructor_kwargs, but these arguments do not affect the behavior of the get_observation_transformer function itself.

**Note**: It is essential to be aware that the initialization process relies on the randomness introduced by the rng parameter, and the shape of the observations depends on the num_players parameter. Additionally, the constructor_kwargs parameter passed to initial_inference_params_and_state affects the creation of the network instance within the _inference function.

**Output Example**: A possible appearance of the code's return value could be a tuple containing the parameters and state of the network, such as (params, net_state), where params represents the initialized parameters of the network and net_state represents its initial state.
#### FunctionDef _inference(observations)
**_inference**: The function of _inference is to perform inference on a given set of observations by creating an instance of a network class and calling its inference method.
**parameters**: The parameters of this Function.
· observations: a set of observations to be used for inference, the structure and content of which are specific to the network class being instantiated
**Code Description**: This function creates an instance of a network class using the provided constructor keyword arguments, then calls the inference method on this instance, passing in the given observations. The inference method is responsible for computing value estimates and actions for the full turn based on the input observations. The _inference function essentially serves as a wrapper around the network's inference method, providing a simplified interface for performing inference.
**Note**: The behavior of this function is closely tied to the implementation of the network class being instantiated, particularly its inference method. Understanding the specifics of the network architecture and its inference process is essential to effectively utilizing this function.
**Output Example**: A possible appearance of the code's return value could be a tuple containing two dictionaries, where the first dictionary contains key-value pairs such as 'value_logits' and 'values', and the second dictionary contains key-value pairs such as 'logits' and 'actions'. For example: ({'value_logits': array([...]), 'values': array([...])}, {'logits': array([...]), 'actions': array([...])})
***
***
### FunctionDef get_observation_transformer(cls, class_constructor_kwargs, rng_key)
**get_observation_transformer**: The function of get_observation_transformer is to return an instance of GeneralObservationTransformer, which is used to transform observations.
**parameters**: The parameters of this Function.
· cls: This parameter represents the class itself and is used to access class methods.
· class_constructor_kwargs: These are keyword arguments that would be used to construct an instance of the class, but they are not actually used in this function.
· rng_key: This is an optional parameter that represents a random number generator key, which can be used to introduce randomness in the observation transformation process.
**Code Description**: The description of this Function. 
The get_observation_transformer function creates and returns an instance of GeneralObservationTransformer, passing the provided rng_key to its constructor. The class_constructor_kwargs parameter is not utilized within the function. This function is called by other methods in the Network class, such as initial_inference_params_and_state and zero_observation, which rely on it to obtain an observation transformer for further processing. In the context of initial_inference_params_and_state, the returned transformer is used to generate a zero observation that is then used to initialize the network's parameters and state. Similarly, in zero_observation, the transformer is directly used to create a zero observation.
**Note**: Points to note about the use of the code. 
It is essential to be aware that the class_constructor_kwargs parameter does not affect the behavior of this function, as it is explicitly deleted within the function body. Additionally, the rng_key parameter allows for the introduction of randomness in the observation transformation process, which may be crucial depending on the specific application.
**Output Example**: A possible appearance of the code's return value could be an instance of GeneralObservationTransformer, such as GeneralObservationTransformer(rng_key=42), where 42 represents a random number generator key.
***
### FunctionDef zero_observation(cls, class_constructor_kwargs, num_players)
**zero_observation**: The function of zero_observation is to return a zero observation for a specified number of players by utilizing an instance of GeneralObservationTransformer obtained through the get_observation_transformer method.
**parameters**: The parameters of this Function.
· cls: This parameter represents the class itself and is used to access class methods, specifically to call the get_observation_transformer method.
· class_constructor_kwargs: These are keyword arguments that would be passed to construct an instance of the class, which are then used by the get_observation_transformer method.
· num_players: This parameter specifies the number of players for which a zero observation is to be generated.
**Code Description**: The description of this Function. 
The zero_observation function operates by first calling the get_observation_transformer method on the class, passing in the provided class_constructor_kwargs. This method returns an instance of GeneralObservationTransformer, which is then used to generate a zero observation for the specified number of players by calling its zero_observation method with num_players as an argument. The result of this operation is directly returned by the zero_observation function. 
**Note**: Points to note about the use of the code. 
It is crucial to understand that the get_observation_transformer method plays a pivotal role in this process, as it provides the necessary observation transformer instance. Additionally, the class_constructor_kwargs passed to zero_observation are used by get_observation_transformer, even though they do not directly influence the behavior of zero_observation itself.
**Output Example**: A possible appearance of the code's return value could be the result of calling the zero_observation method on an instance of GeneralObservationTransformer, which would depend on the specific implementation details of the GeneralObservationTransformer class and its zero_observation method.
***
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to initialize the Network class by setting up its attributes and components.
**parameters**: The parameters of this Function.
· rnn_ctor: Constructor for the RNN, which will be used to create an instance of the RNN with the provided batch_norm_config and rnn_kwargs.
· rnn_kwargs: Keyword arguments for the RNN constructor.
· name: A string representing the name of this module, defaulting to "delta".
· num_players: An integer specifying the number of players in the game, defaulting to 7.
· area_mdf: A province_order.MapMDF object representing the path to the mdf file containing a description of the board organized by area.
· province_mdf: A province_order.MapMDF object representing the path to the mdf file containing a description of the board organized by province.
· batch_norm_config: An optional dictionary containing configuration for batch normalization, defaulting to None.

**Code Description**: The __init__ function initializes the Network class by setting up its attributes and components. It first calls the superclass's constructor using super().__init__(). Then, it sets up the area_mdf and province_mdf attributes using the provided values. The function also creates an instance of the RNN using the rnn_ctor and rnn_kwargs, and assigns it to the _rnn attribute. Additionally, it initializes the _moves_encoder, _board_encoder, and _policy_head attributes. The function uses the normalize function from the utils module to normalize the area_mdf and province_mdf values.

The __init__ function also sets up the batch normalization configuration using the batch_norm_config parameter. If batch_norm_config is not provided, it defaults to None. The function then creates an instance of the BoardEncoder class, passing in the normalized area_mdf and province_mdf values, as well as the batch_norm_config.

The Network class relies on several other classes and functions, including the RNN class, the BoardEncoder class, and the normalize function from the utils module. These dependencies are used to create a complex neural network architecture that can process game-related data.

**Note**: When using the __init__ function, it is essential to provide valid values for the area_mdf and province_mdf parameters, as these are used to initialize critical components of the Network class. Additionally, the batch_norm_config parameter should be carefully configured to ensure proper batch normalization during training. The num_players parameter should also be set according to the specific game being modeled.
***
### FunctionDef loss_info(self, step_types, rewards, discounts, observations, step_outputs)
**Target Object Documentation**

## Overview

The Target Object is a fundamental entity designed to represent a specific goal or objective within a system or application. It serves as a focal point for various operations, allowing users to define, track, and achieve desired outcomes.

## Properties

The following properties are associated with the Target Object:

* **ID**: A unique identifier assigned to each Target Object instance, enabling efficient retrieval and manipulation.
* **Name**: A descriptive label assigned to the Target Object, providing context and clarity for users.
* **Description**: An optional text field allowing users to provide additional information about the Target Object, facilitating understanding and collaboration.

## Methods

The Target Object supports the following methods:

* **Create**: Initializes a new instance of the Target Object, assigning a unique ID and allowing users to set the Name and Description properties.
* **Read**: Retrieves an existing Target Object instance by its ID, returning the associated properties and values.
* **Update**: Modifies an existing Target Object instance, allowing users to update the Name and Description properties.
* **Delete**: Removes a Target Object instance from the system, deleting all associated data and references.

## Relationships

The Target Object can be related to other entities within the system, including:

* **Users**: Multiple users can be assigned to a single Target Object, enabling collaboration and shared ownership.
* **Tasks**: Target Objects can be linked to multiple tasks, representing specific actions or activities required to achieve the desired outcome.

## Constraints

The following constraints apply to the Target Object:

* **Uniqueness**: Each Target Object instance must have a unique ID, ensuring efficient retrieval and preventing data duplication.
* **Data Validation**: The Name and Description properties are subject to validation rules, ensuring that only valid and relevant data is stored.

## Security

Access to the Target Object is governed by a set of security rules, including:

* **Authentication**: Users must be authenticated before creating, reading, updating, or deleting Target Object instances.
* **Authorization**: Users must have the necessary permissions to perform operations on Target Object instances, ensuring that only authorized individuals can modify or delete data.

By understanding the properties, methods, relationships, constraints, and security measures associated with the Target Object, users can effectively utilize this entity to achieve their goals and objectives within the system.
***
### FunctionDef loss(self, step_types, rewards, discounts, observations, step_outputs)
**loss**: The function of loss is to calculate the imitation learning loss.
**parameters**: The parameters of this Function.
· step_types: a tensor representing the type of each step
· rewards: a tensor representing the reward at each step
· discounts: a tensor representing the discount factor at each step
· observations: a tuple containing the initial observation, step observations, and number of actions
· step_outputs: a dictionary containing the network outputs produced by inference

**Code Description**: The loss function calculates the imitation learning loss by calling the loss_info function with the provided parameters. The loss_info function computes various losses, including policy loss, value loss, and entropy, based on the input tensors and network outputs. These losses are then used to calculate the total loss, which is returned by the loss function. The loss_info function also calculates additional metrics such as accuracy and whole accuracy.

The loss function relies on the loss_info function to perform the actual calculation of the losses. The loss_info function takes into account the step types, rewards, discounts, observations, and network outputs to compute the losses. It uses various operations such as concatenation, broadcasting, and masking to manipulate the input tensors and calculate the losses.

The relationship between the loss function and its callee, the loss_info function, is that the loss function acts as a wrapper around the loss_info function. The loss function provides the necessary parameters to the loss_info function and returns the total loss calculated by the loss_info function.

**Note**: The loss function assumes that the input tensors are in the correct format and shape. It also assumes that the network outputs produced by inference are valid and contain the necessary information for calculating the losses. Additionally, the loss function does not perform any error checking or handling on its own and relies on the caller to provide valid inputs.

**Output Example**: The output of the loss function is a single value representing the total loss, which can be used for training and optimization purposes. For example, the output might look like: 0.1234, indicating that the total loss is 0.1234.
***
### FunctionDef shared_rep(self, initial_observation)
**shared_rep**: The function of shared_rep is to process shared information by all units that require an order, encoding board state, season, and previous moves, and computing value head.
**parameters**: The parameters of this Function.
· initial_observation: a dictionary containing the initial observation, including season, build numbers, board state, and last moves phase board state, where each value is a jnp.ndarray.
**Code Description**: This function takes in an initial observation and processes it to extract relevant information. It first extracts the season, build numbers, board state, and last moves from the initial observation. Then, it computes the moves actions by summing the encoded actions since the last moves phase and concatenates this with the last moves. The function then computes the board representation using the _board_encoder and the last moves representation using the _last_moves_encoder. These representations are concatenated to form the area representation. Finally, the function computes the value head by applying the _value_mlp to the mean of the area representation.
The shared_rep function is called by the inference function in the Network class, which uses its output to initialize the inference states for each player and then applies the RNN to compute the value estimates and actions for the full turn. The shared_rep function provides the necessary information for the inference function to make decisions.
**Note**: The shared_rep function assumes that the input initial observation is in the correct format, with the required keys and jnp.ndarray values. It also relies on the _board_encoder, _last_moves_encoder, and _value_mlp functions to compute the board representation, last moves representation, and value head, respectively.
**Output Example**: The output of the shared_rep function is a tuple containing two elements: a dictionary with value logits and values, and the area representation. For example:
(
    {
        'value_logits': jnp.array([0.5, 0.3, 0.2]),
        'values': jnp.array([0.7, 0.2, 0.1])
    },
    jnp.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
)
***
### FunctionDef initial_inference(self, shared_rep, player)
**initial_inference**: The function of initial_inference is to set up the initial state for implementing inter-unit dependence.
**parameters**: The parameters of this Function.
· shared_rep: a jnp.ndarray representing the shared representation
· player: a jnp.ndarray representing the player information
**Code Description**: This function takes in two parameters, shared_rep and player, where shared_rep is a jnp.ndarray representing the shared representation and player is a jnp.ndarray representing the player information. It calculates the batch size from the shape of the shared_rep and then uses jax.vmap to apply a partial function to the shared_rep and player. The partial function takes elements from the shared_rep based on the indices provided by the player, effectively selecting specific parts of the shared representation for each player. The function also initializes the state of an RNN using the _rnn.initial_state method with the calculated batch size. This initial state is then returned along with the result of the jax.vmap operation. In the context of the project, this function is called by the inference method to set up the initial inference states for each player before applying the RNN.
**Note**: The output of this function is a tuple containing two elements: the first element is a dictionary where each value is a jnp.ndarray representing the selected parts of the shared representation for each player, and the second element is the initial state of the RNN. It's also worth noting that the _rnn.initial_state method is used to initialize the RNN state, which suggests that this function is part of a larger network architecture.
**Output Example**: A possible appearance of the code's return value could be a tuple containing a dictionary with jnp.ndarray values and an object representing the initial RNN state, for example: ({'key1': jnp.array([...]), 'key2': jnp.array([...])}, rnn_state_object)
***
### FunctionDef step_inference(self, step_observation, inference_internal_state, all_teacher_forcing)
**step_inference**: The function of step_inference is to compute logits for one unit that requires order.
**parameters**: The parameters of this Function.
· step_observation: a dictionary containing observation data for the current step, including areas and legal actions mask
· inference_internal_state: the internal state used for inference, which includes board representation for each player and RelationalOrderDecoder previous state
· all_teacher_forcing: a boolean flag indicating whether to use teacher forcing for all units

**Code Description**: The step_inference function is designed to process sequential order-related information. It takes in the current observation data, internal inference state, and an optional flag for teacher forcing. The function first extracts the area representation and RNN state from the inference internal state. Then, it calculates the average area representation by taking a weighted sum of the areas using the area representation as weights. This average area representation is then used to create a RecurrentOrderNetworkInput instance, which encapsulates the necessary input data for the recurrent neural network. The function then calls the _rnn method to compute logits and update the RNN state. The computed logits are then used to calculate the policy and legal action mask. Finally, the function returns the action information for this unit and the updated inference internal state.

The step_inference function is called by other components in the project, such as the apply_one_step method, which applies the step_inference function to each player's observation data. The output of the step_inference function is used to update the internal state and compute the next action.

**Note**: When using the step_inference function, it is essential to ensure that all parameters are properly initialized and passed to the function. Additionally, the specific requirements for each parameter, such as shape and data type, must be adhered to in order to maintain the integrity of the network's functionality.

**Output Example**: The output of the step_inference function will be a tuple containing two elements: a dictionary with action information (actions, legal_action_mask, policy, logits) and the updated inference internal state. For example:
({
    'actions': jnp.array([...]),
    'legal_action_mask': jnp.array([...]),
    'policy': jnp.array([...]),
    'logits': jnp.array([...])
}, (area_representation, updated_rnn_state))
***
### FunctionDef inference(self, observation, num_copies_each_observation, all_teacher_forcing)
**inference**: The function of inference is to compute value estimates and actions for the full turn.
**parameters**: The parameters of this Function.
· observation: a tuple containing the initial observation, step observations, and sequence lengths, where each element is a dictionary or array with specific structure and content
· num_copies: not present in the provided function signature but the function seems to be designed to handle batched inputs, however there's an optional parameter called all_teacher_forcing which defaults to False, another parameter num_actions is also not present but sequence_lengths seem to serve a similar purpose, instead we have num_copies in the caller functions
· all_teacher_forcing: an optional boolean parameter that defaults to False, used to control teacher forcing behavior
**Code Description**: This function takes in a tuple of observations and other parameters, then uses these inputs to compute value estimates and actions for the full turn. It first unpacks the observation tuple into initial observation, step observations, and sequence lengths. Then it calls the shared inference logic, which applies the network's inference process to the given inputs. The function returns a tuple containing the initial outputs and the step outputs. The initial outputs contain value estimates, while the step outputs contain action probabilities and other information. The function is designed to handle batched inputs and can be used with or without teacher forcing.
**Note**: The inference function seems to be part of a larger network architecture, likely a reinforcement learning model. It's used by other functions in the project, such as loss_info, to compute losses and update the network's policy. The function's behavior can be controlled using the all_teacher_forcing parameter, which determines whether teacher forcing is applied during inference.
**Output Example**: A possible appearance of the code's return value could be a tuple containing two dictionaries, where the first dictionary contains key-value pairs such as 'value_logits' and 'values', and the second dictionary contains key-value pairs such as 'logits' and 'actions'. For example: ({'value_logits': array([...]), 'values': array([...])}, {'logits': array([...]), 'actions': array([...])})
#### FunctionDef _apply_rnn_one_player(player_step_observations, player_sequence_length, player_initial_state)
**_apply_rnn_one_player**: The function of _apply_rnn_one_player is to apply a Recurrent Neural Network (RNN) to a sequence of player observations for one player.

**parameters**: The parameters of this Function.
· player_step_observations: A tensor with shape [B, 17, ...] representing the observations of the player at each step.
· player_sequence_length: A tensor with shape [B] representing the length of the sequence for each player.
· player_initial_state: A tensor with shape [B] representing the initial state of the RNN for each player.

**Code Description**: The function first converts the player_step_observations to JAX arrays using tree.map_structure and jnp.asarray. It then defines a nested function apply_one_step, which applies the RNN to one step of the sequence. This function takes in the current state and the index of the step, and returns the updated state and output. The function uses hk.scan to apply the apply_one_step function to each step of the sequence, and finally swaps the axes of the output tensor.

The apply_one_step function first extracts the observations at the current step using tree.map_structure and lambda functions. It then calls the step_inference method to get the output and next state of the RNN. The function then updates the state using a conditional statement that checks if the current index is greater than or equal to the sequence length. If it is, the function keeps the old state; otherwise, it uses the new state. The function also updates the output in a similar way.

**Note**: The function assumes that the input tensors have the correct shape and type. It also assumes that the step_inference method is implemented correctly and returns the expected output. Additionally, the function uses JAX and Haiku libraries, so it requires these libraries to be installed and imported.

**Output Example**: The output of this function will be a tensor with shape [B, MAX_ORDERS, ...] representing the output of the RNN at each step for each player, where MAX_ORDERS is a constant defined in the action_utils module. For example, if B = 32 and MAX_ORDERS = 10, the output might look like: 
[[[0.1, 0.2], [0.3, 0.4], ...], [[0.5, 0.6], [0.7, 0.8], ...], ...]
##### FunctionDef apply_one_step(state, i)
**apply_one_step**: The function of apply_one_step is to compute and update the state and output for one step of the network's inference process.
**parameters**: The parameters of this Function.
· state: the current internal state of the network
· i: an index used to select specific elements from the player step observations
**Code Description**: The apply_one_step function takes in the current state and an index i, and uses these to compute the output and next state by calling the step_inference function. This function is designed to process sequential order-related information for one unit that requires order. It first extracts specific elements from the player step observations using the provided index, then calls step_inference with these extracted elements, the current state, and a flag indicating whether all teacher forcing should be applied. The result of this call is an output and a next state, which are then used to update the internal state by selectively replacing elements based on the index i and the player sequence length. Finally, the function returns the updated state and an output that has been selectively replaced with zeros.
The apply_one_step function relies heavily on the step_inference function to perform its computations, using this function to calculate the output and next state for one unit that requires order. This relationship highlights the importance of proper initialization and passing of parameters to both functions in order to maintain the integrity of the network's functionality.
**Note**: When using the apply_one_step function, it is essential to ensure that all parameters are properly initialized and passed to the function, including the state and index i. Additionally, the specific requirements for each parameter, such as shape and data type, must be adhered to in order to maintain the integrity of the network's functionality.
**Output Example**: The output of the apply_one_step function will be a tuple containing two elements: the updated internal state and an output that has been selectively replaced with zeros. For example, this could be a pair of nested dictionaries or arrays, where each element represents the updated state or output for one unit in the network.
###### FunctionDef update(x, y, i)
**update**: The function of update is to conditionally return either the input x or y based on a given index i and player sequence length.
**parameters**: The parameters of this Function.
· x: The first input value to be returned if the condition is not met.
· y: The second input value to be returned if the condition is met.
· i: The index used in the conditional check, defaults to the outer scope variable i.

**Code Description**: This function utilizes the jnp.where function from the JAX library, which returns either the first or second argument based on a given condition. In this case, the condition checks whether the index i is greater than or equal to the player sequence length at the current position np.s_[:,] + (None,) * (x.ndim - 1). If the condition is true, it returns y; otherwise, it returns x.

**Note**: The use of np.s_[:,] suggests that this function operates on a multi-dimensional array and relies on NumPy's advanced indexing. Additionally, the default value of i being set to the outer scope variable i implies that this function may be used within a loop or other iterative context where the index is updated externally.

**Output Example**: The output of this function will be either the input x or y, depending on the conditional check. For instance, if x is an array [1, 2, 3] and y is an array [4, 5, 6], and the condition is met for the first two elements but not the third, the output might look like [4, 5, 3].
***
***
***
***
