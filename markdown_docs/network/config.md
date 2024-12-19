## FunctionDef get_config
**get_config**: The function `get_config` returns a configuration dictionary for a network.

**parameters**: This function does not take any parameters.

**Code Description**: The `get_config` function is designed to create and return a configuration settings object for a neural network model. It uses the `config_dict.ConfigDict` class to store and organize various hyperparameters and settings that define the behavior of the network.

The function starts by initializing an empty `ConfigDict` object named `config`. It then sets the `network_class` attribute of this config object to `network.Network`, indicating the class of the neural network to be used.

Next, it configures the kwargs for the network by creating another `ConfigDict` under `config.network_kwargs`. This sub-config includes several key settings:

- `rnn_ctor`: Set to `network.RelationalOrderDecoder`, which specifies the type of recurrent neural network (RNN) constructor to be used.

- `rnn_kwargs`: A dictionary containing parameters for the RNN constructor, including:

  - `adjacency`: Computed by normalizing an adjacency matrix built from map data using `province_order.build_adjacency` and `province_order.get_mdf_content` with `MapMDF.STANDARD_MAP`.

  - `filter_size`: Set to 64, likely referring to the size of filters in convolutional layers.

  - `num_cores`: Set to 4, possibly indicating the number of parallel processing units or attention heads.

- `name`: Set to "delta", which might be a identifier for this particular network configuration.

- `num_players`: Set to 7, indicating that the network is configured to handle games with seven players.

- `area_mdf` and `province_mdf`: Both set to `MapMDF.BICOASTAL_MAP` and `MapMDF.STANDARD_MAP` respectively, specifying different map definitions for areas and provinces.

- `is_training`: Set to False, suggesting that this configuration is for inference or evaluation rather than training.

- `shared_filter_size` and `player_filter_size`: Both set to 160, likely referring to the size of filters in shared and player-specific layers.

- `num_shared_cores` and `num_player_cores`: Set to 12 and 3 respectively, possibly indicating the number of cores or attention heads in shared and player-specific components.

- `value_mlp_hidden_layer_sizes`: Set to a tuple containing 256, defining the sizes of hidden layers in a multi-layer perceptron (MLP) used for value estimation.

- `actions_since_last_moves_embedding_size`: Set to 10, likely referring to the embedding size for representing the number of actions since the last moves.

**Note**: This function is crucial for setting up the network's architecture and parameters. It ensures that all necessary configurations are centralized and easily manageable. Users should be aware that modifying these parameters can significantly affect the network's performance and behavior. Additionally, since this configuration seems tailored for a specific type of game (with seven players and specific map definitions), it may not be directly applicable to different scenarios without adjustments.

**Output Example**: The function returns a `ConfigDict` object structured as follows:

```python
config = ConfigDict()
config.network_class = network.Network
config.network_kwargs = ConfigDict(
    rnn_ctor=network.RelationalOrderDecoder,
    rnn_kwargs=ConfigDict(
        adjacency=normalized_adjacency_matrix,
        filter_size=64,
        num_cores=4,
    ),
    name="delta",
    num_players=7,
    area_mdf=province_order.MapMDF.BICOASTAL_MAP,
    province_mdf=province_order.MapMDF.STANDARD_MAP,
    is_training=False,
    shared_filter_size=160,
    player_filter_size=160,
    num_shared_cores=12,
    num_player_cores=3,
    value_mlp_hidden_layer_sizes=(256,),
    actions_since_last_moves_embedding_size=10
)
```

Here, `normalized_adjacency_matrix` is the result of processing map data through `province_order.build_adjacency` and `network.normalize_adjacency`.
