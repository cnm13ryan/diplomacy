## FunctionDef get_config
**get_config**: The function of get_config is to return network configuration settings.
**parameters**: This Function has no parameters.
**Code Description**: 
The `get_config` function initializes and returns a `config_dict.ConfigDict` object containing various configuration settings for the network. Here's a detailed breakdown:

1. **Initialization**: A new instance of `ConfigDict` is created to hold all necessary configurations.

2. **Network Class and Keyword Arguments (kwargs)**: 
   - The `network_class` field is set to `network.Network`, specifying the class that will be used for network operations.
   - The `network_kwargs` are defined using another `config_dict.ConfigDict`. This dictionary contains several key-value pairs:
     - `rnn_ctor`: Set to `network.RelationalOrderDecoder`, indicating the type of Recurrent Neural Network (RNN) constructor to use.
     - `rnn_kwargs`: A nested dictionary that further configures the RNN:
       - `adjacency`: The adjacency matrix, which is normalized and built from province order data. This involves multiple steps including building an adjacency matrix using `province_order.build_adjacency`, normalizing it with `network.normalize_adjacency`, and getting MDF content with `province_order.get_mdf_content`.
       - `filter_size` and `num_cores`: Parameters for the RNN, specifying the size of filters and number of cores.
     - Additional fields like `name`, `num_players`, `area_mdf`, `province_mdf`, etc., are set to specific values relevant to the network configuration.

3. **Additional Configuration**: 
   - Flags such as `is_training` are set to indicate whether the model is in training mode or not.
   - Various sizes and numbers of cores are defined, including shared filters (`shared_filter_size`), player-specific filters (`player_filter_size`), number of shared cores (`num_shared_cores`), and number of player cores (`num_player_cores`).
   - The `value_mlp_hidden_layer_sizes` define the hidden layer architecture for value prediction.
   - An embedding size is set for actions since the last moves.

4. **Return**: Finally, the fully configured `config_dict.ConfigDict` object is returned by the function.

**Note**: Ensure that all dependencies (`network`, `province_order`, etc.) are correctly imported and available when this function is called. The configuration settings should be tailored according to specific requirements of the network implementation.

**Output Example**: 
An example output could look like:
```
config = {
    'network_class': <class 'network.Network'>,
    'network_kwargs': {
        'rnn_ctor': <class 'network.RelationalOrderDecoder'>,
        'rnn_kwargs': {
            'adjacency': <normalized adjacency matrix>,
            'filter_size': 64,
            'num_cores': 4
        },
        'name': 'delta',
        'num_players': 7,
        'area_mdf': <specific map definition>,
        'province_mdf': <specific map definition>,
        'is_training': False,
        'shared_filter_size': 160,
        'player_filter_size': 160,
        'num_shared_cores': 12,
        'num_player_cores': 3,
        'value_mlp_hidden_layer_sizes': (256,),
        'actions_since_last_moves_embedding_size': 10
    }
}
```
