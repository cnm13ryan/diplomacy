## FunctionDef get_config
**get_config**: The function of get_config is to return a configuration dictionary specifically tailored for network settings.

parameters: This Function does not take any parameters.
Code Description: The function initializes an instance of `config_dict.ConfigDict` and populates it with various network-related configurations. It sets the `network_class` attribute to `network.Network`. For the `network_kwargs`, it constructs a detailed configuration dictionary using `config_dict.create()`. This includes specifying the type of recurrent neural network (RNN) constructor as `network.RelationalOrderDecoder` along with its parameters such as `adjacency`, `filter_size`, and `num_cores`. The adjacency matrix is derived from predefined map data (`province_order.MapMDF.STANDARD_MAP`) processed through normalization. Additional parameters like `name`, `num_players`, `area_mdf`, `is_training`, and various filter sizes and core numbers are also set to configure the network appropriately for its intended use.

Note: Ensure that all referenced modules (`config_dict`, `network`, `province_order`) are correctly imported in your project to avoid runtime errors. The function is designed to provide a standardized configuration, so modifications should be made with consideration of how they affect the overall functionality and performance of the network.
Output Example: Mock up a possible appearance of the code's return value.

{
  'network_class': <class 'network.Network'>,
  'network_kwargs': {
    'rnn_ctor': <class 'network.RelationalOrderDecoder'>,
    'rnn_kwargs': {
      'adjacency': [[0, 1, 0], [1, 0, 1], [0, 1, 0]], # Example adjacency matrix
      'filter_size': 64,
      'num_cores': 4
    },
    'name': "delta",
    'num_players': 7,
    'area_mdf': <province_order.MapMDF.BICOASTAL_MAP: 2>,
    'province_mdf': <province_order.MapMDF.STANDARD_MAP: 1>,
    'is_training': False,
    'shared_filter_size': 160,
    'player_filter_size': 160,
    'num_shared_cores': 12,
    'num_player_cores': 3,
    'value_mlp_hidden_layer_sizes': (256,),
    'actions_since_last_moves_embedding_size': 10
  }
}
