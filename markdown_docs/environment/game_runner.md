## ClassDef DiplomacyTrajectory
**DiplomacyTrajectory**: The function of DiplomacyTrajectory is to store data from a Diplomacy game.
attributes: The attributes of this Class.
· observations: A list that stores Observation objects representing the state of the game at each step.
· legal_actions: A list of numpy arrays, where each array contains the legal actions available to players in a particular step.
· actions: A list of numpy arrays, where each array contains the actions taken by players in a particular step.
· step_outputs: A list of dictionaries, where each dictionary stores additional outputs from policies at each step.
· returns: An optional numpy array that represents the final returns (scores) for each player after the game ends.

Code Description: The DiplomacyTrajectory class is designed to encapsulate all relevant data collected during a single run of a Diplomacy game. It initializes with empty lists for observations, legal actions, and actions taken by players, as well as step outputs from policies. These attributes are populated through the append_step method, which takes in an observation, legal actions, player actions, and policy outputs at each turn of the game. The terminate method is used to set the final returns after the game concludes.

The DiplomacyTrajectory class plays a crucial role in the run_game function within the environment/game_runner.py module. This function simulates a full Diplomacy game by iteratively updating the game state based on player actions and policy outputs, appending each step's data to a DiplomacyTrajectory object. The game continues until it reaches a terminal state or exceeds a specified maximum length. Upon termination, the final returns are calculated and stored in the DiplomacyTrajectory object, which is then returned as the output of the run_game function.

Note: It is important to ensure that the append_step method is called with data corresponding to each turn of the game in chronological order to maintain the integrity of the trajectory. The terminate method should only be called once after the game has ended to set the final returns.

Output Example: A possible appearance of the code's return value could be a DiplomacyTrajectory object containing the following attributes:
· observations: [Observation1, Observation2, ..., ObservationN]
· legal_actions: [np.array([...]), np.array([...]), ..., np.array([...])]
· actions: [np.array([...]), np.array([...]), ..., np.array([...])]
· step_outputs: [{'policy1': {...}, 'policy2': {...}}, {'policy1': {...}, 'policy2': {...}}, ..., {'policy1': {...}, 'policy2': {...}}]
· returns: np.array([score_player1, score_player2, ..., score_player7])
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to initialize a new instance of the DiplomacyTrajectory class by setting up empty lists and an optional return value.

parameters: This Function does not take any parameters.
Code Description: Upon instantiation, this method initializes several attributes that are intended to store data related to observations, legal actions, taken actions, step outputs, and returns. Specifically:
- `self.observations` is initialized as an empty list of Observation objects from the utils module, designed to hold observation data throughout the trajectory.
- `self.legal_actions` is initialized as an empty list of numpy arrays, intended to store the set of legal actions available at each step in the trajectory.
- `self.actions` is also initialized as an empty list of numpy arrays, meant to record the actions taken during the trajectory.
- `self.step_outputs` is initialized as an empty list of dictionaries with string keys and values of any type, used to store outputs from each step in the trajectory.
- `self.returns` is initialized as None but is expected to be a numpy array that will eventually hold return values associated with the trajectory.

Note: The user should ensure that these attributes are populated appropriately throughout the lifecycle of the DiplomacyTrajectory instance, typically by appending data to the lists and setting the returns attribute when necessary.
Output Example: Upon initialization, an instance of DiplomacyTrajectory would have the following internal state:
- observations = []
- legal_actions = []
- actions = []
- step_outputs = []
- returns = None
***
### FunctionDef append_step(self, observation, legal_actions, actions, step_outputs)
**append_step**: The function of append_step is to add a new step's data to the trajectory of a game.

parameters: 
· observation: An instance of utils.Observation representing the current state of the game.
· legal_actions: A numpy array containing the legal actions available to each player in the current state.
· actions: A numpy array containing the actions taken by each player in the current step.
· step_outputs: A dictionary that holds additional outputs from policies or other components for the current step.

Code Description: The append_step function is designed to store information about a single step in a game trajectory. It takes four parameters: observation, legal_actions, actions, and step_outputs. These parameters encapsulate all relevant data for a particular turn of the game. Specifically:
- The observation parameter captures the state of the game at that point.
- The legal_actions parameter lists the permissible actions each player can take given the current state.
- The actions parameter records the actual actions chosen by each player during this step.
- The step_outputs parameter is a dictionary that may include additional information, such as intermediate results or diagnostics from policies.

The function appends these pieces of data to their respective lists within the DiplomacyTrajectory object. This allows for the accumulation of game history over multiple turns, which can be used later for analysis, training, or replaying the game.

In the context of the project, this function is called by the run_game method in environment/game_runner.py. The run_game method simulates a full game of diplomacy, executing turns until the game ends due to reaching a terminal state (e.g., one player winning) or exceeding a specified maximum length. During each turn, run_game collects data about the current state and actions taken by players, then calls append_step to store this information in the DiplomacyTrajectory object.

Note: It is crucial that all parameters passed to append_step are correctly formatted and correspond to the same game step to maintain the integrity of the trajectory. Failure to do so can lead to inconsistencies or errors when analyzing or using the stored data.
***
### FunctionDef terminate(self, returns)
**terminate**: The function of terminate is to set the final returns or outcomes of a game trajectory.
parameters: 
· returns: This parameter represents the final results or rewards obtained from the game, which will be assigned to the `returns` attribute of the DiplomacyTrajectory object.

Code Description: The `terminate` method is designed to capture and store the final outcome of a game within a `DiplomacyTrajectory` instance. It accepts one argument, `returns`, which encapsulates the results or rewards from the completed game session. This method is crucial for marking the end of a game trajectory by assigning the computed returns to the `returns` attribute of the object. The `returns` could represent various metrics such as scores, win/loss conditions, or other relevant data depending on the context of the game.

In the project's structure, this function is called within the `run_game` method located in `environment/game_runner.py`. Specifically, after the main loop that simulates the turns of a Diplomacy game concludes (either due to reaching the maximum length, forced draw conditions being met, or the game state becoming terminal), the `returns` are determined. If no specific returns have been set during the game (i.e., `returns` is None), they are calculated based on the final state of the game using `state.returns()`. Subsequently, the `terminate` method is invoked with these computed returns to finalize the trajectory.

Note: It is essential that this function is called after all game logic has been executed and the final outcomes have been determined to ensure accurate recording of the game's results within the trajectory object.
Output Example: If the game ends with a win for player 1, the `returns` might be an array like `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`, indicating that player 1 received a full reward while all other players received no rewards. This array would then be assigned to the `returns` attribute of the `DiplomacyTrajectory` instance through the `terminate` method.
***
## FunctionDef _draw_returns(points_per_supply_centre, board, num_players)
**_draw_returns**: The function of _draw_returns is to compute the returns (number of supply centers) when the game ends in a draw.

**parameters**: 
· points_per_supply_centre: A boolean indicating whether to assign points per supply centre in a draw (rather than 0/1 for win/loss).
· board: A numpy array representing the current state of the game board.
· num_players: An integer representing the number of players in the game.

**Code Description**: The function _draw_returns calculates the returns for each player when a game ends in a draw. If `points_per_supply_centre` is True, it computes the number of supply centers each player controls and uses these numbers as their respective returns. Otherwise, it assigns a return value of 1 to players who control at least one supply center and 0 to those who do not. The function then normalizes these returns so that they sum up to 1 by dividing each player's return by the total sum of all returns.

The function is called within the `run_game` method in the same file, specifically when a draw is forced either due to the elimination of a particular slot or after exceeding a certain number of years with a specified probability. In these scenarios, `_draw_returns` determines how points should be distributed among players based on their supply center control.

**Note**: The function assumes that `utils.sc_provinces(i, board)` returns a list of provinces controlled by player `i`. It also expects `num_players` to correctly reflect the number of participants in the game.

**Output Example**: If there are three players and `points_per_supply_centre` is True, with supply centers distributed as [3, 2, 1] respectively, the function will return an array like `[0.5, 0.3333, 0.1667]`. If `points_per_supply_centre` is False and players control at least one supply center, it might return `[0.5, 0.5, 0]` assuming the third player has no supply centers.
## FunctionDef run_game(state, policies, slots_to_policies, max_length, min_years_forced_draw, forced_draw_probability, points_per_supply_centre, draw_if_slot_loses)
**run_game**: The function of run_game is to simulate and execute a full game of diplomacy based on given policies and parameters.

parameters: 
· state: A DiplomacyState object representing the initial state of the game, specifically set to Spring 1901.
· policies: A sequence of Policy objects that define the strategies for each player in the game.
· slots_to_policies: A sequence mapping each player slot (position) to an index in the policies list, indicating which policy controls which player.
· max_length: An optional integer specifying the maximum number of full diplomacy turns the game can run. If not provided, the game will continue until a terminal state is reached.
· min_years_forced_draw: An integer representing the minimum number of years after which a forced draw may be considered based on probability.
· points_per_supply_centre: A boolean indicating whether to assign points per supply centre in a draw (rather than 0/1 for win/loss).
· board: A numpy array representing the current state of the game board, used primarily by _draw_returns when calculating returns in case of a draw.
· num_players: An integer representing the number of players in the game, also used by _draw_returns.
· draw_probability: A float indicating the probability of forcing a draw after min_years_forced_draw years have passed.

Code Description: The run_game function simulates a full game of diplomacy. It initializes with a given state and policies for each player. During each turn, it collects legal actions available to players, requests actions from their respective policies, and updates the game state accordingly. The simulation continues until one of the following conditions is met:
- A terminal state is reached (e.g., one player wins).
- The maximum number of turns specified by max_length is exceeded.
- After min_years_forced_draw years have passed, a draw is forced with a probability given by draw_probability.

If a draw is forced, the function _draw_returns is called to compute the returns based on supply center control or other criteria. The results are then stored in a DiplomacyTrajectory object using the append_step method for each turn and the terminate method when the game ends.

Note: It is crucial that the policies provided can handle the state representation used by the game, and that the slots_to_policies mapping correctly assigns policies to player slots. Additionally, the function assumes that the initial state is set to Spring 1901 as required by the Diplomacy rules.

Output Example: A possible return value of run_game could be a DiplomacyTrajectory object containing the history of the game, including observations, legal actions, chosen actions, and step outputs for each turn. If the game ends in a win for player 1 with a score of 10 supply centers out of a total of 20, and players 2 and 3 have 5 and 5 respectively, the returns might be [1.0, 0.5, 0.5]. The trajectory would also include detailed information about each turn's state and actions taken by each player.
