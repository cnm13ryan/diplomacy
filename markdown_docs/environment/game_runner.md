## ClassDef DiplomacyTrajectory
**DiplomacyTrajectory**: The function of DiplomacyTrajectory is to store data from a Diplomacy game.
**attributes**: The attributes of this Class.
· observations: A list of Observation objects representing the state of the game at each step.
· legal_actions: A list of numpy arrays representing the legal actions available to each player at each step.
· actions: A list of numpy arrays representing the actions taken by each player at each step.
· step_outputs: A list of dictionaries containing output from policies at each step.
· returns: An optional numpy array representing the return values for the game.

**Code Description**: The DiplomacyTrajectory class is designed to record and store data from a Diplomacy game. It has methods to append new steps to the trajectory, including observations, legal actions, actions taken, and output from policies. The terminate method allows setting the return values for the game. This class is used in conjunction with the run_game function, which runs a game of diplomacy and returns a DiplomacyTrajectory object containing the recorded data.

**Note**: When using this class, it's essential to ensure that the append_step method is called consistently to maintain accurate records of the game state. Additionally, the terminate method should be called once the game has ended to set the return values.

**Output Example**: A possible DiplomacyTrajectory object may contain the following data:
- observations: [Observation1, Observation2, ...]
- legal_actions: [numpy array of legal actions at step 1, numpy array of legal actions at step 2, ...]
- actions: [numpy array of actions taken at step 1, numpy array of actions taken at step 2, ...]
- step_outputs: [{policy output at step 1}, {policy output at step 2}, ...]
- returns: numpy array of return values for the game

This data can be used to analyze and understand the progression of the Diplomacy game.
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to initialize the DiplomacyTrajectory object by setting up its internal state with empty lists and a None value for returns.
**parameters**: There are no parameters for this Function.
**Code Description**: This function initializes the DiplomacyTrajectory object's attributes, which include self.observations, self.legal_actions, self.actions, self.step_outputs, and self.returns. The self.observations, self.legal_actions, self.actions, and self.step_outputs are initialized as empty lists, indicating they will store collections of Observation objects, numpy arrays representing legal actions, numpy arrays representing actions taken, and dictionaries containing step output data, respectively. The self.returns attribute is initialized with a value of None, suggesting it will be used to store return values, potentially in the form of a numpy array, at a later point.
**Note**: It's essential to note that this function does not take any parameters and solely focuses on setting up the initial state of the DiplomacyTrajectory object. The attributes initialized here are likely to be populated with data through other methods or functions within the class.
**Output Example**: Since __init__ is a special method in Python classes known as a constructor, it doesn't explicitly return a value like other functions might. Instead, its purpose is to set up the object's state upon creation. For example, after creating an instance of DiplomacyTrajectory, you could access its attributes to see their initial states, such as empty lists for observations, legal actions, actions, and step outputs, and None for returns.
***
### FunctionDef append_step(self, observation, legal_actions, actions, step_outputs)
**append_step**: The function of append_step is to add a new step to the DiplomacyTrajectory by storing the observation, legal actions, actions taken, and outputs from each policy at that step.
**parameters**: The parameters of this Function.
· observation: an instance of utils.Observation representing the current state of the game
· legal_actions: a numpy array containing the legal actions available to each player
· actions: a numpy array containing the actions taken by each player
· step_outputs: a dictionary where the keys are strings and the values are of type Any, representing the outputs from each policy at this step
**Code Description**: The append_step function is used to store the history of a game as it progresses. It takes in the current observation, legal actions available to each player, the actions actually taken by each player, and any additional output from each policy. This information is then appended to the DiplomacyTrajectory object's internal lists for observations, legal actions, actions, and step outputs. The function is called at each step of a game by the run_game function in the game_runner module, which manages the execution of a game of diplomacy. By storing this information, the append_step function enables the creation of a complete record of the game, including all states, actions, and policy outputs.
**Note**: When using the append_step function, it is essential to ensure that the input parameters are correctly formatted and contain valid data, as incorrect inputs may lead to errors or inconsistencies in the stored game history. Additionally, the function assumes that the DiplomacyTrajectory object has already been initialized and is ready to store new steps.
***
### FunctionDef terminate(self, returns)
**terminate**: The function of terminate is to set the returns value for a DiplomacyTrajectory instance.
**parameters**: The parameters of this Function.
· self: a reference to the current instance of the class
· returns: the value to be set as the returns for the DiplomacyTrajectory instance
**Code Description**: This function is a method of the DiplomacyTrajectory class and is used to terminate the trajectory by setting its returns value. The function takes in two parameters, self and returns, where self is a reference to the current instance of the class and returns is the value to be set as the returns for the DiplomacyTrajectory instance. When this function is called, it simply assigns the provided returns value to the instance's returns attribute. This function is called by the run_game function in the game_runner module when the game has ended, either due to a terminal state being reached or a forced draw. In this context, the terminate function serves as a way to finalize the trajectory and store its outcome.
**Note**: The returns value should be provided by the caller of this function, which is typically determined based on the final state of the game. It's also worth noting that this function does not perform any validation or error checking on the provided returns value, so it's up to the caller to ensure that a valid value is passed in.
**Output Example**: There is no explicit return value for this function, as its purpose is to modify the instance's state rather than produce an output. However, after calling this function, the returns attribute of the DiplomacyTrajectory instance can be accessed to retrieve the stored returns value, which might look something like: `[1, 0, 0, 0, 0, 0, 0]`, indicating the outcome of the game for each player.
***
## FunctionDef _draw_returns(points_per_supply_centre, board, num_players)
**_draw_returns**: The function of _draw_returns is to compute returns when the game ends in a draw, specifically calculating the number of supply centers each player has.
**parameters**: The parameters of this Function.
· points_per_supply_centre: A boolean indicating whether to assign points per supply centre in a draw or use 0/1 for win/loss.
· board: A numpy array representing the game board.
· num_players: An integer specifying the number of players in the game.
**Code Description**: The _draw_returns function calculates the returns for each player when the game ends in a draw. It first checks if points are to be assigned per supply centre. If so, it computes the returns as the number of supply centres controlled by each player. Otherwise, it assigns 1 point to players with at least one supply centre and 0 points to those without any. The function then normalizes these returns by dividing them by their sum, ensuring they add up to 1. This normalization is crucial for determining the relative share of the draw reward for each player. In the context of the game runner, this function is called when a draw is forced due to a player's elimination or when a random draw is triggered after a certain number of years.
**Note**: The use of points_per_supply_centre allows for flexibility in how draws are evaluated, enabling either a binary win/loss assessment or a more nuanced point-based system. Additionally, the normalization step ensures that the returns can be interpreted as probabilities or relative shares of the draw outcome.
**Output Example**: For a game with 7 players where points_per_supply_centre is True and player 0 controls 3 supply centres, player 1 controls 2 supply centres, and so on, the output might look like: [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], indicating the normalized returns for each player based on their controlled supply centres.
## FunctionDef run_game(state, policies, slots_to_policies, max_length, min_years_forced_draw, forced_draw_probability, points_per_supply_centre, draw_if_slot_loses)
**run_game**: The function of run_game is to execute a game of diplomacy based on given parameters and return the trajectory of the game.
**parameters**: The parameters of this Function.
· state: A DiplomacyState object representing the initial state of the game, specifically in Spring 1901.
· policies: A sequence of Policy objects that will be acting during the game.
· slots_to_policies: A sequence mapping each player slot to the index of the corresponding policy in the policies sequence.
· max_length: An optional integer specifying the maximum number of full diplomacy turns to play before terminating the game.
· min_years_forced_draw: An integer representing the minimum years after which a forced draw may be considered.
· forced_draw_probability: A float indicating the probability of a draw each year after the first min_years_forced_draw.
· points_per_supply_centre: A boolean flag determining whether to assign points per supply centre in case of a draw or use 0/1 for win/loss.
· draw_if_slot_loses: An optional integer representing the player slot that, if eliminated, will trigger a draw.

**Code Description**: The run_game function is designed to manage the execution of a diplomacy game. It initializes by checking the consistency of the input parameters, such as ensuring the number of slots matches the expected number of players and validating policy indices. It then iterates through each turn of the game, updating the state based on actions taken by policies, until either the maximum length is reached or the game ends in a terminal state. The function also checks for conditions that trigger a draw, such as a player's elimination or reaching a certain number of years with a specified probability. Throughout the game, it constructs a DiplomacyTrajectory object to store observations, legal actions, actual actions taken, and policy outputs at each step. This trajectory is returned once the game concludes.

The run_game function relies on several other components within its scope, including Policy objects for decision-making, the DiplomacyState class for tracking the game's state, and utility functions like _draw_returns for calculating returns in case of a draw. The function's operation is heavily influenced by these dependencies, underscoring the interconnected nature of the game's logic.

**Note**: It is crucial to ensure that all input parameters are correctly formatted and valid, as incorrect inputs may lead to errors or inconsistencies during gameplay. Additionally, understanding how draws are evaluated (either per supply centre or binary win/loss) is essential for interpreting the game's outcome.

**Output Example**: The return value of run_game will be a DiplomacyTrajectory object containing detailed information about each step of the game, including observations, actions taken by policies, and the final state of the game. This could conceptually resemble a structured dataset where each entry corresponds to a turn in the game, encapsulating the dynamic evolution of the game's state over time.
