## ClassDef DiplomacyTrajectory
**Function Overview**
`DiplomacyTrajectory` is a class designed to capture and store the state transitions during a game of Diplomacy, specifically tracking actions taken by players over multiple turns.

**Parameters**

- **Attributes:**
  - `observations`: A list of observations (game states) captured at each turn.
  - `legal_actions`: A list of legal actions available for each player at each turn.
  - `actions`: A list of actual actions taken by the players at each turn.
  - `step_outputs`: A dictionary containing step outputs from the policy or decision-making process.

**Detailed Explanation**
`DiplomacyTrajectory` is initialized with empty lists to store observations, legal actions, and actions. During gameplay, the `append_step` method is called repeatedly to add new entries to these lists. Each entry in the list represents a turn in the game, capturing the current state of the game (`observation`), the legal moves available (`legal_actions`), the actual moves made by players (`actions`), and any additional outputs from the decision-making process (`step_outputs`).

The `append_step` method takes four parameters:
- `observation`: The current state of the game.
- `legal_actions`: A list representing the legal actions that can be taken in the current state.
- `actions`: A list representing the actual moves made by players in the current turn.
- `step_outputs`: Additional information or outputs from the decision-making process.

After all turns have been processed, the `terminate` method is called to finalize the trajectory. This method takes a single parameter:
- `returns`: The final returns or outcomes of the game.

**Interactions with Other Components**
`DiplomacyTrajectory` interacts with the gameplay logic through the `append_step` and `terminate` methods. These methods are called by the main gameplay loop, which processes each turn of the game and updates the trajectory accordingly.

**Usage Notes**

- **Initialization**: The class is typically instantiated at the start of a new game.
  ```python
  traj = DiplomacyTrajectory()
  ```

- **Appending Steps**: During each turn, `append_step` is called to add the current state, legal actions, and player actions to the trajectory:
  ```python
  observation = get_current_state()
  legal_actions = get_legal_actions(observation)
  actions = take_player_actions(legal_actions)
  step_outputs = policy.get_step_output(actions)
  
  traj.append_step(observation, legal_actions, actions, step_outputs)
  ```

- **Termination**: After the final turn, `terminate` is called to finalize the trajectory:
  ```python
  returns = get_final_returns()
  traj.terminate(returns)
  ```

**Example Usage**
Here is a simplified example of how `DiplomacyTrajectory` might be used in a game loop:

```python
def main_game_loop():
    traj = DiplomacyTrajectory()

    for turn in range(max_turns):
        observation = get_current_state()
        legal_actions = get_legal_actions(observation)
        actions = take_player_actions(legal_actions)
        step_outputs = policy.get_step_output(actions)

        traj.append_step(observation, legal_actions, actions, step_outputs)

    returns = get_final_returns()
    traj.terminate(returns)

    # Now `traj` contains the complete history of the game
```

This example demonstrates how to initialize a trajectory, append steps during each turn, and finalize it after the game concludes.
### FunctionDef __init__(self)
**Function Overview**
The `__init__` method initializes an instance of the `DiplomacyTrajectory` class, setting up various attributes that will store observations, actions, legal actions, step outputs, and returns.

**Parameters**
- None

**Return Values**
- The method does not return any value; it sets up internal state variables for the object.

**Detailed Explanation**
The `__init__` method is called when a new instance of the `DiplomacyTrajectory` class is created. It initializes several key attributes that will be used throughout the lifecycle of an instance:

1. **Observations**: A list to store observations, which are likely instances of `utils.Observation`. Observations represent the state or information available at each step in the game.
2. **Legal Actions**: A list to store legal actions for each step. These are typically represented as NumPy arrays (`np.ndarray`), indicating valid moves or decisions that can be made by a player.
3. **Actions**: A list to store actual actions taken during each step of the trajectory. Similar to `legal_actions`, these are also stored as NumPy arrays.
4. **Step Outputs**: A list to store outputs from each step, which could include various information such as rewards or other metrics relevant to the game state.
5. **Returns**: An optional NumPy array that stores cumulative returns over a trajectory. This is set to `None` by default and can be populated later if needed.

The method initializes these attributes with empty lists, preparing them for use in subsequent methods of the class.

**Interactions with Other Components**
- The `__init__` method interacts with other parts of the project through its attributes, which are used in various game-related operations such as decision-making, state transitions, and reward calculation.
- It is part of a larger system where instances of `DiplomacyTrajectory` might be created for each episode or step in the game.

**Usage Notes**
- The method should always be called when creating an instance of `DiplomacyTrajectory`.
- Ensure that any required dependencies (like `utils.Observation` and NumPy) are properly imported.
- The attributes can be accessed and modified by other methods within the class to track and manage game states.

**Example Usage**
```python
import numpy as np

# Create an instance of DiplomacyTrajectory
trajectory = DiplomacyTrajectory()

# Simulate adding observations, actions, legal actions, step outputs, and returns
trajectory.observations.append(utils.Observation(...))  # Example observation
trajectory.actions.append(np.array([1, 2, 3]))          # Example action
trajectory.legal_actions.append(np.array([0, 1]))       # Example legal action
trajectory.step_outputs.append({"reward": 10})           # Example step output

# Accessing the returns attribute (which is None by default)
print(trajectory.returns)  # Output: None
```

This example demonstrates how to create an instance of `DiplomacyTrajectory` and populate its attributes with sample data, showcasing typical usage patterns for this class.
***
### FunctionDef append_step(self, observation, legal_actions, actions, step_outputs)
**Function Overview**
The `append_step` function appends observations, legal actions, player actions, and step outputs from a game turn into the trajectory of a Diplomacy game.

**Parameters**

1. **observation (utils.Observation)**: The current observation state of the game at a particular turn.
2. **legal_actions (np.ndarray)**: An array representing the legal actions available to each player in the current turn.
3. **actions (np.ndarray)**: An array representing the actual actions taken by players during the current turn.
4. **step_outputs (Dict[str, Any])**: A dictionary containing additional outputs or metadata from the game step.

**Return Values**
None

**Detailed Explanation**

The `append_step` function is responsible for recording each turn of a Diplomacy game in the trajectory object. Here’s how it works:

1. The function receives four parameters:
   - **observation**: This parameter holds the current state of the game, including board positions, player resources, and other relevant information.
   - **legal_actions**: An array where each element corresponds to the legal actions available for a specific player in the current turn.
   - **actions**: Another array representing the actual moves or orders made by players during this turn. This is often derived from the `padded_actions` variable which ensures that all players have an equal number of actions, even if some players did not take any action.
   - **step_outputs**: A dictionary containing additional information about the game state at this step, such as player returns or other metadata.

2. The function appends these parameters to the trajectory object:
   ```python
   self.append_step(observation,
                    padded_legal_actions,
                    padded_actions,
                    policies_step_outputs)
   ```

3. This method ensures that each turn of the game is recorded in a structured and consistent manner, allowing for easy analysis or replay of the game.

**Interactions with Other Components**

- The `append_step` function interacts with the trajectory object to maintain a record of all turns.
- It relies on the `observation`, `legal_actions`, and `actions` being correctly generated by the game logic before each turn.
- The `step_outputs` are typically generated by other parts of the game engine or policies that make decisions during the game.

**Usage Notes**

- Ensure that the `observation`, `legal_actions`, `actions`, and `step_outputs` are properly formatted and contain all necessary information for a complete record of the game turn.
- The function should be called after each turn in the game loop to maintain an accurate trajectory.
- Performance considerations: While this function is straightforward, it can become a bottleneck if the game state or actions are complex. Optimizing these data structures could improve performance.

**Example Usage**

Here’s an example of how `append_step` might be used within the context of the game loop:

```python
# Example usage in the game loop

for turn in range(num_turns):
    # Generate observations, legal_actions, and actions for this turn
    observation = generate_observation(turn)
    legal_actions = get_legal_actions(observation)
    actions = take_player_actions(legal_actions)  # Player decisions or AI logic
    
    # Record the turn in the trajectory
    traj.append_step(observation,
                     padded_legal_actions,
                     padded_actions,
                     policies_step_outputs)

# After all turns are recorded, terminate the trajectory with final returns
traj.terminate(final_returns)
```

In this example, `generate_observation`, `get_legal_actions`, and `take_player_actions` represent functions that would be defined elsewhere in the game logic to handle specific aspects of the turn. The `append_step` function is used to record each completed turn into the trajectory object.
***
### FunctionDef terminate(self, returns)
**Function Overview**
The `terminate` method in the `DiplomacyTrajectory` class finalizes a game trajectory by setting the returns attribute, which encapsulates the outcome of the game.

**Parameters**
- **returns**: A dictionary containing the return values for each player. This dictionary is set as an attribute of the `DiplomacyTrajectory` object to store the results of the game.

**Detailed Explanation**
The `terminate` method takes a single parameter, `returns`, which is expected to be a dictionary mapping each player's slot to their corresponding return value (e.g., points or other metrics). The method sets this `returns` parameter as an attribute of the `DiplomacyTrajectory` object. This step ensures that the trajectory object retains information about the game’s outcome, facilitating further analysis and storage.

**Interactions with Other Components**
- **DiplomacyTrajectory Class**: The `terminate` method is part of the `DiplomacyTrajectory` class, which manages the steps and outcomes of a game. It interacts directly with other methods in this class to append steps and finalize the trajectory.
- **traj.terminate(returns)**: This line within the `traj.terminate(returns)` call ensures that the `returns` dictionary is stored as an attribute of the `DiplomacyTrajectory` object, making it accessible for further processing or analysis.

**Usage Notes**
- The `terminate` method should be called after all steps in a game have been appended to ensure that the final outcome is recorded.
- It is crucial to provide accurate and complete return values for each player to ensure the trajectory reflects the correct game state.
- Performance implications are minimal as this method only involves setting an attribute, making it efficient.

**Example Usage**
```python
# Example of using terminate in a DiplomacyTrajectory object

from diplomacy_trajectory import DiplomacyTrajectory  # Assuming a hypothetical module

# Create a new trajectory instance
traj = DiplomacyTrajectory()

# Append steps to the trajectory (not shown here)
# ...

# Define returns for each player
returns = {
    0: 10,  # Player 0's return value
    1: 20,  # Player 1's return value
    2: 30   # Player 2's return value
}

# Terminate the trajectory with the returns
traj.terminate(returns)

print(traj.returns)  # Output: {0: 10, 1: 20, 2: 30}
```

This example demonstrates how to use the `terminate` method within a `DiplomacyTrajectory` object by setting the returns attribute with player-specific return values.
***
## FunctionDef _draw_returns(points_per_supply_centre, board, num_players)
**Function Overview**
The `_draw_returns` function computes the returns (number of supply centers) when a game ends in a draw. This function is used within the `run_game` method to determine how points are distributed among players based on their control over supply centers.

**Parameters**

- **points_per_supply_centre**: A boolean value indicating whether to assign points per supply center or not.
- **supply_centers**: An integer representing the number of supply centers controlled by a player. This parameter is passed as part of the game state and is used within the function.

**Return Values**
The function returns an integer representing the total points assigned to a player based on their control over supply centers, depending on the value of `points_per_supply_centre`.

**Detailed Explanation**

1. **Input Parameters**: The function takes two parameters: `points_per_supply_centre` and `supply_centers`.
2. **Boolean Logic**:
   - If `points_per_supply_centre` is `True`, the function returns the number of supply centers controlled by the player.
   - If `points_per_supply_centre` is `False`, the function returns 1, indicating a fixed point value regardless of the number of supply centers.

3. **Logic Flow**:
   - The function first checks the value of `points_per_supply_centre`.
   - Based on this value, it either returns the exact count of `supply_centers` or returns 1.
   
4. **Example Calculation**:
   - If `points_per_supply_centre = True` and `supply_centers = 5`, the function will return 5.
   - If `points_per_supply_centre = False` and `supply_centers = 3`, the function will return 1.

**Interactions with Other Components**

- The `_draw_returns` function is called within the `run_game` method to determine player returns when a game ends in a draw. It interacts directly with the game state, specifically the number of supply centers controlled by each player.
- The result from this function is used as part of the overall game results, which are then passed back to the `run_game` method for final termination and return.

**Usage Notes**

- **Preconditions**: Ensure that `points_per_supply_centre` is a boolean value (True or False) and `supply_centers` is an integer.
- **Performance Considerations**: The function is simple and does not involve complex operations, making it highly efficient. However, the performance impact of this function is negligible in the overall game execution.
- **Edge Cases**:
  - If `points_per_supply_centre` is neither True nor False, or if `supply_centers` is not an integer, the behavior is undefined and may lead to incorrect results.
  - If multiple players are involved, ensure that each player's returns are calculated correctly based on their supply centers.

**Example Usage**

```python
# Example where points_per_supply_centre = True and supply_centers = 5
returns = _draw_returns(True, 5)
print(returns)  # Output: 5

# Example where points_per_supply_centre = False and supply_centers = 3
returns = _draw_returns(False, 3)
print(returns)  # Output: 1
```

This example demonstrates how to use the `_draw_returns` function in a simple scenario.
## FunctionDef run_game(state, policies, slots_to_policies, max_length, min_years_forced_draw, forced_draw_probability, points_per_supply_centre, draw_if_slot_loses)
### Function Overview

The `calculateDiscount` function computes a discount amount based on the original price and the discount rate. It returns the discounted price as well as the calculated discount value.

### Parameters

1. **originalPrice**: A floating-point number representing the original price of an item or service.
2. **discountRate**: A floating-point number indicating the percentage discount to be applied, expressed as a decimal (e.g., 0.1 for 10%).

### Return Values

- **discountedPrice**: The final price after applying the discount rate to the original price.
- **discountAmount**: The amount of money saved due to the discount.

### Detailed Explanation

The `calculateDiscount` function performs the following steps:

1. **Input Validation**:
   - It first checks if both `originalPrice` and `discountRate` are non-negative numbers. If either is negative, it returns an error message indicating invalid input.
   
2. **Calculation of Discount Amount**:
   - The discount amount is calculated by multiplying the original price with the discount rate: `discountAmount = originalPrice * discountRate`.

3. **Calculation of Discounted Price**:
   - The discounted price is then computed by subtracting the discount amount from the original price: `discountedPrice = originalPrice - discountAmount`.
   
4. **Return Values**:
   - Finally, it returns both the `discountedPrice` and `discountAmount`.

### Interactions with Other Components

This function interacts primarily with other parts of a pricing system where discounts need to be applied. It can be called from various modules such as checkout processes or inventory management systems.

### Usage Notes

- **Preconditions**: Ensure that the input values are valid (non-negative numbers). Invalid inputs will result in an error message.
- **Performance Implications**: The function is simple and performs minimal calculations, making it efficient for use in real-time applications.
- **Security Considerations**: This function does not involve any security-sensitive operations. However, ensure that user-provided input values are validated to prevent potential issues.
- **Common Pitfalls**: Be cautious of applying negative discount rates or non-numeric inputs, as these can lead to incorrect calculations.

### Example Usage

```python
def calculateDiscount(originalPrice, discountRate):
    if originalPrice < 0 or discountRate < 0:
        return "Invalid input: Both price and rate must be non-negative."
    
    discountAmount = originalPrice * discountRate
    discountedPrice = originalPrice - discountAmount
    
    return {"discountedPrice": discountedPrice, "discountAmount": discountAmount}

# Example usage
result = calculateDiscount(100.0, 0.2)
print(result)  # Output: {'discountedPrice': 80.0, 'discountAmount': 20.0}
```

This example demonstrates how to use the `calculateDiscount` function with valid input values and shows the expected output format.
