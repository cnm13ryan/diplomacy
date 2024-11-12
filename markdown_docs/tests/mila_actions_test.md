## ClassDef MilaActionsTest
---

**Function Overview**

The `MilaActionsTest` class is a test suite designed to validate the conversion and consistency between DeepMind (DM) actions and MILA (Machine Intelligence for Learning Adjudication) actions within a game environment. This ensures that actions can be accurately translated back and forth without losing information or introducing errors.

**Parameters**

- **None**: The `MilaActionsTest` class does not require any parameters as it is designed to operate on predefined lists of actions (`action_list.POSSIBLE_ACTIONS` and `action_list.MILA_ACTIONS_LIST`) and utilizes helper functions from the `mila_actions`, `human_readable_actions`, and `action_utils` modules.

**Return Values**

- **None**: The test methods within `MilaActionsTest` do not return any values. Instead, they use assertions to verify that conditions are met, and if an assertion fails, it raises a `TestCase.failureException`.

**Detailed Explanation**

The `MilaActionsTest` class contains several test methods, each focusing on different aspects of the DM-to-MILA action conversion process:

1. **Test Method 1: Inversion of Action Conversion**
   - **Purpose**: Ensures that converting an action from DM to MILA and back to DM results in the original action.
   - **Logic**:
     - Iterates over each action in `action_list.POSSIBLE_ACTIONS`.
     - Converts the action to MILA format using a hypothetical conversion function (not explicitly defined in the provided code).
     - Converts the MILA action back to DM format.
     - Asserts that the final DM action matches the original action.

2. **Test Method 2: Handling of Ambiguous Actions**
   - **Purpose**: Ensures that actions with multiple valid interpretations are correctly handled during conversion.
   - **Logic**:
     - Iterates over each ambiguous action in a hypothetical list (not explicitly defined).
     - Converts the action to MILA format.
     - Asserts that the resulting MILA action is one of the expected valid interpretations.

3. **Test Method 3: Validation of MILA Action List**
   - **Purpose**: Ensures that all actions in the MILA action list are well-formed and adhere to specific rules.
   - **Logic**:
     - Iterates over each action in `action_list.MILA_ACTIONS_LIST`.
     - Validates the action using a hypothetical validation function (not explicitly defined).
     - Asserts that the action is valid.

4. **Test Method 4: Counting of Illegal MILA Actions**
   - **Purpose**: Counts and categorizes actions from the DM list that are not included in the MILA list, providing insights into why these actions are excluded.
   - **Logic**:
     - Iterates over each action in `action_list.POSSIBLE_ACTIONS`.
     - Converts the action to MILA format.
     - If the resulting MILA action is not in `action_list.MILA_ACTIONS_LIST`, categorizes it based on its type (e.g., long convoy, other convoy).
     - Compares the counts of categorized actions against expected values.

**Usage Notes**

- **Limitations**: The test methods assume that the conversion and validation functions (`mila_actions.convert_to_mila`, `mila_actions.validate_action`) are correctly implemented. Any errors in these functions will affect the accuracy of the tests.
  
- **Edge Cases**: The test suite does not explicitly handle edge cases such as invalid input formats or unexpected action types. It is assumed that these cases are handled by the conversion and validation functions.

- **Performance Considerations**: The performance of the test suite may be impacted by the size of the action lists and the efficiency of the conversion and validation functions. For large lists, optimizations such as parallel processing or caching intermediate results could improve performance.

---

This documentation provides a comprehensive overview of the `MilaActionsTest` class, detailing its purpose, logic, and usage considerations based on the provided code snippet.
### FunctionDef test_inversion_dm_actions(self)
**Function Overview**: The `test_inversion_dm_actions` function tests whether converting a Domain Model (DM) action to MILA actions and back to DM actions recovers the original DM action.

**Parameters**: None. This function does not take any parameters.

**Return Values**: None. This function does not return any values; it performs assertions within its body.

**Detailed Explanation**:
- The function iterates over each possible action defined in `action_list.POSSIBLE_ACTIONS`.
- For each original DM action, it converts the action to a list of MILA actions using the `mila_actions.action_to_mila_actions` method.
- It then iterates over each MILA action obtained from the conversion and checks if the original DM action is included in the set of possible DM actions that can be derived from this MILA action using the `mila_actions.mila_action_to_possible_actions` method.
- If the original DM action is not found in the set of possible DM actions derived from any MILA action, an assertion error is raised with a message indicating which MILA action does not map back to a set including the original DM action.

**Usage Notes**:
- This test ensures that the conversion between DM actions and MILA actions is reversible and accurate.
- The function assumes that `action_list.POSSIBLE_ACTIONS`, `mila_actions.action_to_mila_actions`, and `mila_actions.mila_action_to_possible_actions` are correctly implemented and available in the scope where this function is called.
- The test may be resource-intensive if `action_list.POSSIBLE_ACTIONS` contains a large number of actions, as it performs multiple conversions and checks for each action.
- Edge cases, such as actions that cannot be converted to MILA actions or actions that map back to multiple DM actions, should be handled appropriately within the implementations of `mila_actions.action_to_mila_actions` and `mila_actions.mila_action_to_possible_actions`.
***
### FunctionDef test_inversion_mila_actions(self)
**Function Overview**

The function `test_inversion_mila_actions` is designed to verify that converting a MILA action to its possible DM (Decision Making) actions and then back to MILA actions recovers the original MILA action.

**Parameters**

- **self**: The instance of the class `MilaActionsTest`.

**Return Values**

- None. This function does not return any value; it asserts conditions that ensure the correctness of the action conversion process.

**Detailed Explanation**

The function `test_inversion_mila_actions` is a unit test method within the `MilaActionsTest` class. Its primary purpose is to validate the integrity and accuracy of the action conversion mechanisms implemented in the `mila_actions` module. The logic of this function can be broken down into the following steps:

1. **Iteration Over MILA Actions**: The function iterates over each action defined in `action_list.MILA_ACTIONS_LIST`. This list presumably contains a comprehensive set of MILA actions that need to be tested.

2. **Conversion to Possible DM Actions**: For each MILA action (`original_action`), the function calls `mila_actions.mila_action_to_possible_actions(original_action)`. This method is expected to return a list of possible DM actions that can result from the given MILA action.

3. **Verification of Inversion**: The function then iterates over each DM action in the list of possible DM actions (`possible_dm_actions`). For each DM action, it calls `mila_actions.action_to_mila_actions(dm_action)`, which is expected to convert the DM action back into a set of MILA actions.

4. **Assertion Check**: The function asserts that the original MILA action (`original_action`) is included in the set of MILA actions returned by `action_to_mila_actions`. This assertion ensures that the conversion from MILA to DM and back to MILA is reversible and accurately recovers the original MILA action.

5. **Error Reporting**: If the assertion fails, an error message is generated using `human_readable_actions.action_string(dm_action, None)`, which provides a human-readable representation of the DM action that failed the test. This helps in identifying which specific DM action did not map back to the expected set of MILA actions.

**Usage Notes**

- **Assumptions**: The function assumes that:
  - `action_list.MILA_ACTIONS_LIST` contains all relevant MILA actions.
  - `mila_actions.mila_action_to_possible_actions` correctly generates a list of possible DM actions for each MILA action.
  - `mila_actions.action_to_mila_actions` accurately converts DM actions back to MILA actions.

- **Edge Cases**: The function does not explicitly handle edge cases such as:
  - Empty lists of possible DM actions.
  - DM actions that do not map back to any MILA actions.

- **Performance Considerations**:
  - The performance of this test is dependent on the size of `action_list.MILA_ACTIONS_LIST` and the complexity of the conversion methods.
  - For large lists or complex conversions, the test may take a significant amount of time to complete.
***
### FunctionDef test_all_mila_actions_have_dm_action(self)
**Function Overview**:  
The function `test_all_mila_actions_have_dm_action` is designed to verify that every action listed in `MILA_ACTIONS_LIST` has at least one corresponding DM (Decision Making) action.

**Parameters**:  
- None. The function does not take any parameters.

**Return Values**:  
- The function does not return any values. It raises an assertion error if any condition fails.

**Detailed Explanation**:  
The function iterates over each `mila_action` in the predefined list `MILA_ACTIONS_LIST`. For each `mila_action`, it retrieves a list of possible DM actions using the function `mila_actions.mila_action_to_possible_actions(mila_action)`. The function then asserts that this list is not empty. If any `mila_action` does not have associated DM actions, an assertion error is raised with a message indicating which `mila_action` lacks DM actions.

**Usage Notes**:  
- This test assumes the existence and correct implementation of `MILA_ACTIONS_LIST` and the function `mila_actions.mila_action_to_possible_actions`.
- The test will fail if any action in `MILA_ACTIONS_LIST` does not have at least one DM action associated with it.
- Performance considerations: The efficiency of this test is dependent on the size of `MILA_ACTIONS_LIST` and the speed of the function `mila_action_to_possible_actions`. If either is large or slow, the test execution time may increase.
***
### FunctionDef test_only_disband_remove_ambiguous_mila_actions(self)
**Function Overview**

The function `test_only_disband_remove_ambiguous_mila_actions` is designed to test that any ambiguous MILA actions result exclusively in either a "disband" or "remove" action. It iterates through a list of MILA actions and checks if the corresponding possible actions (DM actions) are limited to these two specific orders.

**Parameters**

- **None**: The function does not take any parameters.

**Return Values**

- **None**: The function does not return any values; it asserts conditions that should hold true for the input data.

**Detailed Explanation**

The function `test_only_disband_remove_ambiguous_mila_actions` performs the following steps:

1. **Iteration Over MILA Actions**: It iterates over each MILA action in the predefined list `action_list.MILA_ACTIONS_LIST`.

2. **Convert MILA Action to DM Actions**: For each MILA action, it converts it into a set of possible DM actions using the function `mila_actions.mila_action_to_possible_actions(mila_action)`.

3. **Check for Ambiguity**: It checks if the number of possible DM actions (`dm_actions`) is greater than one. If so, it asserts that there should be exactly two possible DM actions, as indicated by `self.assertLen(dm_actions, 2, f'{mila_action} gives >2 dm_actions')`.

4. **Extract Orders**: It extracts the orders from each of these DM actions using a set comprehension and the function `action_utils.action_breakdown(dm_action)[0]`. This step ensures that all possible orders are captured.

5. **Verify Orders**: Finally, it asserts that the set of extracted orders should only contain `action_utils.REMOVE` and `action_utils.DISBAND`, as indicated by `self.assertEqual(orders, {action_utils.REMOVE, action_utils.DISBAND}, f'{mila_action} ambiguous but not a disband/remove action')`.

**Usage Notes**

- **Assumptions**: The function assumes that the MILA actions in `action_list.MILA_ACTIONS_LIST` are valid and that the conversion to DM actions is correctly implemented.
  
- **Edge Cases**: If a MILA action does not result in any possible DM actions, it will not be tested by this function. Additionally, if there are more than two possible orders for an ambiguous MILA action, the test will fail.

- **Performance Considerations**: The performance of this function is dependent on the size of `action_list.MILA_ACTIONS_LIST` and the complexity of converting each MILA action to DM actions. For large lists or complex conversions, the function may take a significant amount of time to execute.
***
### FunctionDef test_all_dm_actions_have_possible_mila_action_count(self)
**Function Overview**

The `test_all_dm_actions_have_possible_mila_action_count` function is designed to verify that each possible DM (Diplomatic Move) action corresponds to a valid number of MILA actions. This ensures consistency and correctness in how DM actions are translated into MILA actions within the application.

**Parameters**

- **action**: The function iterates over `action_list.POSSIBLE_ACTIONS`, which is a predefined list of all possible DM actions.
  
**Return Values**

- **None**: The function does not return any value. It asserts conditions and raises an error if any condition fails.

**Detailed Explanation**

The function's primary purpose is to validate the mapping between DM actions and MILA actions. Here’s how it works:

1. **Iteration Over Actions**: The function iterates over each action in `action_list.POSSIBLE_ACTIONS`.
2. **Mapping to MILA Actions**: For each action, it uses the `mila_actions.action_to_mila_actions(action)` method to determine the corresponding MILA actions.
3. **Validation of MILA Action Count**: It checks if the number of generated MILA actions (`len(mila_actions_list)`) falls within a predefined set of valid counts: `{1, 2, 3, 4, 6}`.
4. **Assertion and Error Handling**: If the count is not within the valid range, it raises an assertion error with a message indicating which action failed validation.

The rationale behind this function is that DM actions can be ambiguous in terms of specifying unit types or coastlines when these details can be inferred from the game board state. Consequently, each DM action can correspond to multiple MILA actions, and the number of possible MILA actions depends on the specifics of the DM action (e.g., whether it involves an army, a fleet, or fleets in bicoastal provinces).

**Usage Notes**

- **Limitations**: The function assumes that `action_list.POSSIBLE_ACTIONS` is correctly defined and contains all valid DM actions.
- **Edge Cases**: The function does not handle cases where the mapping from DM to MILA actions might be ambiguous or undefined. It relies on the correctness of the `mila_actions.action_to_mila_actions` method.
- **Performance Considerations**: The performance of this test is directly related to the number of actions in `action_list.POSSIBLE_ACTIONS`. If this list is large, the function may take a significant amount of time to execute.
***
### FunctionDef test_expected_number_missing_mila_actions(self)
**Function Overview**

The function `test_expected_number_missing_mila_actions` tests that MILA actions do not include any unexpected actions except those known to be related to long convoys.

**Parameters**

- The function does not take any parameters directly. It relies on global variables and modules imported from the project structure, such as `action_list`, `mila_actions`, and `action_utils`.

**Return Values**

- The function does not return any values. Instead, it asserts that the counts of illegal MILA actions match expected values using `self.assertEqual`. If the assertion fails, it raises an AssertionError with a message indicating an unexpected number of actions not in the MILA list.

**Detailed Explanation**

1. **Initialization**:
   - A defaultdict named `mila_actions_to_dm_actions` is created to map MILA actions to their corresponding DeepMind actions.
   - A set named `long_convoys` is initialized to track long convoy routes.

2. **Iterating Over Possible Actions**:
   - The function iterates over each action in `action_list.POSSIBLE_ACTIONS`.
   - For each action, it converts the action into a list of MILA actions using `mila_actions.action_to_mila_actions(action)`.
   - Each MILA action is added to the `mila_actions_to_dm_actions` dictionary with its corresponding DeepMind action.
   - If a MILA action is not in `action_list.MILA_ACTIONS_LIST`, it checks if the order of the action is `CONVOY_TO`. If so, it adds the convoy route `(p1, p2)` to the `long_convoys` set.

3. **Categorizing Illegal MILA Actions**:
   - A dictionary named `reasons_for_illegal_mila_action` is initialized to count various categories of illegal MILA actions.
   - The function iterates over each MILA action in `mila_actions_to_dm_actions`.
   - For each MILA action not in `action_list.MILA_ACTIONS_LIST`, it categorizes the reason for its illegality based on the order of the DeepMind action:
     - **Long convoy to**: Incremented if the order is `CONVOY_TO`.
     - **Long convoy**: Incremented if the order is `CONVOY` and the route `(p3, p2)` is in `long_convoys`.
     - **Other convoy**: Incremented if the order is `CONVOY` and the route `(p3, p2)` is not in `long_convoys`.
     - **Support long convoy to**: Incremented if the order is `SUPPORT_MOVE_TO` and the route `(p3, p2)` is in `long_convoys`.
     - **Support alternative convoy too long**: Incremented if the order is `SUPPORT_MOVE_TO` and the route `(p3, p2)` is not in `long_convoys`.
     - **Unknown**: Incremented for any other orders.

4. **Assertion**:
   - The function asserts that the counts of illegal MILA actions match the expected values defined in `expected_counts`. If they do not match, it raises an AssertionError with a message indicating an unexpected number of actions not in the MILA list.

**Usage Notes**

- This test assumes that all possible actions and their corresponding MILA actions are correctly defined in the project's action lists.
- The function relies on manual checks for certain categories of illegal MILA actions (e.g., long convoys, other convoys) to ensure they are well-formatted and relevant.
- Performance considerations: The function iterates over all possible actions and their corresponding MILA actions, which can be computationally expensive if the number of actions is large. However, this is a one-time operation during testing and should not significantly impact performance in typical use cases.
***
