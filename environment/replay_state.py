"""Replay-based DiplomacyState implementation for testing/integration.

This implementation replays a pre-recorded sequence of observations and legal
actions. It is intended to satisfy the DiplomacyState protocol so the rest of
the stack (policies, game runner, tests) can run end-to-end without a live
adjudicator. Actions passed to `step` are ignored; progression is controlled by
the provided trajectories.
"""

from typing import List, Sequence, Optional
import numpy as np

from diplomacy.environment import diplomacy_state
from diplomacy.environment import observation_utils as utils


class ReplayDiplomacyState(diplomacy_state.DiplomacyState):
  """A simple DiplomacyState that replays known observations/legal actions."""

  def __init__(
      self,
      observations: Sequence[utils.Observation],
      legal_actions: Sequence[Sequence[np.ndarray]],
      returns: Optional[np.ndarray] = None,
  ) -> None:
    if len(observations) != len(legal_actions):
      raise ValueError(
          "observations and legal_actions lengths must match: "
          f"{len(observations)} != {len(legal_actions)}")
    self._observations: List[utils.Observation] = list(observations)
    # Normalize legal actions to lists-of-lists of Python ints per step.
    self._legal_actions: List[List[List[int]]] = []
    for step_legals in legal_actions:
      step_legals_list: List[List[int]] = []
      for p in range(7):
        arr = np.asarray(step_legals[p]) if p < len(step_legals) else np.array([], dtype=np.int64)
        # Filter out padding zeros if present (0 is not a valid action in our encoding).
        step_legals_list.append([int(a) for a in arr if int(a) != 0])
      self._legal_actions.append(step_legals_list)

    self._cursor = 0
    self._terminal = False
    self._returns = (np.asarray(returns, dtype=np.float32)
                     if returns is not None else np.zeros((7,), dtype=np.float32))

  def is_terminal(self) -> bool:
    return self._terminal or self._cursor >= len(self._observations)

  def observation(self) -> utils.Observation:
    if self.is_terminal():
      # If terminal, return the last known observation.
      return self._observations[-1]
    return self._observations[self._cursor]

  def legal_actions(self) -> Sequence[Sequence[int]]:
    if self.is_terminal():
      return [[] for _ in range(7)]
    return self._legal_actions[self._cursor]

  def returns(self) -> np.ndarray:
    # Returns are zero during play; non-zero only after terminal.
    if self.is_terminal():
      return self._returns
    return np.zeros_like(self._returns)

  def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
    # This is a replay: ignore actions and just advance.
    del actions_per_player
    if self.is_terminal():
      return
    self._cursor += 1
    if self._cursor >= len(self._observations):
      self._terminal = True

