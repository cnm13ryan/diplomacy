"""Concrete ObservationTest implementation using a replay state.

To use this test, place the reference files in a directory and set the
environment variable DIPLOMACY_DATA_DIR to that directory, or adjust the
DEFAULT_* paths below. Alternatively, keep DIPLOMACY_*_PATH env vars to point
to each artifact individually.
"""

import os
from typing import Any, Dict, Sequence, Tuple

import dill
import jax
import numpy as np
import tree

from absl.testing import absltest

from diplomacy.environment import observation_utils as utils
from diplomacy.environment import diplomacy_state
from diplomacy.environment import game_runner
from diplomacy.environment import observation_transformation
from diplomacy.network import config as net_config
from diplomacy.network import parameter_provider as provider_mod
from diplomacy.tests import observation_test as base_test
from diplomacy.environment.replay_state import ReplayDiplomacyState


def _data_path(name: str) -> str:
  base = os.environ.get("DIPLOMACY_DATA_DIR", "./data")
  return os.environ.get(f"DIPLOMACY_{name.upper()}_PATH", os.path.join(base, name))


def _maybe_load_dill(path: str):
  # Try dill first; fall back to numpy npz with allow_pickle if needed.
  with open(path, "rb") as f:
    try:
      return dill.load(f)
    except Exception:
      pass
  # Numpy fallback
  arr = np.load(path, allow_pickle=True)
  # np.load may return an NpzFile mapping; try to pull the first array/list-like
  if isinstance(arr, np.lib.npyio.NpzFile):
    # Heuristic: grab the first entry
    keys = list(arr.keys())
    if not keys:
      raise ValueError(f"Empty npz at {path}")
    return arr[keys[0]].tolist()
  return arr.tolist()


class _RandomInitParameterProvider:
  """ParameterProvider that initialises network params randomly on demand."""

  def __init__(self):
    cfg = net_config.get_config()
    self._net_cls = cfg.network_class
    self._kwargs = dict(cfg.network_kwargs)
    self._rng = jax.random.PRNGKey(0)
    # Prepare params/state via the network convenience method.
    self._rng, sub = jax.random.split(self._rng)
    params, net_state = self._net_cls.initial_inference_params_and_state(
        self._kwargs, sub, num_players=7)
    self._params = params
    self._net_state = net_state
    self._step = np.array(0, dtype=np.int64)

  def params_for_actor(self):
    return self._params, self._net_state, self._step


class UserObservationTest(base_test.ObservationTest):

  def get_diplomacy_state(self) -> diplomacy_state.DiplomacyState:
    # Load reference observations and legal actions, and replay them.
    obs_list = [base_test.construct_observations(o)
                for o in self.get_reference_observations()]
    legals = self.get_reference_legal_actions()
    # No specific returns are required; default zeros suffice.
    return ReplayDiplomacyState(obs_list, legals, returns=None)

  def get_parameter_provider(self) -> provider_mod.ParameterProvider:
    # If a pre-trained params file exists, use it; otherwise fall back to a
    # random-init provider so tests can at least execute.
    params_path = _data_path("sl_params.npz")
    if os.path.exists(params_path):
      with open(params_path, "rb") as f:
        return provider_mod.ParameterProvider(f)
    # Fallback: random init
    return _RandomInitParameterProvider()  # type: ignore[return-value]

  def get_reference_observations(self) -> Sequence[Dict[str, Any]]:
    path = _data_path("observations.npz")
    if not os.path.exists(path):
      raise FileNotFoundError(
          f"Missing observations at {path}. Set DIPLOMACY_DATA_DIR or DIPLOMACY_OBSERVATIONS_PATH.")
    return _maybe_load_dill(path)

  def get_reference_legal_actions(self) -> Sequence[np.ndarray]:
    path = _data_path("legal_actions.npz")
    if not os.path.exists(path):
      raise FileNotFoundError(
          f"Missing legal_actions at {path}. Set DIPLOMACY_DATA_DIR or DIPLOMACY_LEGAL_ACTIONS_PATH.")
    return _maybe_load_dill(path)

  def get_reference_step_outputs(self) -> Sequence[Dict[str, Any]]:
    path = _data_path("step_outputs.npz")
    if not os.path.exists(path):
      raise FileNotFoundError(
          f"Missing step_outputs at {path}. Set DIPLOMACY_DATA_DIR or DIPLOMACY_STEP_OUTPUTS_PATH.")
    return _maybe_load_dill(path)

  def get_actions_outputs(
      self) -> Sequence[Tuple[Sequence[Sequence[int]], Any]]:
    path = _data_path("actions_outputs.npz")
    if not os.path.exists(path):
      raise FileNotFoundError(
          f"Missing actions_outputs at {path}. Set DIPLOMACY_DATA_DIR or DIPLOMACY_ACTIONS_OUTPUTS_PATH.")
    return _maybe_load_dill(path)


if __name__ == "__main__":
  absltest.main()

