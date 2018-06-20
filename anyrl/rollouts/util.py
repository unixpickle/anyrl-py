"""
Utilities for manipulating rollouts.
"""


def reduce_states(state_batch, env_idx):
    """
    Reduce a batch of states to a batch of one state.
    """
    if state_batch is None:
        return None
    elif isinstance(state_batch, tuple):
        return tuple(reduce_states(s, env_idx) for s in state_batch)
    return state_batch[env_idx:env_idx + 1].copy()


def inject_state(state_batch, state, env_idx):
    """
    Replace the state at the given index with a new state.
    """
    if state_batch is None:
        return
    elif isinstance(state_batch, tuple):
        return tuple(inject_state(sb, s, env_idx)
                     for sb, s in zip(state_batch, state))
    state_batch[env_idx:env_idx + 1] = state


def reduce_model_outs(model_outs, env_idx):
    """
    Reduce a batch of model outputs to a batch of one
    model output.
    """
    out = dict()
    for key in model_outs:
        val = model_outs[key]
        if val is None:
            out[key] = None
        elif isinstance(val, tuple):
            out[key] = reduce_states(val, env_idx)
        else:
            out[key] = val[env_idx:env_idx + 1].copy()
    return out
