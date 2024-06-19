import numpy as np

# Linear Increase Schedule
# Progress will linearly increase from 1 (beginning) to 0 (end).
def sched_linear_inc(initial_value, final_value, total_steps):
    def func(progress_remaining):
        current_step = total_steps * (1 - progress_remaining)
        return initial_value + (final_value - initial_value) * current_step / total_steps
    return func

# Exponential Increase Schedule
# Progress will exponentially increase from 1 (beginning) to 0 (end).
def sched_exp_inc(initial_value, final_value, total_steps):
    growth_rate = (final_value / initial_value) ** (1 / total_steps)
    def func(progress_remaining):
        current_step = total_steps * (1 - progress_remaining)
        return initial_value * (growth_rate ** current_step)
    return func

# Exponential Decrease Schedule
# Progress will exponentially decrease from 1 (beginning) to 0 (end).
def sched_exp_dec(initial_value, final_value, total_steps):
    decay_rate = (final_value / initial_value) ** (1 / total_steps)
    def func(progress_remaining):
        current_step = total_steps * (1 - progress_remaining)
        return initial_value * (decay_rate ** current_step)
    return func

# Logarithmic Decrease Schedule
# Progress will logarithmically decrease from 1 (beginning) to 0 (end).
def sched_log_dec(initial_value, final_value, total_steps):
    def func(progress_remaining):
        current_step = total_steps * (1 - progress_remaining)
        if current_step == 0:
            return initial_value
        return initial_value + (final_value - initial_value) * np.log1p(current_step) / np.log1p(total_steps)
    return func
