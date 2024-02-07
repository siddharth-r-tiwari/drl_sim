## init

Initialize the experiment parameters and create the stimulus and reward matrices.

Parameters:
tsk (str): The type of task.
n_trials (int): The number of trials.
t_int (float): The total duration of each trial.
delta (float): The time step size.
types (int): The number of stimulus types.
t_stim (torch.Tensor): The time points at which each stimulus type is presented.
t_rew (torch.Tensor): The time points at which each reward is delivered.
s_rew (torch.Tensor): The size of each reward.
uncertainty (float): The level of uncertainty in the reward delivery.

Returns:
T (torch.Tensor): The time points of the simulation.
x (torch.Tensor): The stimulus matrix.
r (torch.Tensor): The reward matrix.

## sim

Simulates a reinforcement learning task using the given parameters.

Parameters:
T (int): The number of time steps in the simulation.
x (torch.Tensor): The input data tensor of shape (types, steps, T).
r (torch.Tensor): The reward tensor of shape (types, n_trials, T).
tsk (float): The task probabilities.
num_predictors (int): The number of predictors.
a (float or torch.Tensor): The alpha parameter(s) for learning rate.
g (float or torch.Tensor): The gamma parameter(s) for discount factor.
l (float or torch.Tensor): The lambda parameter(s) for eligibility trace.
fname (str, optional): The file name to save the simulation results. Defaults to "none".

Returns:
dict: A dictionary containing the experiment results/details including:
    - 'V' (torch.Tensor): The value estimates tensor of shape (num_predictors, types, n_trials, steps).
    - 'deltas' (torch.Tensor): The deltas tensor of shape (num_predictors, types, n_trials, steps - 1).
    - 'time_steps' (int): The number of time steps in the simulation.
    - 'rewards' (torch.Tensor): The reward tensor of shape (types, n_trials, T).
    - 'states' (list): The list of generated states during the simulation.
    - 'alpha' (torch.Tensor): The alpha parameter tensor of shape (num_predictors, 2).
    - 'gamma' (torch.Tensor): The gamma parameter tensor of shape (num_predictors, 2).
    - 'lambda' (torch.Tensor): The lambda parameter tensor of shape (num_predictors, 2).

## val_at_t

Calculate and visualize the values at a specific time point for a given state in an experiment.

Parameters:
    exp (dict): Experiment data containing value estimates, reward, etc.
    state (int): The state within the experiment trials to consider.
    dt (float): The time step used in the experiment.
    time (float): The specific time point (within the interval) to visualize.
    diversify (list, optional): List of diversification parameters. Defaults to ['none'].
    trials (list, optional): List of trial indices to consider. Defaults to ['none'].
    fname (list, optional): List containing the filename to save the plot. Defaults to ['none'].

## heatmap

Generate a heatmap plot of the given experiment data.

Parameters:
- exp (dict): Experiment data containing value estimates, reward, etc.
- state (int): The state within the experiment trials to consider.
- prs (list): The list of indices to iterate over for plotting.
- diversify (list, optional): List of diversification parameters. Defaults to ['none'].
- fname (list, optional): Filename for saving the plot. Default is ["none"].

## val_over_t

Plot the value, reward, and TD error over time for a given experiment and state.

Parameters:
- e (dict): Experiment data containing value estimates, reward, etc.
- st (int): The state within the experiment trials to consider.
- dt (float): The time step for the x-axis. (NOTE: Not same usage as in other functions, above)
- fname (list, optional): The filename to save the animation. Default is ["none"].
- diversify (list, optional): The diversification parameters. Default is ["none"].
- trials (list, optional): List of diversification parameters. Defaults to ['none'].

Returns:
animation: The animation object showing the plots of value, reward, and TD error over time.

## step

Converts a continuous time value to a discrete time step.

Args:
    t (float): The continuous time value.

Returns:
    int: The corresponding discrete time step.

## gen_state

Generates a state based on the given probabilities.

Parameters:
ps (numpy.ndarray): An array of probabilities.

Returns:
int: The generated state.

## t_unc

Return a random time step (given uncertainty).

Parameters:
- step (int): The step for which to calculate the time uncertainty.

Returns:
- int: Random time step (within bounds of "uncertainty").

## task_probs

Returns the probability distribution of different states for a given task.

Parameters:
tsk (str): The task for which the probability distribution is required.

Returns:
list: The probability distribution of states for the given task.

## params

Return list of parameter values for all value predictors from inputted bound, list, or float.

Args:
    bound: The bound for generating parameters. It can be a tuple (for random values within that bound), list (discrete), or float (all predictors have the same value).

Returns:
    torch.Tensor or float: Tensor or float of parameter values.
    bool: True if the generated parameters are a tensor, False otherwise.

## ret_apos

Calculate the positive learning rate (for alpha parameter generation).

Parameters:
sc (float): Scaling factor
aneg (float): Pessimistic learning rate.

Returns:
float: Optimistic learning rate.

## compare_floats

Compare two floating-point numbers with a given error tolerance.

Args:
    float1 (float): The first floating-point number.
    float2 (float): The second floating-point number.
    error (float): The maximum allowable difference between the two numbers.

Returns:
    bool: True if the absolute difference between the two numbers is less than or equal to the error, False otherwise.
