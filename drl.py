############## IMPORTS ##############

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import torch
import numpy as np


############## SIMULATION FUNCTIONS ##############

def init(tsk, n_trials, t_int, delta, types, t_stim,
                   t_rew, s_rew, uncertainty):
    """
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
    """
    global dt, unc
    dt = delta
    unc = uncertainty

    #time points
    steps = step(t_int) + 1
    T = torch.arange(0, t_int + dt, dt)

    #stimulus
    x = torch.zeros(types, steps, steps)
    for i in range(types):
        x[i, step(t_stim[i].item())-1:, step(t_stim[i].item())-1:] = torch.eye(steps - step(t_stim[i].item()) + 1)

    #size of reward (EDIT!!)
    r = torch.zeros(types, n_trials, steps)
    for i in range(types):
        for n in range(n_trials):
            if tsk == 'hyp_unif':
                r[i, n, t_unc(step(t_rew[i].item()) - 1)] = s_rew[i].item()
            elif tsk == 'hyp_var':
                r[i, n, t_unc(step(t_rew[i].item()) - 1)] = torch.normal(s_rew[i, 0],
                        s_rew[i, 1])
            elif tsk == 'var':
                r[i, n, t_unc(step(t_rew[i].item()) - 1)] = s_rew[gen_state(s_rew[:,
                        1]), 0]
            elif tsk == 'delay':
                if i == 4:
                    rew = [0, 3]
                    r[i, n, t_unc(step(2.3) - 1)] = torch.tensor(rew[gen_state([0.5,0.5])])
                    r[i, n, t_unc(step(5.45) - 1)] = torch.tensor(rew[gen_state([0.5,0.5])])
                elif i == 5:
                    for t in np.arange(1.25, 7, 1.05):
                        r[i, n, t_unc(step(t) - 1)] = torch.tensor(rew[gen_state([5/6, 1/6])])
                else:
                    r[i, n, t_unc(step(t_rew[i].item()) - 1)] = s_rew[i]
            #elif tsk == 'NEW' (EDIT HERE!!)
               # Add new rewards at time steps, as above!! 
            
    return T, x, r

def sim(T, x, r, tsk, num_predictors, a, g, l, fname=["none"]):
    """
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

    """
    
    global n_predictors
    n_predictors = num_predictors

    # Assigning parameters
    alpha_sc, a_t = params(a)
    gamma, g_t = params(g)
    lmbda, l_t = params(l)

    if a_t:
        alpha = torch.rand(n_predictors, 2)
        for i in range(alpha.shape[0]):
            alpha[i, 0] = ret_apos(alpha_sc[i].item(), alpha[i, 1].item())
            while alpha[i, 0].item() > 1:
                alpha[i, 1] = random.random()
                alpha[i ,0] = ret_apos(alpha_sc[i].item(), alpha[i, 1].item())
    else:
        alpha = alpha_sc

    types = len(x[:,0,0])
    steps = len(x[0,:,0])
    n_trials = len(r[0,:,0])

    #weights
    w = torch.zeros(n_predictors, types, steps)

    #value
    V = torch.zeros(n_predictors, types, n_trials, steps)

    #deltas
    deltas = torch.zeros(n_predictors, types, n_trials, steps - 1)

    #eligibility trace
    el = torch.zeros(n_predictors, types, steps, steps - 1)
    for i in range(n_predictors):
        for s in range(types):
            for t in range(steps - 1):
                if l_t:
                    el[i, s, :, t] = pow(lmbda[i], dt) * el[i, s, :, t-1] + x[s, :, t]
                else:
                    el[i, s, :, t] = pow(lmbda, dt) * el[i, s, :, t-1] + x[s, :, t]


    states = []

    for n in range(n_trials):
        #update `task_probs` for needs of your experiment!
        state = gen_state(task_probs(tsk))
        for i in range(n_predictors):
            V[i, state, n, :] = torch.matmul(w[i, state, :], x[state, :, :])
            if g_t:
                d = r[state, n, :-1] + (pow(gamma[i].item(), dt) * V[i, state, n, 1:]) - V[i, state, n, :-1]
            else:
                d = r[state, n, :-1] + (pow(gamma, dt) * V[i, state, n, 1:]) - V[i, state, n, :-1]

            for t in range(steps - 1):
                if a_t:
                    aoc = torch.sum(el[i, state, :, t])
                    rpe_sgn = alpha_c(True, d[t], 0)
                    if aoc.item() < 1:
                        dw = (pow(alpha[i, rpe_sgn], dt) * d[t]) * el[i, state, :,t]
                    else:
                        dw = (pow(alpha[i, rpe_sgn], dt) * d[t] / aoc.item()) * el[i, state, :,t]
                else:
                    aoc = torch.sum(el[i, state, :, t])
                    if aoc.item() < 1:
                        dw = (pow(alpha, dt) * d[t]) * el[i, state, :,t]
                    else:
                        dw = (pow(alpha, dt) * d[t] / aoc.item()) * el[i, state, :,t]
                w[i, state, :] = w[i, state, :] + dw
            
            deltas[i, state, n, :] = d

        states.append(state)

    exp = {'V' : V, 'deltas' : deltas, 'T' : T, 'r': r, 'states' : states, 'alpha' : alpha, 'gamma' : gamma, 'lmbda' : lmbda}
    if fname != "none":
        torch.save(exp, f'{fname[0]}.pth')
    return exp


############## VISUALIZATION FUNCTIONS ##############

def val_at_t(exp,
    state,
    dt,
    time,
    diversify=['none'],
    trials=['none'],
    fname=['none']
    ):
    """
    Calculate and visualize the values at a specific time point for a given state in an experiment.

    Parameters:
        exp (dict): Experiment data containing value estimates, reward, etc.
        state (int): The state within the experiment trials to consider.
        dt (float): The time step used in the experiment.
        time (float): The specific time point (within the interval) to visualize.
        diversify (list, optional): List of diversification parameters. Defaults to ['none'].
        trials (list, optional): List of trial indices to consider. Defaults to ['none'].
        fname (list, optional): List containing the filename to save the plot. Defaults to ['none'].
    """
    t = step(time) - 1
    deg_free = len(diversify)

    predictors = []
    tls = [index for (index, value) in enumerate(exp['states']) if value
           == state]

    for i in range(len(exp['V'][:,0,0,0])):
        value = []
        if trials[0] == 'none':
            n_trials = len(tls)
            for s in range(len(tls)):
                value.append(exp['V'][i, state, tls[s], t].item())
        else:
            n_trials = trials[1] - trials[0]
            for s in range(trials[0],trials[1]):
                value.append(exp['V'][i, state, tls[s], t].item())
        predictors.append(value)

    if deg_free == 2:
        (fig, ax) = plt.subplots()

        norm = plt.Normalize(vmin=diversify[1][0], vmax=diversify[1][1])
        for i in range(len(predictors)):
            if diversify[0] == 'alpha':
                normalized_value = exp['alpha'][i, 0].item() / (exp['alpha'][i, 0].item() + exp['alpha'][i, 1].item())
            elif diversify[0] == 'gamma':
                normalized_value = exp['gamma'][i].item()
            elif diversify[0] == 'lmbda':
                normalized_value = exp['lmbda'][i].item()
            else:
                normalized_value = 0  # Default value if diversify[0] is not recognized

            plt.plot(predictors[i], color=plt.cm.jet(normalized_value))

        # Create a normalized color map using Normalize class
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label('Color Range')

        ax.set_xlabel('Number of trials')
        ax.set_ylabel('Value')

        n_pred = len(exp['V'][:,0,0,0])
        ax.set_title(f'{diversify[0]} (n={n_pred}) at t={time}, {n_trials} Trials')

        ax.grid(True)

        plt.legend()
        plt.show()
    
    elif deg_free == 4:
        scs = diversify[3]
        print(scs)
        n_pred = len(exp['V'][:,0,0,0])
        fig, axs = plt.subplots(1, len(scs), figsize=(15, 5))

        for i, sc in enumerate(scs):
            predictors_sc = []
             # Filter predictors based on the current gamma value

            if diversify[2] == 'alpha':
                predictors_sc = [(j, predictors[j]) for j in range(len(predictors)) if compare_floats(round(exp['alpha'][j, 0].item() / (exp['alpha'][j, 0].item() + exp['alpha'][j, 1].item()), 2), sc, 0.01)]
            elif diversify[2] == 'gamma':
                predictors_sc = [(j, predictors[j]) for j in range(len(predictors)) if compare_floats(exp['gamma'][j].item(), sc, 0.01)]
            elif diversify[2] == 'lmbda':
                predictors_sc = [(j, predictors[j]) for j in range(len(predictors)) if compare_floats(exp['lmbda'][j].item(), sc, 0.01)]

            # Plot the predictors on the corresponding subplot
            axs[i].set_xlabel('Number of trials')
            axs[i].set_ylabel('Value')
            axs[i].set_title(f'{diversify[0]} and {diversify[2]} = {sc} (n={len(predictors_sc)}) at t={time}, {n_trials} Trials')
            axs[i].grid(True)
           
            for j, predictor in enumerate(predictors_sc):
                if diversify[0] == 'alpha':
                    val = exp['alpha'][predictor[0], 0].item() / (exp['alpha'][predictor[0], 0].item() + exp['alpha'][predictor[0], 1].item())
                elif diversify[0] == 'gamma':
                    val = exp['gamma'][predictor[0]].item()
                elif diversify[0] == 'lmbda':
                    val = exp['lmbda'][predictor[0]].item()
                color = plt.cm.jet(val)
                axs[i].plot(predictor[1], color=color)

            # Create a colorbar for the current subplot
            norm = plt.Normalize(vmin=diversify[1][0], vmax=diversify[1][1])
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[i])
            if diversify[0] == 'alpha':
                cbar.set_label('Alphas')
            elif diversify[0] == 'gamma':
                cbar.set_label('Gammas')
            elif diversify[0] == 'lmbda':
                cbar.set_label('Lambdas')
           
        plt.tight_layout()
        plt.show()

    else:
        (fig, ax) = plt.subplots()
        normalized_value = 0  # Default value if diversify[0] is not recognized
        norm = plt.Normalize(vmin=0, vmax=1)
        
        for i in range(len(predictors)):
            plt.plot(predictors[i], color=plt.cm.jet(normalized_value))
        
        ax.set_xlabel('Number of trials')
        ax.set_ylabel('Value')

        n_pred = len(exp['V'][:,0,0,0])
        n_st = len(exp['V'][0,0,0,:])
        ax.set_title(f'No distribute (n={n_pred}) at t={time}, {n_trials} Trials')

        ax.grid(True)

        plt.legend()
        plt.show()

    if fname != "none":
        plt.savefig(f'{fname[0]}.png')

def heatmap(exp, state, prs, diversify=['none'], fname=["none"]):
    """
    Generate a heatmap plot of the given experiment data.

    Parameters:
    - exp (dict): Experiment data containing value estimates, reward, etc.
    - state (int): The state within the experiment trials to consider.
    - prs (list): The list of indices to iterate over for plotting.
    - diversify (list, optional): List of diversification parameters. Defaults to ['none'].
    - fname (list, optional): Filename for saving the plot. Default is ["none"].
    """
    deg_free = len(diversify)

    tls = [index for (index, value) in enumerate(exp['states']) if value == state]
    for i in prs:
        td_i = exp['deltas'][i, state, tls, :]
        n = td_i.shape[0]

        if deg_free == 2:
            if (diversify[0] == 'alpha' or diversify[1] == 'alpha') and (diversify[0] == 'gamma' or diversify[1] == 'gamma'):
                alpha_i = exp['alpha'][i, 0].item() / (exp['alpha'][i, 0].item() + exp['alpha'][i, 1].item())
                plt.title(f"i: {i}, n: {n}, sf: {alpha_i:.2f}, gamma: {exp['gamma'][i].item():.2f}, lambda: {exp['lmbda']}")
                pass
            elif (diversify[0] == 'alpha' or diversify[1] == 'alpha') and (diversify[0] == 'lmbda' or diversify[1] == 'lmbda'):
                alpha_i = exp['alpha'][i, 0].item() / (exp['alpha'][i, 0].item() + exp['alpha'][i, 1].item())
                plt.title(f"i: {i}, n: {n}, sf: {alpha_i:.2f}, gamma: {exp['gamma'][i].item():.2f}, lmbda: {exp['lmbda']}")
                pass
            elif (diversify[0] == 'lmbda' or diversify[1] == 'lmbda') and (diversify[0] == 'gamma' or diversify[1] == 'gamma'):
                plt.title(f"i: {i}, n: {n}, sf: {exp['alpha']}, gamma: {exp['gamma'][i].item():.2f}, lmbda: {exp['lmbda'][i].item():.2f}")
                pass
        elif deg_free == 1:
            if diversify[0] == 'alpha':
                alpha_i = exp['alpha'][i, 0].item() / (exp['alpha'][i, 0].item() + exp['alpha'][i, 1].item())
                plt.title(f"i: {i}, n: {n}, sf: {alpha_i:.2f}, gamma: {exp['gamma']}, lmbda: {exp['lmbda']}")
                pass
            elif diversify[0] == 'gamma':
                plt.title(f"i: {i}, n: {n}, sf: {exp['alpha']}, gamma: {exp['gamma'][i].item():.2f}, lmbda: {exp['lmbda']}")
                pass
            elif diversify[0] == 'lmbda':
                plt.title(f"i: {i}, n: {n}, sf: {exp['alpha']}, gamma: {exp['gamma']}, lmbda: {exp['lmbda'][i].item():.2f}")
                pass

        plt.imshow(td_i, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.xlabel('Step')
        plt.ylabel('Trial')
        plt.show()

        if fname != "none":
            plt.savefig(f'{fname[0]}_{i}.png')

def val_over_t(e, st, dt, fname=["none"], diversify=["none"], trials = ["none"]):
    """
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
    """

    global exp, state
    exp = e
    state = st

    deg_free = len(diversify)
    if deg_free == 2:
        (fig, axs) = plt.subplots(3, 1, figsize=(8, 12))
    elif deg_free == 4:
        (fig, axs) = plt.subplots(3, 3, figsize=(24, 12))
        discrete = diversify[3]
    else:
        (fig, axs) = plt.subplots(3, 1, figsize=(8, 12))

    if deg_free == 2:
        lines_V = []
        lines_r = []
        lines_delta = []

        for i in range(n_predictors):
            (line_V, ) = axs[0].plot([], [])
            (line_r, ) = axs[1].plot([], [])
            (line_delta, ) = axs[2].plot([], [])
            lines_V.append(line_V)
            lines_r.append(line_r)
            lines_delta.append(line_delta)
    
        norm = plt.Normalize(vmin=diversify[1][0], vmax=diversify[1][1])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])

    elif deg_free == 4:
        lines_V = [[] for _ in range(len(discrete))]
        lines_r = [[] for _ in range(len(discrete))]
        lines_delta = [[] for _ in range(len(discrete))]

        for i in range(n_predictors):
            for (col_idx, category) in enumerate(discrete):
                (line_V, ) = axs[0, col_idx].plot([], [])
                (line_r, ) = axs[1, col_idx].plot([], [])
                (line_delta, ) = axs[2, col_idx].plot([], [])
                lines_V[col_idx].append(line_V)
                lines_r[col_idx].append(line_r)
                lines_delta[col_idx].append(line_delta)

        norm = plt.Normalize(vmin=diversify[1][0], vmax=diversify[1][1])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])

    else:
        lines_V = []
        lines_r = []
        lines_delta = []

        for i in range(n_predictors):
            (line_V, ) = axs[0].plot([], [])
            (line_r, ) = axs[1].plot([], [])
            (line_delta, ) = axs[2].plot([], [])
            lines_V.append(line_V)
            lines_r.append(line_r)
            lines_delta.append(line_delta)
        
    
    def update_plot(frame):
        trial_num = frame - 1

        if deg_free == 2:
            for i in range(n_predictors):
                V_frame = exp['V'][i, state, frame, :]
                delta_frame = exp['deltas'][i, state, frame, :]
                r_frame = exp['r'][state, frame, :]

            # Update lines for each plot

                lines_V[i].set_data(exp['T'][:-1], V_frame[:-1])
                lines_r[i].set_data(exp['T'][:-1], r_frame[:-1])
                lines_delta[i].set_data(exp['T'][:-1], delta_frame)

                if diversify[0] == 'alpha':
                    normalized_value = exp['alpha'][i, 0].item() / (exp['alpha'][i,
                            0].item() + exp['alpha'][i, 1].item())
                    pass
                elif diversify[0] == 'gamma':
                    normalized_value = exp['gamma'][i].item()
                    pass
                elif diversify[0] == 'lmbda':
                    normalized_value = exp['lmbda'][i].item()
                    pass
                lines_V[i].set_color(plt.cm.jet(normalized_value))
                lines_r[i].set_color(plt.cm.jet(normalized_value))
                lines_delta[i].set_color(plt.cm.jet(normalized_value))

            axs[0].set_title(f'{state} Trial: {trial_num + 2}')
            
            return lines_V + lines_r + lines_delta
        elif deg_free == 4:
            predictors_sc = [[] for _ in range(len(discrete))]
            for (col_idx, sc) in enumerate(discrete):
                if diversify[2] == 'alpha':
                    predictors_sc[col_idx] += [j for j in range(n_predictors) if compare_floats(round(exp['alpha'][j, 0].item() / (exp['alpha'][j, 0].item() + exp['alpha'][j, 1].item()), 2), sc, 0.01)]
                elif diversify[2] == 'gamma':
                    predictors_sc[col_idx] += [j for j in range(n_predictors) if compare_floats(exp['gamma'][j].item(), sc, 0.01)]
                elif diversify[2] == 'lmbda':
                    predictors_sc[col_idx] += [j for j in range(n_predictors) if compare_floats(exp['lmbda'][j].item(), sc, 0.01)]


            for i in range(n_predictors):
                V_frame = exp['V'][i, state, frame, :]
                delta_frame = exp['deltas'][i, state, frame, :]
                r_frame = exp['r'][state, frame, :]
                
                for col in range(len(discrete)):
                    if i in predictors_sc[col]:
                        col_idx = col

                lines_V[col_idx][i].set_data(exp['T'][:-1], V_frame[:-1])
                lines_r[col_idx][i].set_data(exp['T'][:-1], r_frame[:-1])
                lines_delta[col_idx][i].set_data(exp['T'][:-1],
                        delta_frame)

                if diversify[0] == 'alpha':
                    normalized_value = exp['alpha'][i, 0].item() / (exp['alpha'][i,
                            0].item() + exp['alpha'][i, 1].item())
                    pass
                elif diversify[0] == 'gamma':
                    normalized_value = exp['gamma'][i].item()
                    pass
                elif diversify[0] == 'lmbda':
                    normalized_value = exp['lmbda'][i].item()
                    pass
                lines_V[col_idx][i].set_color(plt.cm.jet(normalized_value))
                lines_r[col_idx][i].set_color(plt.cm.jet(normalized_value))
                lines_delta[col_idx][i].set_color(plt.cm.jet(normalized_value))

            axs[0][0].set_title(f'State: {state}, Trial: {trial_num + 2}, {diversify[0]} in {diversify[1]}, {diversify[2]} = {discrete[0]} (n={len(predictors_sc[0])})')
            axs[0][1].set_title(f'{diversify[2]} = {discrete[1]} (n={len(predictors_sc[1])})')
            axs[0][2].set_title(f'{diversify[2]} = {discrete[2]} (n={len(predictors_sc[2])})')

            return [artist for sublist in (lines_V, lines_r, lines_delta) for category in sublist for artist in category]
        else:
            for i in range(n_predictors):
                V_frame = exp['V'][i, state, frame, :]
                delta_frame = exp['deltas'][i, state, frame, :]
                r_frame = exp['r'][state, frame, :]

            # Update lines for each plot

                lines_V[i].set_data(exp['T'][:-1], V_frame[:-1])
                lines_r[i].set_data(exp['T'][:-1], r_frame[:-1])
                lines_delta[i].set_data(exp['T'][:-1], delta_frame)

            axs[0].set_title(f'State: {state}, Trial: {trial_num + 2}, {diversify}')

            return lines_V + lines_r + lines_delta


    sts = [index for (index, value) in enumerate(exp['states']) if value == state]
    if trials[0] != "none":
        sts = sts[trials[0]:trials[1]]
    animation = FuncAnimation(fig, update_plot, frames=sts, blit=True)  # Fix the frames argument
    

    if deg_free == 0 or deg_free == 2:
        mag = abs(torch.max(exp['r'][state, :, :])) * 2
        axs[0].set_ylabel('V(t)')
        axs[0].set_ylim(0, mag)
        axs[0].tick_params(axis='x', rotation=90)

        axs[1].set_ylabel('Reward')
        axs[1].set_ylim(0, mag)
        axs[1].tick_params(axis='x', rotation=90)

        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('TD error')
        axs[2].set_ylim(-mag, mag)
        axs[2].tick_params(axis='x', rotation=90)

    if deg_free == 4:
        mag = torch.max(exp['V'][:, state, :, :]) * 2

        axs[0][0].set_ylabel('V(t)')
        for i in range(3):
            axs[0][i].set_ylim(0, mag)
        axs[0][0].tick_params(axis='x', rotation=90)

        axs[1][0].set_ylabel('Reward')
        for i in range(3):
            axs[1][i].set_ylim(0, mag)
        axs[1][0].tick_params(axis='x', rotation=90)

        axs[2][0].set_xlabel('Time (s)')
        axs[2][0].set_ylabel('TD error')
        for i in range(3):
            axs[2][i].set_ylim(-mag, mag)
        axs[2][0].tick_params(axis='x', rotation=90)

    plt.setp(axs, xticks=np.arange(0, exp['T'][-1] + dt, dt))
    plt.tight_layout()

    plt.close(animation._fig)
    HTML(animation.to_jshtml())
    if fname != "none":
        animation.save(f'{fname[0]}.gif', writer=PillowWriter(fps=7))



############## HELPER FUNCTIONS ##############

def step(t):
    """
    Converts a continuous time value to a discrete time step.

    Args:
        t (float): The continuous time value.

    Returns:
        int: The corresponding discrete time step.
    """
    return round(t / dt)


def gen_state(ps):
    """
    Generates a state based on the given probabilities.

    Parameters:
    ps (numpy.ndarray): An array of probabilities.

    Returns:
    int: The generated state.

    """
    rand = np.random.random()
    c_ps = np.cumsum(ps)
    return np.argmax(rand < c_ps)

def t_unc(step):
    """
    Return a random time step (given uncertainty).

    Parameters:
    - step (int): The step for which to calculate the time uncertainty.

    Returns:
    - int: Random time step (within bounds of "uncertainty").

    """
    sd = int(0.1 * unc * step)
    return round(random.uniform(step - sd, step + sd))

def task_probs(tsk):
    """
    Returns the probability distribution of different states for a given task.

    Parameters:
    tsk (str): The task for which the probability distribution is required.

    Returns:
    list: The probability distribution of states for the given task.
    """

    if tsk == 'hyp_unif':
        return [1]
    if tsk == 'hyp_var':
        return [1]
    elif tsk == 'var':
        return [1]
    elif tsk == 'delay':
        return [
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
        ]
    #elif tsk == 'NEW' (EDIT HERE!!)

def params(bound):
    """
    Return list of parameter values for all value predictors from inputted bound, list, or float.

    Args:
        bound: The bound for generating parameters. It can be a tuple (for random values within that bound), list (discrete), or float (all predictors have the same value).

    Returns:
        torch.Tensor or float: Tensor or float of parameter values.
        bool: True if the generated parameters are a tensor, False otherwise.
    """
    if type(bound) is tuple:
        return torch.rand(n_predictors, 1) * (bound[1] - bound[0]) + bound[0], True
    elif type(bound) is list:
        return torch.tensor([random.choice(bound) for _ in range(n_predictors)]), True
    elif type(bound) is float:
        return bound, False

def ret_apos(sc, aneg):
    """
    Calculate the positive learning rate (for alpha parameter generation).

    Parameters:
    sc (float): Scaling factor
    aneg (float): Pessimistic learning rate.

    Returns:
    float: Optimistic learning rate.
    """
    return sc * aneg / (1 - sc)

def alpha_c(bl, delta, c):
    if bl:
      # drl.step function
        if delta > 0:
            return 0
        else:
            return 1
    else:
        return 1 / (1 + math.exp(-c * delta))

def compare_floats(float1, float2, error):
    """
    Compare two floating-point numbers with a given error tolerance.

    Args:
        float1 (float): The first floating-point number.
        float2 (float): The second floating-point number.
        error (float): The maximum allowable difference between the two numbers.

    Returns:
        bool: True if the absolute difference between the two numbers is less than or equal to the error, False otherwise.
    """
    return abs(float1 - float2) <= error
