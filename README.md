# Diversifying Parameters in Reinforcement Learning (Distributional RL)
Package designed to simulate the expected shape of an agent using a distributional TD-$\lambda$ learning scheme, an algorithm thought to be encoded by dopamine neurons in the striatum (when learning). The following functionality is included to model experiments and agents:
- Task contigencies (number and timing of cues, reward sizes and delays, etc.)
- Diversity of parameters in TD error computation ($\alpha$, $\gamma$, $\lambda$)
- Associated visualizations (learning/value at time of cue over trials, animations, etc.)
- Saving simulation data/visualizations through function parameters
<br><br>

Change the objects/functions as needed, particularly in the `drl.py` file!

Created by Siddharth Tiwari for the Uchida Lab (2024).

## Implementing Simulations

Simply follow three steps when simulating/visualizing experiments!
1. <b>Update Trial Parameters</b>
2. <b>Specify Parameters, Simulate</b>
3. <b>Visualize Experiments</b>

### 1. Update Trial Parameters

Specify trial set up (time of stimulus, size/time of reward, number of trials, etc. - see documentation and examples!) to assign schedules for experiment (simulations). <b>This will require editing the `init` function within the `drl.py` file</b> to update how stimuli and rewards are assigned to tensors. Check out sample implementations in `drl.py` and `sim.ipynb`.

### 2. Specify Parameters, Simulate </b>

After obtaining schedules for stimulus and reward, simply choose parameters for simulation! IMPORTANT: Up to 2 parameters can be distributed, where only 1 parameter can be distributed uniformly across an interval. 

Data from simulation can be saved and used for subsequent visualizations. 

### 3. Visualize Experiments

Use 1 of 3 visualizations for simulated data:
- Value at Time (`val_at_t`): Value at inputted time, across trials.
- TD-Error Heatmap (`heatmap`): TD error across trials and time steps in a given trial.
- Value over Time (`val_over_t`): Animation of value over trials for all predictors (at all time steps).

Every visualization has the option to "diversify" parameters, such that we can observe how value predictors with different parameters behave in response to the same reward schedules. The specifications for this are included in `sim.ipynb`

## End Remarks