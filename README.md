# prediction_error_dynamics
Intrinsic motivation on high-dimensional sensory space, with dynamic goals and multi-level monitoring of prediction error dynamics.

Prediction error dynamics are monitored at different levels: at a general system level and at a goal level. At the goal level, dynamics are monitored over time buffer whose size is varying according to the general error trend. Goal selection is performed on the target that is associated with the steepest descent of the error dyanmics at the goal level.

When the dynamics at a general level have a descending trend - that is, when the performance of the system are improving - the size of the error buffers at the goal level is reduced, freeing computional resources. On the contrary, when general performance is worsening, the size of the error buffer is increased, employing thus more computational resources on finding the most promising goal to explore.

An exploratory noise is also added to the generated motor commands. When general error dynamics have increasing trend, thus when the general performance is worsening, the standard deviation of the Gaussian noise is increased, producing more random exploratory behaviours. On the contrary, when general performance is improving, the standard deviation is reduced, producing more precise and focused goal-directed movemenets.

Work submitted to Epirob2020.
