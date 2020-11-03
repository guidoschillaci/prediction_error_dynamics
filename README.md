# Tracking Emotions: Intrinsic Motivation Grounded on Multi-Level Prediction Error Dynamics
How do cognitive agents decide what is the relevant information to learn and how goals are selected to gain this knowledge? Cognitive agents need to be motivated to perform any action. We discuss that emotions arise when differences between expected and actual rates of progress towards a goal are experienced. Therefore, the tracking of prediction error dynamics has a tight relationship with emotions. Here, we suggest that the tracking of prediction error dynamics allows an artificial agent to be intrinsically motivated to seek new experiences but constrained to those that generate reducible prediction error.We present an intrinsic motivation architecture that generates behaviors towards self-generated and dynamic goals and that regulates goal selection and the balance between exploitation and exploration through multi-level monitoring of prediction error dynamics. This new architecture modulates exploration noise and leverages computational resources according to the dynamics of the overall performance of the learning system. Additionally, it establishes a possible solution to the temporal dynamics of goal selection. The results of the experiments presented here suggest that this architecture outperforms intrinsic motivation approaches where exploratory noise and goals are fixed and a greedy strategy is applied.

Intrinsic motivation on high-dimensional sensory space, with dynamic goals and multi-level monitoring of prediction error dynamics.

Prediction error dynamics are monitored at different levels: at a general system level and at a goal level. At the goal level, dynamics are monitored over time buffer whose size is varying according to the general error trend. Goal selection is performed on the target that is associated with the steepest descent of the error dyanmics at the goal level.

When the dynamics at a general level have a descending trend - that is, when the performance of the system are improving - the size of the error buffers at the goal level is reduced, freeing computional resources. On the contrary, when general performance is worsening, the size of the error buffer is increased, employing thus more computational resources on finding the most promising goal to explore.

An exploratory noise is also added to the generated motor commands. When general error dynamics have increasing trend, thus when the general performance is worsening, the standard deviation of the Gaussian noise is increased, producing more random exploratory behaviours. On the contrary, when general performance is improving, the standard deviation is reduced, producing more precise and focused goal-directed movemenets.

Work published at IEEE ICDL Epirob 2020.

Presentation: https://youtu.be/L3xtXK5vyfw
