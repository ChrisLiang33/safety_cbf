pip install 'stable-baselines3[extra]'

High α near obstacle = Late reaction = Demands impossible hardware speed = Crash.
Low α near obstacle = Early reaction = Demands manageable hardware speed = Safe navigation.

Alpha is a scaler that determines how fast can the agent approach the boundry, high alpha means we are allowed to approach the 
boundry at a fast speed which we also mean that we might need to swerve fast last minute to avoid collision, the allowed velocity is high as for a larger time period.  
a low alpha means we are allowed tp approach the boundry at a slower speed which we try to avoid the last minute collision, the velocity is mostly low for the majority of the time

now for a optimal alpha, i expect the alpha to remain the highest it can be while the motor and the solver allows, that way it can keep the tightest pathing to be energy efficient. as to why the alpha comes back down to 0.1, i dont know

<!-- when epsilon is big. the term becomes small, it basicly vanishes. because math is forcing a huge buffer zone
when its small, it treats the obstcale larger than it is, it will start swerving or braking in advance -->

Q:why always alpha comes down to 0.1
1. The Constraint Deactivates
Once the robot is driving away from the obstacle, the distance h(x) grows larger. Let's say h(x) is 3.0 meters. The right side of your CBF equation (−αh(x)) becomes a large negative number.
Simultaneously, because the robot is moving away, the radar gun dot product (Lg​h⋅u) becomes a positive number.
The solver checks: Positive Number >= Large Negative Number.
This is trivially true. The constraint is completely inactive, and the CBF goes to sleep.

1. The AI Parks the Variable
Because the CBF is asleep, changing α from 0.1 to 5.0 has absolutely zero effect on the robot's trajectory. If it doesn't change the trajectory, it doesn't change the reward.
If a variable doesn't change the reward, the neural network receives no mathematical feedback (a gradient of zero) to adjust it.
Since the AI was forced to push α down to 0.1 to survive the initial approach, it simply leaves it parked there for the rest of the episode because it has no incentive to spend brainpower raising it back up.

TODO:
understand PPO
fix the training loop of parallenisim problem


