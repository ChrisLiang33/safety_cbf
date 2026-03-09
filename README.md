pip install 'stable-baselines3[extra]'

the lower the alpha, the more scared the solver panics as it gets closer to the obs, it will try to step in early and interfear with the control. means more careful it is, when closer to the obs the alpha is lowest

the higher the alpha, leads to high performence dodge. it means the solver is highly aggressive and trust its reaction time, so it  breaks at the last second. more recless, so when its far from obs, the alpha is high

when epsilon is big. the term becomes small, it basicly vanishes. because math is forcing a huge buffer zone

when its small, it treats the obstcale larger than it is, it will start swerving or braking in advance

problems:
cvxpy habdles distances flawed, when robot is far away from the obs 5meters away. h_x is a large number 24
if robot is far way and plugged into the constraint
lgh * u >= -alpha h_x + lgh(2) / epsilon

-alpha h(x) becomes a negative number, but if AI tries to optimize the 1/epsilon penalty, lgh**2/ epsilon explodes into a massive numer
like 400

then the constraint looks like:
lgh * u >= -120 + 400
lgh * u >= 280, this forces the robot to move away at max speed even when its 5 meters away


on changing the h(x):
1. The Safety Boundary (The Zero-Level Set)

    Mathematically, the boundary itself did not change at all. The edge of the red circle is exactly where h(x)=0. Whether you use 52−52=0 or 5−5=0, the physical wall is in the exact same coordinate space.

2. The Gradient (The Repulsion Force)
This is where everything changed.

    Old (Squared): h(x)=∥x−xobs​∥2−r2. The derivative is 2(x−xobs​). This means the "repulsion" gradient grows linearly the further away the robot gets.

    New (Linear): h(x)=∥x−xobs​∥−r. The derivative is simply ∥x−xobs​∥x−xobs​​. This is a Unit Vector.

probelm:
Let's trace the math when the robot is 5 meters away from the obstacle:

    The distance vector (x−xobs​) has a length of 5.

    Therefore, Lg​h is a vector with a length of 10.

    This makes ∥Lg​h∥2=100.

    If the AI chooses ϵ=2.0, your robustness margin becomes 2100​=+50.

The solver is being told:
Lg​h⋅u≥−αh(x)+50

Because your robot's hardware can only output a maximum velocity of 2.0 m/s, it is mathematically impossible to satisfy a requirement of +50. The cvxpy solver panics, throws the UserWarning: Solution may be inaccurate you see in the logs, and triggers your fallback code: safe_u = np.array([0.0, 0.0]).

implemented features:
random tagret(location, raduis)
random obs(location, raduis)
dynamic alpha
dynamic kx, ky

TODO:
figure out PPO
fix the train.py file, it has many uselss plots under


