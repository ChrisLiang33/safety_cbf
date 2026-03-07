pip install 'stable-baselines3[extra]'

the lower the alpha, the more scared the solver panics as it gets closer to the obs, it will try to step in early and interfear with the control. 

the higher the alpha, leads to high performence dodge. it means the solver is highly aggressive and trust its reaction time, so it  breaks at the last second.


random tagret(location, raduis)
dynamic alpha
dynamic kx, ky

TODO:
random obsticle(location, raduis)
figure out PPO