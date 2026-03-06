pip install 'stable-baselines3[extra]'

there were problems with running off the map, currently does not drive forward and curve around he obstacle
it seems to learn that by keeping the alpa and epsilon low, it could trigger the violent forcefield 
to bounce the robot safely backward off the map and end the episode without crashing

alpha was dialed down to the min