from controller import Supervisor
import PuPy

checks = []
# Revert the simulation if the robot tumbled ...
checks.append(PuPy.RevertTumbled(grace_time_ms=(3 * 3000)))
# ... or went out of a predefined space
checks.append(PuPy.RevertOutOfArena(arena_size=(-10, 10, -10, 10), distance=0))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
