from controller import Supervisor
import PuPy

# checks
checks = []
checks.append(PuPy.RevertOutOfArena(arena_size=(-10, 10, -10, 10), distance=0, grace_time_ms=(3 * 3000)))
# respawn the robot at a random location in a bounded area
checks.append(PuPy.RevertTumbled(grace_time_ms=(3 * 3000)))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
