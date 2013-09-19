"""

.. todo::
    documentation
    

"""
import numpy as np
import pylab
import warnings
#import epuck_arena


def _intersect((o1x, o1y), (d1x, d1y), (o2x, o2y), (d2x, d2y)):
    """Intersection of two bounded lines. The lines are given
    with the origin and direction. Returned is the step length for
    both lines, in the same order as the input.
    
    o1x + t1 * d1x = o2x + t2 * d2x
    o1y + t1 * d1y = o2y + t2 * d2y
    => t1 = (o2x + t2 * d2x - o1x)/d1x
    => o1y + ((o2x + t2 * d2x - o1x)/d1x) * d1y = o2y + t2 * d2y
    => o1y + (o2x + t2 * d2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y + (t2 * d2x + o2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y + t2*d2x*d1y/d1x + (o2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y - o2y + (o2x - o1x) * d1y/d1x = t2 * d2y - t2*d2x*d1y/d1x
    => o1y - o2y + (o2x - o1x) * d1y/d1x = t2 * (d2y - d2x*d1y/d1x)
    => t2 = (o1y - o2y + (o2x - o1x) * d1y/d1x) / (d2y - d2x*d1y/d1x)
    
    """
    tol = 1e-14
    if abs(d1y - 0.0) < tol :
        # o_dir = (!0.0, 0.0)
        if abs(d2y - d2x*d1y/d1x) < tol: # parallel
            t0, t1 = float('inf'), float('inf')
        else:
            nom = o2y - o1y - d1y * (o2x - o1x)/d1x
            denom = (d1y*d2x)/d1x - d2y
            t0 = nom/denom
            t1 = (o2x - o1x + t0 * d2x)/d1x
    else:
        # o_dir = (0.0, !0.0)
        if abs(d2x - d2y*d1x/d1y) < tol: # parallel
            t0, t1 = float('inf'), float('inf')
        else:
            nom = o2x - o1x - d1x * (o2y - o1y)/d1y
            denom = (d1x*d2y)/d1y - d2x
            t0 = nom/denom
            t1 = (o2y - o1y + t0 * d2y) / d1y
    
    return t1, t0

def _in_obstacle(loc, obstacle):
    """Check if a location is within an obstacle.
    
    Assuming the obstacle edges are given in the right order (meaning
    that the polygon is defined through lines between successive
    points).
    
    As reference, the origin is picked. This implies that the obstacle
    must not include the origin.
    
    Edges and corners count as within the obstacle
    
    """
    if any([loc == obs for obs in obstacle]):
        return True
    
    faces = [(p0,p1) for p0,p1 in zip(obstacle[:-1], obstacle[1:])]
    faces.append((obstacle[-1], obstacle[0]))
    
    num_intersect = sum([_obs_intersect((loc, (0.0, 0.0)), line) for line in faces])
    if num_intersect % 2 == 0:
        return False
    else:
        return True

def _obs_intersect(((x0,y0), (x1,y1)), ((x2,y2), (x3,y3))):
    """Check if two lines intersect. The boundaries don't count as
    intersection."""
    base1 = (x0, y0)
    base2 = (x2, y2)
    dir1 = (x1-x0, y1-y0)
    dir2 = (x3-x2, y3-y2)
    t1, t2 = _intersect(base1, dir1, base2, dir2)
    
    eps = 0.00001
    if -eps < t1 and t1 < 1.0 + eps and -eps < t2 and t2 < 1.0 + eps:
        return True
    else:
        return False

class Robot(object):
    """
    
    obstacles are polygons, defined by the corner points
    obstacle_line are single lines, not necessarily connected
    
    the robot cannot be reset within a polygon
    polygons may not include the origin, (0, 0)
    
    """
    def __init__(self, obstacle_line=None, obstacles=None, speed=0.5, step_time=1.0, tol=0.0):
        
        if obstacles is None:
            obstacles = []
        
        if obstacle_line is None:
            obstacle_line = []
        
        obstacle_line = obstacle_line[:]
        for obs in obstacles:
            obstacle_line.extend([(x0,y0,x1,y1) for (x0,y0), (x1,y1) in zip(obs[:-1], obs[1:])])
            obstacle_line.append((obs[-1][0], obs[-1][1], obs[0][0], obs[0][1]))
        
        if tol > 0.0:
            warnings.warn("tolerance > 0 doesn't work properly; It only works if the robot faces the wall (not when parallel or away from the wall).")
        
        self.sensors = [2*np.pi*i/8.0 for i in range(8)]
        #self.obstacles = [ (x0,y0,x1,y1) ]
        self.obstacle_line = obstacle_line
        self._ir_max, self.tol = 15.0, tol
        self.obstacles = self._cmp_obstacles(self.obstacle_line)
        self.polygons = obstacles[:]
        self.speed, self.step_time = speed, step_time
        self.reset()
    
    def _cmp_obstacles(self, lines):
        """
        """
        obstacles = []
        for x0,y0,x1,y1 in lines:
            o_vec = (x1-x0, y1-y0)
            if o_vec[0] == 0.0 and o_vec[1] == 0.0:
                raise Exception('Obstacle line must have a direction')
            o_base = (x0,y0)
            o_limit = 1.0
            obstacles.append((o_vec, o_base, o_limit))
        return obstacles
    
    def _cmp_obstacle_lines(self, obstacles):
        """
        """
        lines = []
        for o_vec, o_base, o_limit in obstacles:
            x0, y0 = o_base
            if o_limit == float('inf'):
                raise Exception('Infinite lines not supported')
            x1 = o_base[0] + o_limit * o_vec[0]
            y1 = o_base[1] + o_limit * o_vec[1]
            lines.append((x0,y0,x1,y1))
        return lines
    
    def reset(self):
        """Reset the robot to the origin."""
        self.loc = (0.0, 0.0)
        self.pose = 0.0
        self.trajectory = [self.loc]
    
    def reset_random(self, loc_lo=-10.0, loc_hi=10.0):
        """Reset the robot to a random location, outside the obstacles."""
        for i in xrange(1000):
            loc = self.loc = (np.random.uniform(loc_lo, loc_hi), np.random.uniform(loc_lo, loc_hi))
            pose = self.pose = np.random.uniform(0, 2*np.pi)
            
            if not any([_in_obstacle(self.loc, obs) for obs in self.polygons]) and not self.take_action(0.0):
                break
        
        if i == 1000:
            warnings.warn('Random reset iterations maximum exceeded')
        
        self.loc = loc
        self.pose = pose
        self.trajectory = [self.loc]
    
    def read_ir(self):
        """Compute the proximities to obstacles in all infrared sensor
        directions."""
        # view-direction
        readout = []
        for sensor in self.sensors:
            s_dist = self._ir_max
            s_ori = self.pose + sensor
            s_dir = (np.cos(s_ori), np.sin(s_ori))
            s_base = self.loc
            
            for o_dir, o_base, o_limit in self.obstacles:
                # obstacles intersection
                t0, t1 = _intersect(o_base, o_dir, s_base, s_dir)
                
                eps = 0.00001
                if t1 >= 0 and (o_limit == float('inf') or (-eps <= t0 and t0 <= o_limit + eps)):
                #if t0 >= 0 and t1 >= 0 and t1 <= 1.0:
                    # intersection at distance (t0 * s_dir)
                    dist = np.linalg.norm((t1 * s_dir[0], t1 * s_dir[1]))
                else:
                    # no intersection
                    dist = self._ir_max
                
                if dist < s_dist:
                    s_dist = dist
            
            readout.append(s_dist)
            
        return readout
    
    def read_sensors(self):
        """Read all sensors. A :py:keyword:`dict` is returned."""
        ir = self.read_ir()
        #noise = np.random.normal(scale=0.01, size=(len(ir)))
        #ir = map(operator.add, ir, noise)
        
        return {'loc': np.atleast_2d(self.loc), 'pose': np.atleast_2d(self.pose), 'ir': np.atleast_2d(ir)}
    
    def take_action(self, action):
        """Execute an ``action`` and move forward
        (speed * step_time units or until collision). Return
        :py:keyword:`True` if the robot collided.
        
        """
        # turn
        if isinstance(action, np.ndarray): action = action.flatten()[0]
        self.pose = (self.pose + action) % (2*np.pi)
        #self.pose = action % (2*np.pi)
        
        # move forward
        t = self.speed * self.step_time # distance per step
        
        # Collision detection
        eps = 0.00001
        r_vec = (np.cos(self.pose), np.sin(self.pose))
        wall_dists = [(idx, _intersect(self.loc, r_vec, o_base, o_vec)) for idx, (o_vec, o_base, o_limit) in enumerate(self.obstacles)]
        wall_dists = [(idx, r_dist) for idx, (r_dist, o_dist) in wall_dists if r_dist >= 0.0 and r_dist < float('inf') and -eps <= o_dist and o_dist <= 1.0 + eps] # FIXME: Replace 1.0 by o_limit
        if len(wall_dists) > 0:
            # Distance to the wall
            wall_idx, min_wall_dist = min(wall_dists, key=lambda (idx, dist): dist)
            dist = np.linalg.norm((min_wall_dist * r_vec[0], min_wall_dist * r_vec[1]))
            
            # angle between wall and robot trajectory
            o_vec = self.obstacles[wall_idx][0]
            a = np.arccos( (o_vec[0] * r_vec[0] + o_vec[1] * r_vec[1]) / (np.linalg.norm(o_vec) * np.linalg.norm(r_vec)) )
            if a > np.pi/2.0:
                a = np.pi - a
            
            # maximum driving distance
            k = self.tol / np.sin(a)
            t_max = dist - k
            
        else:
            # no wall ahead
            t_max = float('inf')
        
        collide = t >= t_max
        t = min(t, t_max)
        
        # next location
        self.loc = (self.loc[0] + np.cos(self.pose) * t, self.loc[1] + np.sin(self.pose) * t) # t doesn't denote the distance in moving direction!
        self.trajectory.append(self.loc)
        return collide
    
    def plot_trajectory(self, wait=False, with_tol=True, tol=None, full_view=True):
        """Plot the robot trajectory in a :py:mod:`pylab` figure.
        
        .. todo::
            pylab figure configurable
        
        ``wait``
        
        ``with_tol``
        
        ``tol``
        
        ``full_view``
        
        """
        pylab.clf()
        self._plot_obstacles(with_tol, tol)
        x,y = zip(*self.trajectory)
        pylab.plot(x,y,'b-')
        pylab.plot(x,y,'b*')
        if full_view:
            x0,x1,y0,y1 = pylab.axis()
        else:
            x0,x1,y0,y1 = min(x), max(x), min(y), max(y)
        pylab.axis((
            x0 + x0*0.1,
            x1 + x1*0.1,
            y0 + y0*0.1,
            y1 + y1*0.1
            ))
        
        pylab.show(block=wait)

    def _plot_obstacles(self, with_tol=True, tol=None):
        """Plot all obstacles and walls into a :py:mod:`pylab` figure.
        
        .. todo::
            pylab figure configurable
        
        ``with_tol``
            
        
        ``tol``
            
        
        """
        if tol is None: tol = self.tol
        for vec, base, limit in self.obstacles:
            # obstacle line
            pylab.plot((base[0], base[0]+limit*vec[0]), (base[1], base[1]+limit*vec[1]), 'k')
            
            if with_tol and tol > 0:
                if vec[1] == 0.0:
                    y = (-vec[1]/vec[0], 1.0)
                else:
                    y = (1.0, -vec[0]/vec[1])
                    
                y = (y[0] * tol / np.linalg.norm(y), y[1] * tol / np.linalg.norm(y))
                base_tn = (base[0] - y[0], base[1] - y[1])
                base_tp = (base[0] + y[0], base[1] + y[1])
            
                # obstacle tolerance
                pylab.plot((base_tn[0], base_tn[0]+limit*vec[0]), (base_tn[1], base_tn[1]+limit*vec[1]), 'k:')
                pylab.plot((base_tp[0], base_tp[0]+limit*vec[0]), (base_tp[1], base_tp[1]+limit*vec[1]), 'k:')
                
                # tolerance edges
                # tip_hi = base+limit*vec+tol/nrm(vec)*vec
                # line from base_tn+limit*vec to tip_hi
                # line from base_tp+limit*vec to tip_hi
                # 
                # tip_lo = base - tol/nrm(vec)*vec
                # line from base_tn to tip_lo
                # line from base_tp to tip_lo
                
        pass

class AbsoluteRobot(Robot):
    """
    
    
    
    """
    def take_action(self, action):
        """Execute an ``action`` and move forward
        (speed * step_time units or until collision). Return
        :py:keyword:`True` if the robot collided.
        
        """
        if isinstance(action, np.ndarray): action = action.flatten()[0]
        self.pose = action % (2*np.pi)
        return super(AbsoluteRobot, self).take_action(0.0)

def simulation_loop(acd, robot, max_step=-1, max_episodes=-1, max_total_iter=-1):
    """
    
    ``acd``
    
    ``robot``
    
    ``max_step``
    
    ``max_episodes``
    
    ``max_total_iter``
    
    """
    if max_step < 0 and max_episodes < 0 and max_total_iter < 0:
        raise Exception('The simulation cannot run forever.')
    
    policy = acd.policy
    num_episode = 0
    num_total_iter = 0
    while True:
        
        # init episode
        acd.new_episode()
        robot.reset()
        #robot.reset_random(loc_lo=-9.0, loc_hi=9.0)
        policy.reset()
        a_curr = np.atleast_2d([policy.action])
        
        num_step = 0 # k
        while True:
            
            # Apply current action
            collided = robot.take_action(a_curr)
            
            # Observe sensors
            s_next = robot.read_sensors()
            
            # Execute ACD
            a_next = acd(s_next, num_step, num_step+1, 1)
            
            # Iterate
            num_step += 1
            num_total_iter += 1
            if collided:
                break
            if max_step > 0 and num_step >= max_step:
                break
            acd.a_curr = a_curr = a_next
        
        if num_step <= 3:
            print "Warning: episode ended prematurely"
        
        num_episode += 1
        if max_episodes > 0 and num_episode >= max_episodes:
            break
        if max_total_iter > 0 and num_total_iter >= max_total_iter:
            break
    
    return acd
