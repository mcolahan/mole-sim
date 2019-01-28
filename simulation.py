import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class Simulation:

    def __init__(self, time_start=None, time_end=None, time_step=None, ):
        self.box = None
        self.particles = None

        if time_end is not None and time_start is None:
            self.time_start = 0
        elif time_start is not None:
            self.time_start = time_start

        self.time_end = time_end
        self.time_step = time_step

        if all([val != None for val in (time_start, time_end, time_step)]):
            self.times = np.arange(start_time, end_time, time_step)

        self.run_flag = False

    def set_time(self, start, end, step):
        self.time_start = start
        self.time_end = end
        self.time_step = step
        
        try:
            self.times = np.arange(self.time_start, self.time_end, self.time_step)
        except:
            raise ValueError('Time specification values are not numeric.') 

    
    
    def create_random_spherical_particle_loc_in_box(self, n_parts, part_diam=0, allow_wall_overlap=False):
        box_size = self.box.size
        n_dims = self.n_dims
        part_radius = part_diam / 2
        box_dims = np.array([np.zeros(n_dims), box_size]) 

        box_vol = np.prod(box_size)
        spheres_vol = n_parts * np.pi / 6 * part_diam ** 3
        if box_vol < 5 * spheres_vol:
            raise ValueError("Particles take up too much volume in box. Consider" + 
                        "a crystallographic method instead.")

        part_locs = np.random.rand(n_parts, n_dims) * box_size

        for i in range(n_parts):
            in_free_space = False
            n_loops = 0
            while in_free_space == False:
                # while loops safety:
                n_loops += 1
                if n_loops > 100:
                    break

                part_i_loc = part_locs[i, :]

                # test if overlaps box wall
                if allow_wall_overlap == False:
                    dist = abs(box_dims - part_i_loc)
                    if (dist < part_radius).any() == True:
                        # set to new location and do over
                        part_locs[i, :] = np.random.rand(1, n_dims) * box_size
                        continue

                restart_flag = False
                for j in range(i+1, n_parts):
                    # Calc distance between particle
                    part_j_loc = part_locs[j, :]
                    dist = Simulation.calc_distance_between_locs(part_i_loc, part_j_loc)
                    if dist < part_diam:
                        part_locs[i, :] = np.random.rand(1, n_dims) * box_size
                        restart_flag = True
                        break

                if restart_flag == False:
                    in_free_space = True

        return part_locs


    @staticmethod
    def calc_distance_between_locs(loc1, loc2):
        return np.sqrt(sum([(loc1[k] - loc2[k])**2 for k in range(len(loc1))])) 
        
        

    def plot_initial(self):
        if particles is None:
            return None
        plt.figure()
        plt.subplot(111)
        
        x = [part.location[0] for part in self.particles]
        y = [part.location[1] for part in self.particles]
        sizes = [part.diameter for part in self.particles]
        
        #TODO: make the marker size respective to the particle
        plt.scatter(x, y)
        plt.show()

    def animate_results(self):

        if self.run_flag is not True:
            raise RuntimeError('Please run calculation before animating results.')

        fig, ax = plt.subplots()
        x, y = [], []
        pts, = plt.plot([], [], 'o', animated=True)

        def init():
            ax.set_xlim(0, self.box.size[0])
            ax.set_ylim(0, self.box.size[1])
            return pts,

        def update(frame):
            n = frame
            x = [part.location[n, 0] for part in self.particles]
            y = [part.location[n, 1] for part in self.particles]
            pts.set_data(x, y)
            return pts,

        ani = FuncAnimation(fig, update, frames=np.arange(len(self.times)),
                            init_func=init, blit=True)
        plt.show()

    def calculate(self):
        time_step = self.time_step
        start_time = self.time_start
        end_time = self.time_end
        times = self.times
        n_times = len(times)

        n_dims = self.n_dims

        particles = self.particles
        n_parts = len(particles)

        box_dims = self.box.size
        box_dims = np.array([np.zeros(n_dims), box_dims])

        loc_array = np.zeros((n_parts, n_times, n_dims))
        for n in range(n_parts):
            loc_array[n, 0, :] = particles[n].location

        vel_array = np.zeros((n_parts, n_times, n_dims))
        for n in range(n_parts):
            vel_array[n, 0, :] = particles[n].velocity
        

        for n in range(n_times):
            if n == 0:
                part_locs[:, :, n] = part_locs_init
                continue

            delta_t = times[n] - times[n - 1]
            new_part_loc = part_locs[:, :, n-1] + part_vel * delta_t
            for n in range(n_dims):
                new_part_loc[new_part_loc[:,n] > box_size[n], n] -= box_size[n]
                new_part_loc[new_part_loc[:,n] < 0, n] += box_size[n]
            
            part_locs[:, :, n] = new_part_loc




        self.run_flag = True

        for n in range(n_parts):
            particles[n].velocity = vel_array[n, :, :]
            particles[n].location = loc_array[n, :, :]


        # results_dict = {'locations': loc_array, 'velocities': vel_array}
        # return results_dict


class Box:
    def __init__(self, ndims, size):
        self.ndims = ndims
        if len(size) == ndims:
            self.size = size
        else:
            raise ValueError('Box size does not match number of dimensions.')

class AbstractParticle():

    counter = 0 

    def __init__(self):
        self.size = None
        self.potential_func = None
        self.location = None
        self.type = None
        
        self.id = AbstractParticle.counter
        AbstractParticle.counter += 1

    def __repr__(self):
        return f'{self.type} {self.id}'


class HardSphere(AbstractParticle):

    def __init__(self, location, initial_velocity, diameter=3):
        super().__init__()
        self.type = "Hard Sphere"
        self.location = location
        self.velocity = initial_velocity
        self.diameter = diameter #Angstroms
    
    




if __name__ == "__main__":

    sim = Simulation()
    n_dims = 2
    box_size = (10,10)
    sim.box = Box(ndims=n_dims, size=box_size)
    sim.n_dims = n_dims


    n_particles = 20
    part_diam = 0.5
    part_locs = sim.create_random_spherical_particle_loc_in_box(n_particles, part_diam=part_diam)
    

    # mean_velocity = 0.5
    # velocities = np.random.normal(loc=mean_velocity, scale=1, size=(n_particles, n_dims))

    # particles = [HardSphere(location=part_locs[n,:], initial_velocity=velocities[n,:], diameter=1) for n in range(n_particles)]
    # sim.particles = particles

    # sim.set_time(0, 10, 0.5)
    # sim.calculate()

    # sim.plot_initial()
    plt.figure()
    ax = plt.subplot(111, aspect='equal')
    for n in range(n_particles):
        circle = patches.Circle(xy=part_locs[n, :], radius=part_diam/2)
        ax.add_artist(circle)
    ax.set_xlim([0, box_size[0]])
    ax.set_ylim([0, box_size[1]])
    plt.show()

    


        
