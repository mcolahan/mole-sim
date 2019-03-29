import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Lennard_Jones(r, dist_cutoff=3):
    # r is a 2D array where row i, and col j correspond to particle i and dimension j (x:0, y:1, z:2)
    # outputs: a 3D array where f[i,j,k] is the force on i because of j in dimension k,
    # sum of potential energy, and virial for instantaneous pressure calcs

    #             part i    part j  (x,y,z)

    f = np.zeros([n_parts, n_parts, n_dims])
    E_p_sum = 0

    virial = 0  # for pressure calculation

    for i in range(n_parts-1):
        # get position vectors and distances between particles
        r_ij_vect = r[i, :] - r[(i + 1):, :]
        r_ij_vect -= np.around(r_ij_vect / box_size) * box_size  # pbc

        r_ij = np.sqrt(np.sum(r_ij_vect ** 2, axis=1))  # magnitude (norm) of r_ij_vect

        # get index of particles within range of the spherical cutoff
        parts_in_range = np.where(r_ij <= dist_cutoff)[0]

        # distance magnitude of particles within the spherical cutoff
        r_near = r_ij[parts_in_range]

        # Lennard Jones
        # eq 5.3 in Allen and Tildesley 2nd ed
        # fij = -du(rij)/dr * unit vector of r_ij_vect
        f_mag = np.zeros([n_parts - i - 1])
        f_mag[parts_in_range] = (
            24 * epsilon / r_near * (2 * (sigma / r_near) ** 12 - (sigma / r_near) ** 6)
        )

        # new axis converts 1D array to a 2D array
        f_vect = f_mag[:, np.newaxis] * r_ij_vect / r_ij[:, np.newaxis]
        f[i, (i + 1) :, :] = f_vect
        f[(i + 1) :, i, :] = -f_vect  # newton's 3rd law

        virial += np.sum(r_ij_vect * f_vect)

        Ep = np.zeros([n_parts - i - 1])
        Ep[parts_in_range] = (
            4 * epsilon * ((sigma / r_near) ** 12 - (sigma / r_near) ** 6)
        )

        E_p_sum += np.sum(Ep)

    virial = virial / 3
    return f, E_p_sum, virial



def molecular_dynamics(part_props, ensemble, box_size, time_props, Tsp=None, deltat_tao = 0.005, 
                       save_csv=False, sim_name="", return_results=True):
    global n_parts, n_dims, sigma, epsilon, m

    epsilon = part_props['epsilon']
    sigma = part_props['sigma']
    m = part_props['mass']
    number_density = part_props['density']
    min_dist = part_props['min_dist']      # minimum distance between particles in initialization
    
    n_dims = len(box_size)

    delta_t = time_props['delta_t']        # time between time steps
    n_timesteps = time_props['timesteps']     # how many time steps?
    output_steps = time_props['steps between out']
    dist_cutoff = 3 * sigma
    
    time =  np.arange(0, n_timesteps * delta_t, delta_t)
    time_out = time[::output_steps]   # get time after output_steps steps 
    n_outputs = np.shape(time_out)[0]
    
    # Ensemble considerations:
    if ensemble == 'Microcanonical':
        const_T = False
    elif ensemble == 'Canonical':
        const_T = True
    else:
        raise ValueError('Ensemble type not defined.')
        

    ## Initialization and determining particle starting locations (from assignment 1)

    n_parts = int(number_density * np.prod(box_size))
    
    V = np.prod(box_size)
    Kb = 1
    
    init_locs = np.random.rand(n_parts, n_dims) * box_size
    min_dists = []
    for i in range(n_parts):
        continue_looping = True
        if i == 0:
            continue
        loop_count = 0
        while continue_looping == True:
            if loop_count > 10000:
                raise InterruptedError('Loop was interrupted')

            dr = init_locs[i,:] - init_locs[:i,:]
            dr = dr - np.around(dr / box_size) * box_size   # PBC 
            dist = np.sqrt(np.sum(dr**2, axis=1))

            if (dist < min_dist).any() == True:
                init_locs[i, :] = np.random.rand(1, n_dims) * box_size
            else:
                continue_looping = False
                min_dists.append(min(dist))

            loop_count += 1


    ## Time Integration

    #initialize position and velocity arrays (indices of arrays: particles, dimensions, time)
    # e.g., r[i, :, k] is position of particle i in all dimensions (x, y, z) at time k
    r = np.zeros([n_parts, n_dims, n_timesteps+1])   
    r[:, :, 0] = init_locs
    v = np.zeros([n_parts, n_dims, n_timesteps+1])
    a = np.zeros([n_parts, n_dims])

    # outputs init
    r_out = np.zeros([n_parts, n_dims, n_outputs])
    v_out = np.zeros([n_parts, n_dims, n_outputs])
    Ep = np.zeros(n_outputs)
    Ek = np.zeros(n_outputs)
    E = np.zeros(n_outputs)
    P = np.zeros(n_outputs)
    T = np.zeros(n_outputs)
    
    out_iter = 0

#     printed = False    # for debugging purposes

    for t in time:
        if t == 0:
            r = init_locs
            v = np.zeros([n_parts, n_dims])
            
            # Calculate initial acceleration
            f, pot_energy, virial = Lennard_Jones(r[:,:], dist_cutoff=dist_cutoff)
            f_vect = np.sum(f, axis=1) 
            a_vect = f_vect / m
            

        # using velocity-verlet algorithm
        v_halfdt = v + 0.5*delta_t * a_vect
        r = r + delta_t * v_halfdt

        # check if out of box and update if so
        r = r - np.floor(r / box_size) * box_size

        # update forces and acceleration with new r
        f, pot_energy, virial = Lennard_Jones(r)
        f_vect = np.sum(f, axis=1) 
        a_vect = f_vect / m

        v = v_halfdt + 0.5 * delta_t * a_vect

        if const_T == True:
            v_mag = np.sqrt(np.sum(v**2, axis=1))
            KE = np.sum(0.5 * m * v_mag**2)
            T_inst = 2 * KE / (3*(n_parts -1) * Kb)
            # Berendesen thermostat
            lam = np.sqrt(1 + deltat_tao * (Tsp / T_inst - 1))

            v *= lam
        
        v_mag = np.sqrt(np.sum(v**2, axis=1))
        KE = np.sum(0.5 * m * v_mag**2)
        T_inst = 2 * KE / (3*(n_parts -1) * Kb) 
        
        P_inst = number_density * T_inst + virial/V
        
        if t in time_out:
            r_out[:,:,out_iter] = r
            v_out[:,:,out_iter] = v
            Ek[out_iter] = KE
            Ep[out_iter] = pot_energy
            E[out_iter] = KE + pot_energy
            T[out_iter] = T_inst
            P[out_iter] = P_inst
            
            out_iter += 1

    if save_csv == True:
        cols = ['t', 'E', 'Ek', 'Ep', 'T', 'P']
        out_df = pd.DataFrame(np.array([time_out, E, Ek, Ep, T, P]).T, columns=cols)
        
        time_df = pd.DataFrame(time_out, columns=pd.MultiIndex.from_tuples([('t','t')]))

        dims = ['x', 'y', 'z']
        mult_ind = pd.MultiIndex.from_product([np.arange(n_parts), dims])
        r_df = pd.DataFrame(columns=mult_ind)
        r_df = pd.concat([time_df, r_df])
        v_df = r_df.copy()
        for i in range(n_outputs):
            for part in range(n_parts):
                r_df.loc[i, [part][:]] = r_out[part, :, i] 
                v_df.loc[i, [part][:]] = v_out[part, :, i]
                
        r_df.to_csv(sim_name + ' - position.csv')
        v_df.to_csv(sim_name + ' - velocity.csv')
        out_df.to_csv(sim_name + ' - ETP.csv')

    if return_results == True:
        return time_out, r_out, v_out, E, Ek, Ep, T, P
            
            

if __name__ == "__main__":

    box_size = np.array([8,8,8])    # (x, y, z)
    part_props = {
        'epsilon': 1,
        'sigma': 1, 
        'mass': 1,
        'density': 0.5,
        'min_dist': 0.8,
    }
    time_props = {
        'delta_t': 0.001,
        'timesteps': 5000,
        'steps between out': 10    # output every n_steps steps 
    }

    molecular_dynamics(part_props, 'Canonical', box_size, time_props, Tsp=0.8, 
                                                save_csv=True, sim_name='Q2 - Gr2', return_results=False)
