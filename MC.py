import numpy as np
import matplotlib.pyplot as plt


beta = 1  # beta = 1 / (k_b * T)
T = 1 / beta
P = 3

# particle properties
epsilon = 1,
sigma = 1 
mass = 1
density = 1

min_dist = 0.8
dist_cutoff = 3

# MC algorithm params:
delta_rmax = 0.5
delta_Vmax = 0.02
delta_rmax_min = 0.1
delta_Vmax_min = 0.005
target_acceptance_range = (0.3, 0.5)


step_count = 20  # number steps between acceptance rate check

# what to change each step:
n_steps = 10000
steps_def = [(0,n_steps,'both')]


n_dims = 3
box_size = np.array([8,8,8])
min_size = np.array([6.5,6.5,6.5])

n_parts = int(density * np.prod(box_size))
V = np.product(box_size)

# r_out = np.zeros([n_parts, n_dims, n_frames])


global sr_12, sr12_new, sr_6, sr_6_new    # (sigma/r)^n


def lennard_jones(r, u, w, part_moved=None):
    global sr_12, sr12_new, sr_6, sr6_new
    sr12_new = np.copy(sr_12)
    sr6_new = np.copy(sr_6)

    for i in range(n_parts):
        if part_moved is not None:
            i = part_moved 

        r_ij_vect = r[i,:] - r[(i+1):,:]
        r_ij_vect -= np.around(r_ij_vect / box_size) * box_size   #pbc
        r_ij = np.sqrt(np.sum(r_ij_vect**2, axis=1))   # magnitude (norm) of r_ij_vect

        # get index of particles within range of the spherical cutoff
        parts_in_range = np.where(r_ij <= dist_cutoff)[0]
        
        # distance magnitude of particles within the spherical cutoff
        r_near = r_ij[parts_in_range] 
        
        # reset row to zeros
        old = np.copy(u[i, (i + 1):])
        u[i, (i+1):] = 0
        w[i, (i + 1):] = 0
        
        locs = i + 1 + parts_in_range
        sr12_new[i, locs] = (sigma / r_near) ** 12
        sr6_new[i, locs] = (sigma / r_near) ** 6

        u[i, :] = (sr12_new[i,:] - sr6_new[i,:]) * 4 * epsilon
        w[i, :] = (sr6_new[i,:] - 2 * sr12_new[i,:]) * 24 * epsilon 
                
        if part_moved is not None:
            break
            
    return u, w


# Create Initial Configuration

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

          
part_locs = init_locs

u = np.zeros([n_parts, n_parts])   # potential
w = np.zeros([n_parts, n_parts])   # virial

sr_12 = np.zeros([n_parts, n_parts])
sr_6 = np.zeros([n_parts, n_parts])
sr12_new = np.zeros([n_parts, n_parts])
sr6_new = np.zeros([n_parts, n_parts])

u, w = lennard_jones(part_locs, u, w, part_moved=None)


# long range corrections
P_LRC = 32/9 * np.pi * density ** 2 * dist_cutoff ** (-9) - \
                16/3* np.pi * density ** 2 * dist_cutoff ** (-3)
E_LRC = 8/9 * np.pi * n_parts * density * dist_cutoff ** (-9) - \
                8/3 * np.pi * n_parts * density * dist_cutoff ** (-3)
E = np.sum(u) + E_LRC

steps_between = 50
pf_step = []
pf_part = []
pf_vol = []
pf = np.zeros(steps_between)

vol_delts = []
P_out = []
E_out = []
step_out = []

H = []

box_dim = []
step_count = 0
total_step_count = 0
for step_info in steps_def:
    start = step_info[0]
    stop = step_info[1]
    for n in np.arange(start, stop, 1):
        if step_info[2] == 'part only':
            rand_num = 0.7
        elif step_info[2] == 'vol only':
            rand_num = 0.4
        else:
            rand_num = np.random.rand()

        if rand_num >= 0.5:
            # move random particle
            move_type = 'part'
            rand_part = int(np.random.rand() * n_parts)
            rand_vect = np.random.rand(n_dims)
            new_loc = part_locs[rand_part,:] + delta_rmax * (2 * rand_vect - 1) 

            # update loc if out of box
            new_loc = new_loc - np.floor(new_loc / box_size) * box_size

            new_locs = np.copy(part_locs)
            new_locs[rand_part,:] = new_loc

            # update energy and virial with new location:
            u_new, w_new = lennard_jones(new_locs, np.copy(u), np.copy(w), rand_part) 

            delta_u = np.sum(u_new[rand_part,:] - u[rand_part,:])
            E_new = E + delta_u
            
            alpha = np.exp(-beta * delta_u)


        else:
            # volume move
            move_type = 'vol'
            
            box_size_new = np.copy(box_size) + delta_Vmax * (2 * np.random.rand() - 1)
            
            #### Box_size too small?
            if (box_size_new < min_size).any():
                n_steps -= 1
                continue
            
            V_new = np.product(box_size_new)
            density_new = n_parts / V_new
            
            # update particle locations:
            new_locs = np.copy(part_locs) * box_size_new / box_size

            # calculate new potential and virial for all particles
            sr12_new = np.copy(sr_12) * 1 / (box_size_new[0] / box_size[0])
            sr6_new = np.copy(sr_6) * 1 / (box_size_new[0] / box_size[1])

            u_new = (sr12_new - sr6_new) * 4 * epsilon
            w_new = (sr6_new - 2 * sr12_new) * 24 * epsilon 

            # long range corrections
            P_LRC_new = 32/9 * np.pi * density_new ** 2 * dist_cutoff ** (-9) - \
                            16/3* np.pi * density_new ** 2 * dist_cutoff ** (-3)
            E_LRC_new = 8/9 * np.pi * n_parts * density_new * dist_cutoff ** (-9) - \
                            8/3 * np.pi * n_parts * density_new * dist_cutoff ** (-3)
            
            E_new = np.sum(u_new) + E_LRC_new
            delta_u = E_new - E
            
            delta_H = delta_u + P * (V_new - V) - n_parts * (1 / beta) * np.log(V_new / V)
            H.append(delta_H)
            alpha = np.exp(-beta * delta_H)

    

        # accept or reject move?
        prob = min([alpha, 1])
        accept = True
        if prob < 1:
            rand_num = np.random.rand()
            if rand_num > prob:
                accept = False

        if accept:
            part_locs = new_locs
            u = u_new
            w = w_new
            E = E_new
            sr_12 = sr12_new
            sr_6 = sr6_new
            if move_type == 'vol':
                box_size = np.copy(box_size_new)
                V = V_new
                density = density_new
                P_LRC = P_LRC_new
                E_LRC = E_LRC_new
                
                pf[step_count] = 3   

            else:

                pf[step_count] = 1
                

            
        else:   # failure
            if move_type == 'vol':
                pf[step_count] = 2

            else:
                pf[step_count] = 0
        


        step_count += 1
        if step_count >= steps_between:
            
            virial = -np.sum(w) / 3
            P_inst = density / beta + virial / V + P_LRC
            P_out.append(P_inst)
            E_out.append(E)
            box_dim.append(box_size[0])
            
            
            # calculate the pass-fail rate of particle moves in chunk
            pf_part_chunk = pf[np.where(pf < 2)]
            part_rate = np.sum(pf_part_chunk)/len(pf_part_chunk)
            
            # calculate the pass-fail rate of volume moves in chunk
            pf_vol_chunk = pf[np.where(pf > 1)] - 2
            vol_rate = np.sum(pf_vol_chunk)/len(pf_vol_chunk)
            
            pf_step.append(total_step_count)
            pf_part.append(part_rate)
            pf_vol.append(vol_rate)
            
            # Adjust delta_rmax
            if part_rate > target_acceptance_range[1]:
                delta_rmax *= 1.03
            elif part_rate < target_acceptance_range[0]:
                delta_rmax *= 0.97
            if delta_rmax < delta_rmax_min:
                delta_rmax = delta_rmax_min
                
            # Adjust delta_Vmax
            if vol_rate > target_acceptance_range[1]:
                delta_Vmax *= 1.03
            elif part_rate < target_acceptance_range[0]:
                vol_rate *= 0.97
            if delta_Vmax < delta_Vmax_min:
                delta_Vmax = delta_Vmax_min
            
            step_count = 0
          
            
            
        total_step_count += 1
        
# if total_step_count > 10:
#             print(pf)
#             break
        


## Plotting of results
plt.figure(figsize=[10,7])
ax_l = plt.subplot(221)
E_plot = plt.plot(pf_step, E_out, label='Energy')

plt.xlabel('Step')
plt.ylabel('Energy')

ax_r = ax_l.twinx()
P_plot = plt.plot(pf_step, P_out,'C3', label='Pressure')

plt.ylabel('Pressure')

lns = E_plot + P_plot
labs = [l.get_label() for l in lns]
ax_l.legend(lns, labs, loc='best', frameon=False)



plt.subplot(222)

# boxcar averaging of acceptance rate
boxcar_stepsize=10
sum_part = 0
sum_vol = 0
part_box = []
vol_box = []
step_box = []
for i in range(len(pf_part)):
    sum_part += pf_part[i]
    sum_vol += pf_vol[i]
    if i % boxcar_stepsize == 0:
        part_box.append(sum_part / boxcar_stepsize)
        vol_box.append(sum_vol / boxcar_stepsize)
        step_box.append(pf_step[i])
        
        sum_part = 0
        sum_vol = 0

plt.plot(step_box, part_box, 'C0', label='Particle')
plt.plot(step_box, vol_box, 'C3', label='Volume')

plt.ylabel('Acceptance Ratio')
plt.xlabel('Step')
plt.legend(loc='best',frameon=False)


plt.subplot(223)
plt.plot(pf_step, box_dim)
plt.ylabel('Box Length')

plt.tight_layout()
plt.savefig('MC2.png',dpi=300)
plt.show()   
print(delta_rmax, delta_Vmax)
