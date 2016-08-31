"""
    2D Molecular Dynamics simulation that models Lennard-Jones particles with periodic boundaries
    
    Author:  Michael Cowan
"""



import numpy as np
import matplotlib.pyplot as plt
from time import time



# All values taken as estimates mainly from [Lecture 2, slide 3]
eps = 1.65E-21        # J/molec
sigma = 3.40E-10      # m
kb = 1.38064852E-23   # J/(molec*K)
temp = 100            # K
m = 6.63E-26          # kg/molec
t = 1E-14             # s
box = 1E-8            # m
n = 225               # molecules
atom = 96             # atom whose trajectory will be plotted


#Dimensionless parameters
temp_dless = kb * temp / eps
m_dless = 1.0
t_dless = t / (sigma * np.sqrt(m / eps))
box_dless = box / sigma
half = round(box_dless / 2.0, 3)
dt = t_dless
cut = 2.              # cut-off r length used in LJ force calc
cut_potential = 4.0 * ((cut ** -12) - (cut ** -6))



#Creates an even distribution of atoms on a cubic lattice (sqrt(n) must = integer)
def lattice():

    space = np.linspace(0, box_dless, np.sqrt(n) + 1)
    space = space[:-1]
    diff = (box_dless - space[-1]) / 2.0
    pos = np.zeros([2, n])
    i = j = k = 0                  

    while k < np.sqrt(n):
        pos[0, i] = space[j] + diff
        pos[1, i] = space[k] + diff
        i += 1
        j += 1
        if j == np.sqrt(n):
            j = 0
        if i % np.sqrt(n) == 0:
            k += 1

    return pos


#Initializes velocities, positions, & "old" positions
def initialize():
    pos = lattice()                                             #placing atoms on cubic lattice
    vel = np.random.random([2, n]) - 0.5                        #random vel : [-0.5, 0.5]

    px = vel[0].sum() * m_dless / n
    py = vel[1].sum() * m_dless / n

    for i in xrange(n):
        vel[0, i] -= px
        vel[1, i] -= py                                         #Shifts center-of-mass momentum to 0 [Lecture 11, slide 29]

    norm_x = ((1.0 / (2 * n)) * ((vel[0] ** 2).sum()))          #Divide by 2N degrees of freedom
    norm_y = ((1.0 / (2 * n)) * ((vel[1] ** 2).sum())) 

    old_pos = np.zeros([2, n])

    for i in xrange(n):
        vel[0, i] *= (np.sqrt(temp_dless / norm_x))
        vel[1, i] *= (np.sqrt(temp_dless / norm_y))             #Scale so <vi*^2> = T*
        old_pos[0, i] = pos[0, i] - vel[0, i] * dt
        old_pos[1, i] = pos[1, i] - vel[1, i] * dt

    return pos,vel,old_pos




#LJ Pairwise additive force [Lecture 11, slide 24] [Lecture 13, slide 9]
def force(pos):
    f = np.zeros([2,n])
    potential = 0
    d = np.zeros(2)

    for i in xrange(n - 1):

        for j in xrange(i + 1, n):
            d = np.zeros(2)

            for k in xrange(2):
                d[k] = pos[k, i] - pos[k, j]

                if d[k] > half:                                 #Central Image Algorithm
                    d[k] = d[k] - box_dless

                elif d[k] < -half:
                    d[k] = d[k] + box_dless

            r = np.sqrt(np.dot(d, d))
            if r < cut:
                potential += 4.0 * ((r ** -12) - (r ** -6))

                #f=(48/r^2)*(1/r^12 - 1/2r^6)
                ff = (r ** -2) * (r ** -12) - (0.5 * (r ** -6))
                f[:, i] = f[:, i] + (d * ff)
                f[:, j] = f[:, j] - (d * ff)

    return 48. * f, (potential / (2 * n)) + cut_potential



#Calculates the kinetic energy based on velocities
def kin(vel):
    v2 = 0

    for i in xrange(n):
        v2 += np.dot(vel[:, i], vel[:, i])                      #v_r = sqrt(vx^2 + vy^2)  so  v_r^2 = vx^2 + vy^2

    return (0.5 * m_dless * v2) / (2 * n)                       #Returns kinetic energy per atom of system



#Verlet algorithm used to determine new positions from forces
def verlet(old_pos, pos, f, check):

    for i in xrange(n):

        for k in range(2):
            if check[k, i] == 1:
                old_pos[k, i] -= box_dless

            elif check[k, i] == 2:
                old_pos[k, i] += box_dless

    new_pos = 2 * pos - old_pos + ((1 / m_dless) * f * (dt ** 2))
    vel=(new_pos-old_pos)/(2.0*dt)                              #Finite difference method to find velocity
    kinetic = kin(vel)

    return new_pos, pos, kinetic



#Velocity-Verlet algorithm used to determine new positions and velocities from force    
def velocity_verlet(pos, vel, f):
    new_pos = pos + vel * dt + 0.5 * f * (dt ** 2)
    new_pos, check = periodic(new_pos)
    f_n, potential = force(new_pos)
    new_vel = vel + 0.5 * (f + f_n) * dt

    return new_pos, new_vel, f_n, potential


    
#Second Order Runge-Kutta algorithm
def runge_kutta(pos, vel, f):
    half_pos = pos + 0.5 * vel * dt
    new_pos = pos + dt * (vel + (0.5 / m_dless) * f * dt)
    half_f, half_potential = force(half_pos)
    new_vel = vel + (dt / m_dless) * half_f

    return new_pos, new_vel



#Converts coordinates to create periodic boundaries
def periodic(pos):
    check = np.zeros([2, n])

    for i in xrange(n):

        for k in range(2):
            if pos[k, i] > box_dless:
                pos[k, i] -= box_dless
                check[k, i] = 1

            if pos[k, i] < 0:
                pos[k, i] += box_dless
                check[k, i] = 2

    return pos,check



#Calculates the radial distribution function
def rdf(pos, box_dless):
    area = box_dless ** 2
    r_max = half
    num_den = n / area
    dr = 0.3
    hist = np.zeros(int(r_max / dr))

    for i in xrange(n - 1):

        for j in xrange(i + 1, n):
            r2 = np.zeros(2)

            for k in xrange(2):
                a = abs(pos[k, i] - pos[k, j])
                if a > r_max:
                    a = box_dless - a
                r2[k] = a ** 2
            r = np.sqrt(r2.sum())

            if r < r_max:
                if int(r / dr) == int(r_max / dr):
                    hist[int(r / dr)-1] += 2

                else:
                    hist[int(r / dr)] += 2

    for x in xrange(len(hist)):
        shell_area = np.pi * (((dr * (x + 1)) ** 2) - ((dr * x) ** 2))
        hist[x] = hist[x] / (n * num_den * shell_area)
    ddr = np.linspace(dr, r_max, r_max / dr)

    return ddr[:-1], hist[:-1]



#Runs simulation with verlet algorithm
def sim_verlet(steps, blocks):
    pos, vel, old_pos = initialize()
    pos, check = periodic(pos)
    f, potential = force(pos)
    kinetic = kin(vel)
    e_tot_0 = potential + kinetic
    gr_avg = ddr = 1

    print 'Total Energy of System (Per Atom)'.rjust(25)
    print 'Initial: '+str(round(e_tot_0, 6)).rjust(15)

    study = np.zeros([2, (blocks * (steps / blocks)) + 1])
    study[:, 0] = pos[:, atom]
    en = np.zeros(blocks + 1)
    en[0] = e_tot_0

    for s in xrange(blocks):
        e_tot = 0

        for x in xrange(steps / blocks):
            pos, old_pos, kinetic = verlet(old_pos, pos, f, check)
            pos, check = periodic(pos)
            e_tot += potential + kinetic
            study[:, (s * (steps / blocks)) + (x + 1)] = pos[:, atom]
            f, potential = force(pos)

        en[s + 1] = e_tot / (steps / blocks)
        ddr, gr = rdf(pos, box_dless)

        if s == 0:
            gr_avg = np.zeros(len(ddr))
        gr_avg += gr
        print 'Block ' + str(s+1) + ': ' + str(round(en[s + 1], 6)).rjust(15)

    gr_avg /= blocks

    return pos, study, en, gr_avg, ddr



#Runs simulation with velocity-verlet algorithm
def sim_vel_verlet(steps, blocks):
    pos, vel, old_pos = initialize()
    f, potential = force(pos)
    kinetic = kin(vel)
    e_tot_0 = potential + kinetic
    gr_avg = ddr = 1

    print 'Total Energy of System (Per Atom)'.rjust(25)
    print 'Initial: '+str(round(e_tot_0, 6)).rjust(15)

    study = np.zeros([2, (blocks * (steps / blocks)) + 1])
    study[:, 0] = pos[:, atom]
    en = np.zeros(blocks + 1)
    en[0] = e_tot_0

    for s in xrange(blocks):
        e_tot = 0

        for x in xrange(steps / blocks):
            pos, vel, f, potential = velocity_verlet(pos, vel, f)
            study[:, (s * (steps / blocks)) + (x + 1)] = pos[:, atom]
            kinetic = kin(vel)
            e_tot += potential + kinetic

        en[s + 1] = e_tot / (steps / blocks)
        ddr, gr = rdf(pos, box_dless)
        if s == 0:
            gr_avg = np.zeros(len(ddr))
        gr_avg += gr
        print 'Block ' + str(s+1) + ': ' + str(round(en[s + 1], 6)).rjust(15)
    gr_avg /= blocks

    return pos, study, en, gr_avg, ddr



#Runs simulation with Runge-Kutta algorithm    
def sim_runge_kutta(steps, blocks):
    pos, vel, old_pos = initialize()
    study = np.zeros([2, (blocks * (steps / blocks)) + 1])
    study[:, 0] = pos[:, atom]
    f, potential = force(pos)
    kinetic = kin(vel)
    e_tot_0 = potential + kinetic
    gr_avg = ddr = 1

    print 'Total Energy of System (Per Atom)'.rjust(25)
    print 'Initial: '+str(round(e_tot_0, 6)).rjust(15)

    en = np.zeros(blocks + 1)
    en[0] = e_tot_0

    for s in xrange(blocks):
        e_tot = 0

        for x in xrange(steps / blocks):
            pos, vel = runge_kutta(pos, vel, f)
            pos, check = periodic(pos)
            study[:, (s * (steps / blocks)) + (x + 1)] = pos[:, atom]
            f, potential = force(pos)
            kinetic = kin(vel)
            e_tot += potential + kinetic
        en[s + 1] = e_tot / (steps / blocks)
        ddr, gr = rdf(pos, box_dless)

        if s == 0:
            gr_avg = np.zeros(len(ddr))
        gr_avg += gr
        print 'Block ' + str(s+1) + ': ' + str(round(en[s + 1], 6)).rjust(15)
    gr_avg /= blocks

    return pos,study,en,gr_avg,ddr



#Adds labels to bar graph
def autolabel(rects, ax):
    # attach some text labels

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                format(round(height, 3), '.3f'),
                ha = 'center', va = 'bottom')



#Writes the results to the text file "fid"
def pr(fid, name, pos, study, en, gr_avg, ddr):
    fid.write('\n\n' + name + '\n\n')
    fid.write('Positions:\n')
    fid.write(str(pos))
    fid.write('\nStudy:\n')
    fid.write(str(study))
    fid.write('\nEnergy:\n')
    fid.write(str(en))
    fid.write('\nRDF:\n')
    fid.write(str(gr_avg))
    fid.write('\nr*:\n')
    fid.write(str(ddr))



#Wrapped function to run a simulation for verlet, velocity-verlet, and Runge-Kutta algorithms
def run(steps = 500, blocks = 100):
    fid = open('sim_results.txt','w')
    print '\n' + '-' * 35 + '\n\n'
    print 'Total Steps per Simulation: '+str(steps)+'\n\n'
    print '-' * 35 + '\n\n'
    print 'Running Verlet Algorithm.\n\n'


    ### Verlet ###
    start_verlet = time()

    pos1, study1, en1, gr_avg1, ddr1 = sim_verlet(steps, blocks)

    stop_verlet = time()

    pr(fid, 'Verlet Algorithm', pos1, study1, en1, gr_avg1, ddr1)
    print '\nVerlet complete.'
    print '\n\n' + '-' * 35 + '\n\n' + 'Running Velocity Verlet Algorithm.\n\n'


    ### Velocity-Verlet ###
    start_vel_verlet = time()

    pos2, study2, en2, gr_avg2, ddr = sim_vel_verlet(steps, blocks)

    stop_vel_verlet = time()

    pr(fid, 'Velocity Verlet Algorithm', pos2, study2, en2, gr_avg2, ddr)
    print '\nVelocity Verlet complete.'
    print '\n\n' + '-' * 35 + '\n\n' + 'Running Runge-Kutta Algorithm.\n\n'


    ### Runge-Kutta ###
    start_runge_kutta=time()

    pos3, study3, en3, gr_avg3, ddr = sim_runge_kutta(steps, blocks)

    stop_runge_kutta = time()

    pr(fid, 'Runge-Kutta Algorithm', pos3, study3, en3, gr_avg3, ddr)
    print '\nRunge-Kutta complete.'


    #Calculates runtime per step for each simulation
    time_verlet = (stop_verlet - start_verlet) / steps
    time_vel_verlet = (stop_vel_verlet - start_vel_verlet) / steps
    time_runge_kutta = (stop_runge_kutta - start_runge_kutta) / steps


    #Creates bar chart for runtimes
    figt = plt.figure('Time Per Step', figsize = (13,11))
    axt = figt.add_subplot(111)
    rects = axt.bar([1, 3, 5], [time_verlet, time_vel_verlet, time_runge_kutta], align = 'center')
    autolabel(rects, axt)
    axt.set_ylabel('average time per step (seconds)')
    axt.set_ylim(0, 1)
    axt.set_xticks([1, 3, 5])
    axt.set_xticklabels(['Verlet', 'Velocity Verlet', 'Runge-Kutta'])
    #figt.savefig('compare_time2.png')


    #Adds runtimes to the text file "fid"
    fid.write('\n\nTimes:\n')
    fid.write('Verlet: ' + str(time_verlet))
    fid.write('\nVelocity Verlet: ' + str(time_vel_verlet))
    fid.write('\nRunge-Kutta: ' + str(time_runge_kutta))
    fid.close()


    #Plots the radial distribution function versus radial distance for each simulation
    figr = plt.figure('Radial Distribution Function', figsize=(13, 11))
    axr = figr.add_subplot(111)
    axr.plot(ddr1, gr_avg1, ddr, gr_avg2, ddr, gr_avg3)
    axr.legend(['Verlet Algorithm', 'Velocity Verlet Algorithm', 'Runge-Kutta Algorithm'],
                    shadow = True, bbox_to_anchor = (1.02, 1.05), ncol = 3)
    axr.set_xlabel('r*')
    axr.set_ylabel('g(r)')
    #figr.savefig('compare_rdf2.png')


    #Plots the average energy (for each blocks) for all simulations
    fige = plt.figure('Energy Fluctuation', figsize = (13, 11))
    axe = fige.add_subplot(111)
    x = range(len(en1))
    axe.plot(x, en1, x, en2, x, en3)
    axe.legend(['Verlet Algorithm', 'Velocity Verlet Algorithm', 'Runge-Kutta Algorithm'],
                    shadow = True, bbox_to_anchor = (1.02, 1.05), ncol = 3)
    axe.set_xlabel('Block')
    axe.set_ylabel('average total dimensionless energy (per atom)')
    #fige.savefig('compare_energy2.png')


    #Plots the final position of atoms in verlet sim and the trajectory of atom tracked
    fig = plt.figure('Verlet MD', figsize = (13, 11))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, box_dless)
    ax.set_xlim(0, box_dless)
    ppos = np.array([np.delete(pos1[0], atom), np.delete(pos1[1], atom)])
    ax.plot(ppos[0, :], ppos[1, :], 'o', color = 'blue')
    ax.plot(pos1[0, atom], pos1[1, atom], 'o', study1[0], study1[1], color = 'green')
    ax.set_xticks([])
    ax.set_yticks([])
    #fig.savefig('verlet_sim.png')


    #Plots the final position of atoms in velocity-verlet sim and the trajectory of atom tracked
    fig2 = plt.figure('Velocity-Verlet MD', figsize = (13, 11))
    ax2 = fig2.add_subplot(111)
    ax2.set_ylim(0, box_dless)
    ax2.set_xlim(0, box_dless)
    ppos = np.array([np.delete(pos2[0], atom), np.delete(pos2[1], atom)])
    ax2.plot(ppos[0, :], ppos[1, :], 'o', color = 'blue')
    ax2.plot(pos2[0, atom], pos2[1, atom], 'o', study2[0], study2[1], color = 'green')
    ax2.set_xticks([])
    ax2.set_yticks([])
    #fig2.savefig('velocity_verlet_sim.png')

    #Plots the final position of atoms in Runge-Kutta sim and the trajectory of atom tracked
    fig3 = plt.figure('Runge-Kutta MD', figsize = (13, 11))
    ax = fig3.add_subplot(111)
    ax.set_ylim(0, box_dless)
    ax.set_xlim(0, box_dless)
    ppos = np.array([np.delete(pos3[0], atom), np.delete(pos3[1], atom)])
    ax.plot(ppos[0, :], ppos[1, :], 'o', color = 'blue')
    ax.plot(pos3[0, atom], pos3[1, atom], 'o', study3[0], study3[1], color = 'green')
    ax.set_xticks([])
    ax.set_yticks([])
    #fig3.savefig('runge_kutta_sim.png')
    plt.show()



if __name__ == '__main__':

#Sample run of program; 50 steps split into 5 blocks 
    run(50, 5)
