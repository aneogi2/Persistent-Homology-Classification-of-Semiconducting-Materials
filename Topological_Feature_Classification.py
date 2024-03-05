#Import Modules
import numpy as np
import MDAnalysis as mda
import gudhi
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Read Lammps Datafile
def read_lammps_data(filename):
    coordinates = []
    atoms = open(filename,"r").readlines()
    for i,line in enumerate(atoms):
        atom_col = line.split()
        if len(atom_col) ==5:
            atom_col = [float(val) for val in atom_col]
        #print(col)
            coordinates.append([float(atom_col[2]),float(atom_col[3]),float(atom_col[4])])
    return np.array(coordinates)

def compute_persistence(data):
    """Computes persistence homology from atomic coordinates."""
    rips_complex = gudhi.RipsComplex(points=data)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence()
    return diag

def compute_persistence_density(diag, num_bins=100):
    """Computes persistence density from persistence diagram."""
    persistence = [p[1] - p[0] for p in diag]
    max_persistence = max(persistence)
    min_persistence = min(persistence)
    bin_size = (max_persistence - min_persistence) / num_bins

    hist, _ = np.histogram(persistence, bins=num_bins, range=(min_persistence, max_persistence))
    persistence_density = hist / float(len(persistence))
    return persistence_density

#Plotting the Persistence Diagram

def plot_persistence_density(persistence_intervals, title, cmap, vmin, vmax):
    birth_values = []
    death_values = []
    persistence_values = []

    for interval in persistence_intervals:
        birth = interval[0]
        death = interval[1]
        if np.isfinite(birth) and np.isfinite(death):
            birth_values.append(birth)
            death_values.append(death)
            persistence_values.append(death - birth)

    plt.scatter(birth_values, death_values, c=persistence_values, cmap=cmap, vmin=vmin, vmax=vmax, label=title)
