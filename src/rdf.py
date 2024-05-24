import numpy as np
from time import time
from mpi4py import MPI
import argparse
import os
import ase.io

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", type=str, help="task directory", default=".")
  parser.add_argument("--ndiscard", type=int, default=0, help="the number of first samples to be discarded")
  parser.add_argument("--nsamp", type=int, help="the number of samples to use for the calculation of rdf")
  parser.add_argument("--atypes", type=int, nargs="*", help="atom types to be considered in the calculation of rdf; same as LAMMPS atom types")
  parser.add_argument("--traj", type=str, help="trajectory file")
  parser.add_argument("--rcut", type=float, help="cutoff radius for the calculation of rdf", default=6.0)
  parser.add_argument("--nbin", type=int, help="number of bins for the calculation of rdf", default=100)
  args = parser.parse_args()

  directory = args.dir
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  nbead = comm.size
  ibead = rank
  atypes = args.atypes

  if rank==0:
    print("Directory: ", directory)
    print("Number of beads: ", nbead)
    print("Atom types: ", atypes)

  assert len(atypes) == 1 or len(atypes) == 2, "Only one or two atom types are allowed!"

  t0 = time()

  if args.traj:
    traj = ase.io.read(os.path.join(directory, args.traj), index=":")
  else:
    traj = ase.io.read(os.path.join(directory, f"{ibead+1:02d}.rerun"), index=":")

  # determine atom types
  atoms0 = traj[0]
  types0 = atoms0.get_atomic_numbers()
  type_list = np.unique(types0)
  ntype = np.unique(types0).shape[0]
  type_num = np.zeros(ntype, dtype=int)
  for i in range(ntype):
      type_num[i] = np.sum(types0==type_list[i])

  # read trajectory; 
  nframe = len(traj)
  natom = len(traj[0])
  coords = np.zeros([nframe, natom, 3])
  prds = np.zeros([nframe, 3])
  for iframe, atoms in enumerate(traj):
      coords[iframe] = atoms.get_positions()
      # types[iframe] = atoms.get_atomic_numbers()
      prds[iframe] = atoms.get_cell().diagonal()
      types = atoms.get_atomic_numbers()
      assert np.all(types==types0), "Atom types are inconsistent along the trajectory!"

  ndiscard=args.ndiscard
  nsamp=args.nsamp
  if nsamp==None:
    nsamp=nframe-ndiscard
  nsub=int(nsamp/10)
  coords=coords[ndiscard:ndiscard+nsamp]

  prds=prds[ndiscard:ndiscard+nsamp]
  natom=coords.shape[1]
  # nO=int(natom/3)

  t2 = time()
  if rank==0:
    print("Reading samples costs %.4f s."%(t2-t0))

  rcut=args.rcut
  nbin=args.nbin
  dr=rcut/nbin
  r_array=np.linspace(0, rcut, num=nbin, endpoint=False)+0.5*dr
  r_array_10 = np.array([r_array for i in range(10)]).reshape(10, nbin)

  if len(atypes) == 1 or (len(atypes) == 2 and atypes[0] == atypes[1]):
    idx_A = np.where(types0==atypes[0])[0]
    coords_A = coords[:, idx_A]
  else:
    idx_A = np.where(types0==atypes[0])[0]
    idx_B = np.where(types0==atypes[1])[0]
    coords_A = coords[:, idx_A]
    coords_B = coords[:, idx_B]
  prds=prds.reshape(nsamp, 1, 1, 3)

  if len(atypes) == 1 or (len(atypes) == 2 and atypes[0] == atypes[1]):
    nA = type_num[np.where(type_list==atypes[0])[0][0]]
    dists_array = np.zeros([nsamp, int(nA*(nA-1)/2)])
  else:
    nA = type_num[np.where(type_list==atypes[0])[0][0]]
    nB = type_num[np.where(type_list==atypes[1])[0][0]]
    dists_array = np.zeros([nsamp, nA*nB])

  g_r_array = np.zeros([nsamp, nbin])

  nbatch = 100
  nloop = int(nsamp/nbatch)
  for iloop in range(nloop):
    t1 = time()
    if len(atypes) == 1 or (len(atypes) == 2 and atypes[0] == atypes[1]):
      coords_A_batch = coords_A[iloop*nbatch:(iloop+1)*nbatch]
      dist_batch = coords_A_batch[:, None, :] - coords_A_batch[:, :, None]
    else:
      coords_A_batch = coords_A[iloop*nbatch:(iloop+1)*nbatch]
      coords_B_batch = coords_B[iloop*nbatch:(iloop+1)*nbatch]
      dist_batch = coords_A_batch[:, None, :] - coords_B_batch[:, :, None]
    prds_batch = prds[iloop*nbatch:(iloop+1)*nbatch]
    dist_pbc=(dist_batch/prds_batch-np.floor(dist_batch/prds_batch+0.5))*prds_batch
    dist_r=np.sqrt((dist_pbc**2).sum(axis=3))
    
    t3 = time()
    if rank==0:
      print("Loop %d: computing pbc distances costs %.4f s."%(iloop, t3-t1))
    for ibatch in range(nbatch):
      isamp = iloop*nbatch + ibatch
      if len(atypes) == 1 or (len(atypes) == 2 and atypes[0] == atypes[1]):
        dists_array[isamp] = dist_r[ibatch][np.triu_indices(nA, 1)]
        Vol = prds[isamp][0, 0, 0]*prds[isamp][0, 0, 1]*prds[isamp][0, 0, 2]
        hist_r = np.histogram(dists_array[isamp], bins=nbin, range=(0, rcut), density=False)
        g_r_array[isamp] = 2*hist_r[0]/4/np.pi/r_array**2/dr/nA/(nA-1)*Vol
      else:
        dists_array[isamp] = dist_r[ibatch].reshape(nA*nB)
        Vol = prds[isamp][0, 0, 0]*prds[isamp][0, 0, 1]*prds[isamp][0, 0, 2]
        hist_r = np.histogram(dists_array[isamp], bins=nbin, range=(0, rcut), density=False)
        g_r_array[isamp] = hist_r[0]/4/np.pi/r_array**2/dr/nA/nB*Vol
      if rank==0:
        if (isamp+1)%100==0:
          t4 = time()
          print("Computing rdf for %d samples costs %.4f s."%(isamp+1, t4-t0))

  g_r_bead = np.zeros([10, nbin]) 
  for isub in range(10):
    g_r_bead[isub] = g_r_array[isub*nsub:(isub+1)*nsub].mean(axis=0)

  g_r_beads = comm.gather(g_r_bead, root=0)

  if rank==0:
    g_r = (np.array(g_r_beads, dtype="float").reshape(nbead, 10, nbin)).mean(axis=0)
    g_output = np.c_[r_array_10.reshape(10*nbin), g_r.reshape(10*nbin)].reshape(10, nbin, 2)
    if len(atypes) == 1:
      np.save(os.path.join(directory, f"g_{atypes[0]}_{atypes[0]}.npy"), g_output)
    else:
      np.save(os.path.join(directory, f"g_{atypes[0]}_{atypes[1]}.npy"), g_output)

if __name__ == "__main__":
  main()