# openmx3.9.9_gpu
## What does this code do?
This is a GPU-accelerated version of [OpenMX](https://www.openmx-square.org/), a first-principles calculation code based on numerical atomic orbitals (NAO). Currently, only certain processes (matrix multiplication and eigenvalue problem processing) for band calculations (collinear and non-collinear) and cluster calculations (collinear and non-collinear) are GPU-accelerated.

## Code author
Hiroyuki Kawai (Niigata Univ.)</br>
X account: [@dc1394](https://x.com/dc1394)

## How to enable GPU acceleration
To enable GPU acceleration, you must specify "cusolver" for "scf.eigen.lib" in the input file (**:warning: GPU acceleration is disabled by default!**). For example:

```ini
scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  300.0       # default=300 (K)
scf.energycutoff           150.0       # default=150 (Ry)
scf.maxIter                 40         # default=40
scf.EigenvalueSolver       band        # DC|GDC|Cluster|Band
scf.Kgrid                  9 9 9       # means n1 x n2 x n3
scf.Mixing.Type           rmm-diisk    # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.30        # default=0.30
scf.Min.Mixing.Weight      0.001       # default=0.001 
scf.Max.Mixing.Weight      0.700       # default=0.40 
scf.Mixing.History          7          # default=5
scf.Mixing.StartPulay       5          # default=6
scf.criterion             1.0e-10      # default=1.0e-6 (Hartree) 
scf.eigen.lib             cusolver     # default=elpa1
```

Set as shown above.

## Build and install
Building and installing is more difficult than with standard OpenMX. The build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) and OpenMPI. The Makefile contains build examples for several supercomputer systems; please refer to them. If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

## Docker image
I have released the OpenMX 3.9.9 GPU Docker image.
You can easily try OpenMX 3.9.9 GPU on computers equipped with NVIDIA GPUs.
The steps are as follows:
1. Install [Docker](https://docs.docker.com/get-started/get-docker/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Run: `docker run --gpus all --shm-size=4gb --rm -it -v /path/to/inputs:/work dc1394/openmx3.9.9-gpu-ubuntu24.04:0.1`. Ensure `/path/to/inputs` is created beforehand.
4. Run tests with `cd openmx_work` and `mpirun -np 4 ./openmx -runtest`.

This should yield results like the following:

```ini
  1  input_example/Benzene.dat        Elapsed time(s)=    6.43  diff Utot= 0.000000000044  diff Force= 0.000000000008
  2  input_example/C60.dat            Elapsed time(s)=  49.17  diff Utot= 0.000000000002  diff Force= 0.000000000001
  3  input_example/CO.dat            Elapsed time(s)=    9.99  diff Utot= 0.000000000072  diff Force= 0.000000001358
  4  input_example/Cr2.dat            Elapsed time(s)=    8.04  diff Utot= 0.000000000439  diff Force= 0.000000000049
  5  input_example/Crys-MnO.dat      Elapsed time(s)=  80.04  diff Utot= 0.000000000038  diff Force= 0.000000002306
  6  input_example/GaAs.dat          Elapsed time(s)=  101.35  diff Utot= 0.000000000021  diff Force= 0.000000000003
  7  input_example/Glycine.dat        Elapsed time(s)=    5.52  diff Utot= 0.000000000001  diff Force= 0.000000000001
  8  input_example/Graphite4.dat      Elapsed time(s)=    7.52  diff Utot= 0.000000000019  diff Force= 0.000000000005
  9  input_example/H2O-EF.dat        Elapsed time(s)=    5.11  diff Utot= 0.000000000105  diff Force= 0.000000000002
  10  input_example/H2O.dat            Elapsed time(s)=    4.96  diff Utot= 0.000000000102  diff Force= 0.000000001624
  11  input_example/HMn.dat            Elapsed time(s)=  12.61  diff Utot= 0.000000000345  diff Force= 0.000000000011
  12  input_example/Methane.dat        Elapsed time(s)=    4.35  diff Utot= 0.000000000006  diff Force= 0.000000000001
  13  input_example/Mol_MnO.dat        Elapsed time(s)=    8.89  diff Utot= 0.000000000584  diff Force= 0.000000000068
  14  input_example/Ndia2.dat          Elapsed time(s)=  23.17  diff Utot= 0.000000000000  diff Force= 0.000000000001


Total elapsed time (s)      327.16
```

You can verify that the calculation is correct.

## Benchmarks
For benchmarks of GPU-accelerated OpenMX, please refer to the following literature.
https://journals.jps.jp/doi/10.7566/JPSJ.94.124003

However, the current version offers improved performance compared to the version described in this paper.

## Important notes
At present, GPU-accelerated OpenMX performs faster than standard OpenMX for calculations involving systems containing hundreds of atoms. For calculations involving systems with fewer than a hundred atoms, standard OpenMX should be used. Please use with caution as it may contain bugs.

## About bug reports
I would appreciate it if you could actively report any bugs. Please report them via GitHub issues or send them to [my X account](https://x.com/dc1394). Bug reports sent to my X account can be in English.