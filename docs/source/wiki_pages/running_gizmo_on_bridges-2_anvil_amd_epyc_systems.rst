Notes for running GIZMO on Bridges-2
------------------------------------

to login to Bridges-2: ssh -XY -p 2222 <your username>@bridges2.psc.edu

to transfer files to Bridges-2: scp -p 2222 <your username>@bridges2.psc.edu:/jet/home/<your_username>

Bridges-2 does not have regular scratch space. You have your $PROJECT space which has a certain storage limit but is not purged.

squeue -u <your username> to see queued jobs

Modules to load in .bashrc (assuming you use the AMD compiler, note that intel and GNU are also available, see module avail): module load mkl module load hdf5 module load fftw module load aocc module load openmpi/4.0.2-clang2.1

Getting the right thread affinity in hybrid jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that hybrid MPI/openMP jobs on Bridges-2 will not automatically allocate the correct processor affinity without an explicit command to do so. A Bridges-2 node has 128 physical cores across 2 sockets, and simply compiling with OPENMP=2 and setting OMP_NUM_THREADS=2 in the submit script will create a situation where each of the 64 MPI ranks per node only gets assigned a single core, and its 2 threads fight over that one core (which is bad). A correct setup sets the affinity of MPI rank n to cores 2n and 2n+1, e.g. MPI rank 0 gets cores 0-1, rank 1 gets cores 2-3, … rank 63 gets cores 126-127. The only way I have figured out how to do this is by spelling it out explicitly in a “rankfile”, given below. There is probably a more elegant way but I haven’t found it.

Example submit script
^^^^^^^^^^^^^^^^^^^^^

Here we run a job with 256 MPI ranks with 2 openMP threads each. Note the additional -rankfile argument to get the thread affinity right. Note that this is not necessary for pure MPI jobs.

.. code:: bash

   #!/bin/bash                                                                                      
   #SBATCH -N 4                                                                                                                                
   #SBATCH --ntasks-per-node=128                                                                                                                                                                                 
   #SBATCH -t 48:00:00                                                                                                                                                                                           
   #SBATCH -A ast200008p                                                                                                                                                                                               
   source $HOME/.bashrc
   export OMP_NUM_THREADS=2
   mpirun -np 256 -rankfile rankfile_4_2.txt ./GIZMO ./params.txt 0 1>gizmo.out 2>gizmo.err &
   wait

Example rankfile for a run with 4 nodes with 256 MPI ranks allocated 2 openMP threads each, with each thread getting 2 physical cores out of the 128 cores per node. Each line has the form rank <MPI rank>=+n<node number> slot=<2*MPI rank-2*MPI rank+1>. See script below for generating this.

::

   rank 0=+n0 slot=0-1
   rank 1=+n0 slot=2-3
   rank 2=+n0 slot=4-5
   rank 3=+n0 slot=6-7
   rank 4=+n0 slot=8-9
   rank 5=+n0 slot=10-11
   rank 6=+n0 slot=12-13
   rank 7=+n0 slot=14-15
   rank 8=+n0 slot=16-17
   rank 9=+n0 slot=18-19
   rank 10=+n0 slot=20-21
   rank 11=+n0 slot=22-23
   rank 12=+n0 slot=24-25
   rank 13=+n0 slot=26-27
   rank 14=+n0 slot=28-29
   rank 15=+n0 slot=30-31
   rank 16=+n0 slot=32-33
   rank 17=+n0 slot=34-35
   rank 18=+n0 slot=36-37
   rank 19=+n0 slot=38-39
   rank 20=+n0 slot=40-41
   rank 21=+n0 slot=42-43
   rank 22=+n0 slot=44-45
   rank 23=+n0 slot=46-47
   rank 24=+n0 slot=48-49
   rank 25=+n0 slot=50-51
   rank 26=+n0 slot=52-53
   rank 27=+n0 slot=54-55
   rank 28=+n0 slot=56-57
   rank 29=+n0 slot=58-59
   rank 30=+n0 slot=60-61
   rank 31=+n0 slot=62-63
   rank 32=+n0 slot=64-65
   rank 33=+n0 slot=66-67
   rank 34=+n0 slot=68-69
   rank 35=+n0 slot=70-71
   rank 36=+n0 slot=72-73
   rank 37=+n0 slot=74-75
   rank 38=+n0 slot=76-77
   rank 39=+n0 slot=78-79
   rank 40=+n0 slot=80-81
   rank 41=+n0 slot=82-83
   rank 42=+n0 slot=84-85
   rank 43=+n0 slot=86-87
   rank 44=+n0 slot=88-89
   rank 45=+n0 slot=90-91
   rank 46=+n0 slot=92-93
   rank 47=+n0 slot=94-95
   rank 48=+n0 slot=96-97
   rank 49=+n0 slot=98-99
   rank 50=+n0 slot=100-101
   rank 51=+n0 slot=102-103
   rank 52=+n0 slot=104-105
   rank 53=+n0 slot=106-107
   rank 54=+n0 slot=108-109
   rank 55=+n0 slot=110-111
   rank 56=+n0 slot=112-113
   rank 57=+n0 slot=114-115
   rank 58=+n0 slot=116-117
   rank 59=+n0 slot=118-119
   rank 60=+n0 slot=120-121
   rank 61=+n0 slot=122-123
   rank 62=+n0 slot=124-125
   rank 63=+n0 slot=126-127
   rank 64=+n1 slot=0-1
   rank 65=+n1 slot=2-3
   rank 66=+n1 slot=4-5
   rank 67=+n1 slot=6-7
   rank 68=+n1 slot=8-9
   rank 69=+n1 slot=10-11
   rank 70=+n1 slot=12-13
   rank 71=+n1 slot=14-15
   rank 72=+n1 slot=16-17
   rank 73=+n1 slot=18-19
   rank 74=+n1 slot=20-21
   rank 75=+n1 slot=22-23
   rank 76=+n1 slot=24-25
   rank 77=+n1 slot=26-27
   rank 78=+n1 slot=28-29
   rank 79=+n1 slot=30-31
   rank 80=+n1 slot=32-33
   rank 81=+n1 slot=34-35
   rank 82=+n1 slot=36-37
   rank 83=+n1 slot=38-39
   rank 84=+n1 slot=40-41
   rank 85=+n1 slot=42-43
   rank 86=+n1 slot=44-45
   rank 87=+n1 slot=46-47
   rank 88=+n1 slot=48-49
   rank 89=+n1 slot=50-51
   rank 90=+n1 slot=52-53
   rank 91=+n1 slot=54-55
   rank 92=+n1 slot=56-57
   rank 93=+n1 slot=58-59
   rank 94=+n1 slot=60-61
   rank 95=+n1 slot=62-63
   rank 96=+n1 slot=64-65
   rank 97=+n1 slot=66-67
   rank 98=+n1 slot=68-69
   rank 99=+n1 slot=70-71
   rank 100=+n1 slot=72-73
   rank 101=+n1 slot=74-75
   rank 102=+n1 slot=76-77
   rank 103=+n1 slot=78-79
   rank 104=+n1 slot=80-81
   rank 105=+n1 slot=82-83
   rank 106=+n1 slot=84-85
   rank 107=+n1 slot=86-87
   rank 108=+n1 slot=88-89
   rank 109=+n1 slot=90-91
   rank 110=+n1 slot=92-93
   rank 111=+n1 slot=94-95
   rank 112=+n1 slot=96-97
   rank 113=+n1 slot=98-99
   rank 114=+n1 slot=100-101
   rank 115=+n1 slot=102-103
   rank 116=+n1 slot=104-105
   rank 117=+n1 slot=106-107
   rank 118=+n1 slot=108-109
   rank 119=+n1 slot=110-111
   rank 120=+n1 slot=112-113
   rank 121=+n1 slot=114-115
   rank 122=+n1 slot=116-117
   rank 123=+n1 slot=118-119
   rank 124=+n1 slot=120-121
   rank 125=+n1 slot=122-123
   rank 126=+n1 slot=124-125
   rank 127=+n1 slot=126-127
   rank 128=+n2 slot=0-1
   rank 129=+n2 slot=2-3
   rank 130=+n2 slot=4-5
   rank 131=+n2 slot=6-7
   rank 132=+n2 slot=8-9
   rank 133=+n2 slot=10-11
   rank 134=+n2 slot=12-13
   rank 135=+n2 slot=14-15
   rank 136=+n2 slot=16-17
   rank 137=+n2 slot=18-19
   rank 138=+n2 slot=20-21
   rank 139=+n2 slot=22-23
   rank 140=+n2 slot=24-25
   rank 141=+n2 slot=26-27
   rank 142=+n2 slot=28-29
   rank 143=+n2 slot=30-31
   rank 144=+n2 slot=32-33
   rank 145=+n2 slot=34-35
   rank 146=+n2 slot=36-37
   rank 147=+n2 slot=38-39
   rank 148=+n2 slot=40-41
   rank 149=+n2 slot=42-43
   rank 150=+n2 slot=44-45
   rank 151=+n2 slot=46-47
   rank 152=+n2 slot=48-49
   rank 153=+n2 slot=50-51
   rank 154=+n2 slot=52-53
   rank 155=+n2 slot=54-55
   rank 156=+n2 slot=56-57
   rank 157=+n2 slot=58-59
   rank 158=+n2 slot=60-61
   rank 159=+n2 slot=62-63
   rank 160=+n2 slot=64-65
   rank 161=+n2 slot=66-67
   rank 162=+n2 slot=68-69
   rank 163=+n2 slot=70-71
   rank 164=+n2 slot=72-73
   rank 165=+n2 slot=74-75
   rank 166=+n2 slot=76-77
   rank 167=+n2 slot=78-79
   rank 168=+n2 slot=80-81
   rank 169=+n2 slot=82-83
   rank 170=+n2 slot=84-85
   rank 171=+n2 slot=86-87
   rank 172=+n2 slot=88-89
   rank 173=+n2 slot=90-91
   rank 174=+n2 slot=92-93
   rank 175=+n2 slot=94-95
   rank 176=+n2 slot=96-97
   rank 177=+n2 slot=98-99
   rank 178=+n2 slot=100-101
   rank 179=+n2 slot=102-103
   rank 180=+n2 slot=104-105
   rank 181=+n2 slot=106-107
   rank 182=+n2 slot=108-109
   rank 183=+n2 slot=110-111
   rank 184=+n2 slot=112-113
   rank 185=+n2 slot=114-115
   rank 186=+n2 slot=116-117
   rank 187=+n2 slot=118-119
   rank 188=+n2 slot=120-121
   rank 189=+n2 slot=122-123
   rank 190=+n2 slot=124-125
   rank 191=+n2 slot=126-127
   rank 192=+n3 slot=0-1
   rank 193=+n3 slot=2-3
   rank 194=+n3 slot=4-5
   rank 195=+n3 slot=6-7
   rank 196=+n3 slot=8-9
   rank 197=+n3 slot=10-11
   rank 198=+n3 slot=12-13
   rank 199=+n3 slot=14-15
   rank 200=+n3 slot=16-17
   rank 201=+n3 slot=18-19
   rank 202=+n3 slot=20-21
   rank 203=+n3 slot=22-23
   rank 204=+n3 slot=24-25
   rank 205=+n3 slot=26-27
   rank 206=+n3 slot=28-29
   rank 207=+n3 slot=30-31
   rank 208=+n3 slot=32-33
   rank 209=+n3 slot=34-35
   rank 210=+n3 slot=36-37
   rank 211=+n3 slot=38-39
   rank 212=+n3 slot=40-41
   rank 213=+n3 slot=42-43
   rank 214=+n3 slot=44-45
   rank 215=+n3 slot=46-47
   rank 216=+n3 slot=48-49
   rank 217=+n3 slot=50-51
   rank 218=+n3 slot=52-53
   rank 219=+n3 slot=54-55
   rank 220=+n3 slot=56-57
   rank 221=+n3 slot=58-59
   rank 222=+n3 slot=60-61
   rank 223=+n3 slot=62-63
   rank 224=+n3 slot=64-65
   rank 225=+n3 slot=66-67
   rank 226=+n3 slot=68-69
   rank 227=+n3 slot=70-71
   rank 228=+n3 slot=72-73
   rank 229=+n3 slot=74-75
   rank 230=+n3 slot=76-77
   rank 231=+n3 slot=78-79
   rank 232=+n3 slot=80-81
   rank 233=+n3 slot=82-83
   rank 234=+n3 slot=84-85
   rank 235=+n3 slot=86-87
   rank 236=+n3 slot=88-89
   rank 237=+n3 slot=90-91
   rank 238=+n3 slot=92-93
   rank 239=+n3 slot=94-95
   rank 240=+n3 slot=96-97
   rank 241=+n3 slot=98-99
   rank 242=+n3 slot=100-101
   rank 243=+n3 slot=102-103
   rank 244=+n3 slot=104-105
   rank 245=+n3 slot=106-107
   rank 246=+n3 slot=108-109
   rank 247=+n3 slot=110-111
   rank 248=+n3 slot=112-113
   rank 249=+n3 slot=114-115
   rank 250=+n3 slot=116-117
   rank 251=+n3 slot=118-119
   rank 252=+n3 slot=120-121
   rank 253=+n3 slot=122-123
   rank 254=+n3 slot=124-125
   rank 255=+n3 slot=126-127

| 
| `Example python script for generating the above rankfile. <https://www.github.com/mikegrudic/make_rankfile>`__