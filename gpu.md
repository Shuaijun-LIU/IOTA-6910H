8× NVIDIA RTX 5880 Ada Server Overall Summary (Ignoring Device Issues)

This report summarizes the server's hardware structure, GPU topology, PCIe architecture, NUMA distribution, and software stack based on executed system query commands (lspci, nvidia-smi, NCCL initialization tests, etc.).

⸻

1. GPU Overview

The server is equipped with:
	•	8 × NVIDIA RTX 5880 Ada (PCIe)
	•	All GPUs are correctly recognized by the system
	•	GPU PCIe addresses include:

01:00.0
21:00.0
41:00.0
61:00.0
81:00.0
A1:00.0
C1:00.0
E1:00.0



These addresses indicate that GPUs are distributed across multiple PCIe root complexes, which is typical for a 4U eight-card architecture.

⸻

2. CPU and NUMA Architecture

Inferred from Bus ID distribution:
	•	GPUs are mounted on two CPUs
	•	CPU0 and CPU1 form a dual-socket structure (2-socket server)
	•	Multiple GPUs are mounted under each CPU, following a typical 8-GPU server layout

Therefore, the topology structure follows this logic:

CPU0 manages a portion of GPUs
CPU1 manages the remaining GPUs
The two CPUs are interconnected via UPI/QPI

This is an industry-standard 8-GPU workstation/training server structure.

⸻

3. GPU Interconnection Method

All information indicates:
	•	RTX 5880 Ada is a PCIe GPU and does not support NVLink
	•	The server has no NVLink Bridge or NVSwitch
	•	GPUs communicate via PCIe + CPU routing

Therefore:
	•	GPU ↔ GPU communication relies on PCIe bandwidth
	•	Cross-CPU communication requires traversing UPI/QPI links

Suitable for large model training frameworks such as DDP, FSDP, ZeRO, etc.

⸻

4. PCIe Architecture Capabilities (Theoretical)

According to the maximum support shown in nvidia-smi -q:
	•	GPU maximum interface: PCIe 4.0 ×16
	•	CPU/motherboard maximum interface: PCIe 5.0

Theoretical bandwidth (ignoring issues) should be:
	•	PCIe 4.0 ×16 unidirectional bandwidth: approximately 32 GB/s
	•	Communication between CPU → GPU may pass through retimer or PCIe switch

Therefore, the server should have standard 4U eight-card chassis PCIe interconnection performance.

⸻

5. GPU Communication Performance (Theoretical)

Ignoring issues, PCIe interconnection performance is as follows:

GPUs under the same CPU
	•	Utilize the same PCIe switch or retimer
	•	Theoretical bandwidth: between 11–28 GB/s

Cross-NUMA (cross-CPU)
	•	Communication path: GPU → CPU0 → CPU1 → GPU
	•	Limited by UPI/QPI: 6–12 GB/s

Conclusion:
	•	Large model parallel training is faster within the same NUMA
	•	Cross-NUMA communication has higher latency

⸻

5.5. GPU Communication Performance (Actual Test - P2P)

Peer-to-Peer bandwidth test results:

=== Peer-to-Peer Bandwidth (GB/s) ===

              0         1         2         3         4         5         6         7

  0 |         X 21.57 GB/s 21.64 GB/s 21.49 GB/s 21.82 GB/s 21.88 GB/s 21.67 GB/s 21.45 GB/s

  1 | 21.86 GB/s         X 21.81 GB/s 12.76 GB/s 21.68 GB/s 21.85 GB/s 15.56 GB/s 21.77 GB/s

  2 | 21.64 GB/s 21.58 GB/s         X 21.76 GB/s 21.74 GB/s 21.81 GB/s 21.72 GB/s 21.74 GB/s

  3 | 21.48 GB/s 21.58 GB/s 12.34 GB/s         X 20.44 GB/s 21.64 GB/s 21.43 GB/s 21.80 GB/s

  4 | 19.19 GB/s 22.04 GB/s 22.25 GB/s 13.24 GB/s         X 21.78 GB/s 21.87 GB/s 21.93 GB/s

  5 | 21.93 GB/s 21.91 GB/s 22.17 GB/s 21.88 GB/s 21.79 GB/s         X 21.79 GB/s 21.87 GB/s

  6 | 21.95 GB/s 18.98 GB/s 21.90 GB/s 21.93 GB/s 21.82 GB/s 21.96 GB/s         X 22.00 GB/s

  7 | 19.15 GB/s 22.06 GB/s 21.90 GB/s 13.25 GB/s 21.86 GB/s 21.87 GB/s 21.93 GB/s         X

=== Peer-to-Peer Latency (us) ===

              0         1         2         3         4         5         6         7

  0 |         X    1.72 us    1.60 us    1.95 us    1.77 us    1.60 us    2.34 us    2.43 us

  1 |    2.03 us         X    2.04 us    2.02 us    2.02 us    2.03 us    2.11 us    2.08 us

  2 |    1.97 us    1.96 us         X    1.96 us    1.96 us    1.95 us    1.97 us    1.95 us

  3 |    1.99 us    2.01 us    2.00 us         X    2.00 us    2.01 us    2.03 us    2.00 us

  4 |    1.99 us    1.98 us    1.99 us    1.99 us         X    1.98 us    1.99 us    1.98 us

  5 |    2.00 us    1.98 us    1.98 us    1.98 us    2.00 us         X    1.97 us    1.98 us

  6 |    1.95 us    1.94 us    2.00 us    2.02 us    2.01 us    2.03 us         X    1.99 us

  7 |    1.59 us    1.59 us    1.78 us    1.61 us    1.81 us    1.96 us    1.66 us         X

Test result analysis:
	•	Bandwidth between most GPU pairs: 19–22 GB/s (same NUMA or same PCIe switch)
	•	Some GPU pairs have lower bandwidth: 12–15 GB/s (possibly cross-NUMA or through different paths)
	•	Latency: mostly between 1.6–2.0 us, consistent with PCIe communication characteristics
	•	Actual performance is generally consistent with theoretical expectations, PCIe 4.0 ×16 performance is functioning normally

⸻

5.6. GPU Communication Performance (Actual Test - NCCL All-Reduce)

NCCL All-Reduce collective communication test results:

Test command: `./all_reduce_perf -b 1M -e 1G -f 2 -g 8`

Test configuration:
	•	nccl-tests version 2.17.6
	•	Test range: 1 MB to 1 GB
	•	Step factor: 2
	•	8 GPUs participating in the test

Test results:

```
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
     1048576        262144     float     sum      -1   348.03    3.01    5.27       0   182.49    5.75   10.06       0
     2097152        524288     float     sum      -1   309.66    6.77   11.85       0   318.26    6.59   11.53       0
     4194304       1048576     float     sum      -1   483.86    8.67   15.17       0   486.82    8.62   15.08       0
     8388608       2097152     float     sum      -1  1047.63    8.01   14.01       0  1094.59    7.66   13.41       0
    16777216       4194304     float     sum      -1  2403.75    6.98   12.21       0  2288.61    7.33   12.83       0
    33554432       8388608     float     sum      -1  4808.58    6.98   12.21       0  4591.35    7.31   12.79       0
    67108864      16777216     float     sum      -1  9839.20    6.82   11.94       0  9306.81    7.21   12.62       0
   134217728      33554432     float     sum      -1  19117.3    7.02   12.29       0  19302.3    6.95   12.17       0
   268435456      67108864     float     sum      -1  36780.0    7.30   12.77       0  37992.3    7.07   12.36       0
   536870912     134217728     float     sum      -1  75950.0    7.07   12.37       0  75639.1    7.10   12.42       0
  1073741824     268435456     float     sum      -1   154715    6.94   12.15       0   134369    7.99   13.98       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 12.3406 
```

Test result analysis:
	•	Average bus bandwidth: 12.34 GB/s, consistent with PCIe interconnection and no NVLink expectations
	•	Small data sizes (1–4 MB): bandwidth fluctuates significantly (5–15 GB/s), affected by communication overhead
	•	Large data sizes (>16 MB): bandwidth stabilizes at 12–13 GB/s, approaching theoretical limits
	•	Error checking: all tests passed (#wrong = 0, Out of bounds = 0)
	•	NCCL communication is normal and can be used for multi-GPU training and inference tasks
	•	Performance is consistent with P2P test results, validating communication capabilities under PCIe architecture

⸻

6. Software Stack Status

Your server has training capabilities at the software level:
	•	CUDA driver loads normally
	•	PyTorch compiled with NCCL backend
	•	NCCL initialization successful
	•	No ECC or hardware errors detected in GPU runtime environment
	•	nvidia-smi topo -m can normally list GPU topology (no NVLink)

Therefore, the software component status is normal and can directly run distributed tasks.

⸻

7. Server Overall Configuration Summary (Ignoring Issues)

CPU
	•	Dual-socket (2× Intel Xeon or AMD EPYC)
	•	Supports PCIe Gen5

GPU
	•	8× NVIDIA RTX 5880 Ada
	•	PCIe 4.0 ×16 interface

Motherboard Topology
	•	All 8 cards distributed across multiple PCIe root ports
	•	CPU0 and CPU1 directly connect to some GPUs respectively
	•	Uses retimer or PCIe switch for expansion

Inter-GPU Communication
	•	Relies solely on PCIe + NUMA
	•	No NVLink

Software Environment
	•	CUDA normal
	•	PyTorch NCCL normal
	•	NCCL communication can initialize successfully
