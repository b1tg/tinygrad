
# how to remove 

CUSTOM_CLAMP + max1

    9 1563.27 ms run,    8.61 ms python,   1.42 ms fetch data, 1553.25 ms AMD,  4.17 loss, 0.001097 LR, global_norm:  3.31, 180.43 GB used,  92697.87 GFLOPS  



CUSTOM_CLAMP + CUSTOM_AMAX
        4 4556.92 ms run,   12.30 ms python,   1.21 ms fetch data, 4543.41 ms AMD,  4.74 loss, 0.001099 LR, global_norm:  4.05, 180.43 GB used,  31749.25 GFLOPS

# not working anymore?


GPUS=2 FP8=0 ok
 1169  884.52 ms run,   21.04 ms python,   1.35 ms fetch data,  862.13 ms AMD * 2,  3.89 loss, 0.000770 LR, global_norm:  1.99, 185.60 GB used, 163281.49 GFLOPS                                                                                                      


GPUS=2 FP8=1 ok
 1522 1282.21 ms run,   28.78 ms python,   1.40 ms fetch data, 1252.03 ms AMD * 2,  3.69 loss, 0.000671 LR, global_norm:  2.52, 239.29 GB used, 122749.77 GFLOPS



2/256 GPUS=2 FP8=0

  675 1654.58 ms run,   24.16 ms python,   2.07 ms fetch data, 1628.34 ms AMD * 2,  3.78 loss, 0.000910 LR, global_norm:  1.99, 361.23 GB used, 174505.96 GFLOPS


2/256 GPUS=2 FP8=1

 1101 2396.07 ms run,   29.26 ms python,   2.04 ms fetch data, 2364.77 ms AMD * 2,  3.34 loss, 0.000789 LR, global_norm:  2.75, 463.28 GB used, 131323.40 GFLOPS                     

 (fix back multi)
  182 1567.94 ms run,   24.27 ms python,   1.99 ms fetch data, 1541.68 ms AMD * 2,  4.09 loss, 0.001049 LR, global_norm:  2.01, 342.63 GB used, 183815.40 GFLOPS                                                                                     
  475 1562.77 ms run,   23.62 ms python,   1.89 ms fetch data, 1537.26 ms AMD * 2,  3.98 loss, 0.000966 LR, global_norm:  1.15, 342.63 GB used, 184423.75 GFLOPS                                                                                     

1/128 GPUS=1 FP8=1
  567 1549.87 ms run,    7.60 ms python,   1.06 ms fetch data, 1541.22 ms AMD,  3.92 loss, 0.000940 LR, global_norm:  1.53, 168.84 GB used,  93189.59 GFLOPS                                                                                         

1/128 GPUS=1 FP8=0
   44 1661.82 ms run,    6.32 ms python,   1.01 ms fetch data, 1654.49 ms AMD,  3.90 loss, 0.001088 LR, global_norm:  1.64, 174.90 GB used,  87162.05 GFLOPS                                                                                         
 1620 1662.23 ms run,    6.12 ms python,   1.04 ms fetch data, 1655.07 ms AMD,  3.47 loss, 0.000643 LR, global_norm:  3.05, 178.11 GB used,  87140.57 GFLOPS                                                                                         



8/1024 GPUS=8 FP8=1


 320 1932.62 ms run,  226.88 ms python,   5.64 ms fetch data, 1700.09 ms AMD * 8,  3.78 loss, 0.001010 LR, global_norm:  0.85, 1418.88 GB used, 593002.33 GFLOPS  

8/1024 GPUS=8 FP8=0 1h55m29s December 11th, 2025 8:44:12 PM
 3060 2092.11 ms run,  224.53 ms python,   6.62 ms fetch data, 1860.96 ms AMD * 8,  1.37 loss, 0.000237 LR, global_norm:  0.69, 1492.04 GB used, 548517.55 GFLOPS
