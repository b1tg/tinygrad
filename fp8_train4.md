
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

tinyamd3
 1433 2025.62 ms run,  265.42 ms python,   6.47 ms fetch data, 1753.73 ms AMD * 8,  1.40 loss, 0.000696 LR, global_norm:  0.81, 1418.88 GB used, 565735.80 GFLOPS
 1434 2022.66 ms run,  265.60 ms python,   6.52 ms fetch data, 1750.53 ms AMD * 8,  1.42 loss, 0.000696 LR, global_norm:  0.79, 1418.88 GB used, 566563.98 GFLOPS
    3063 2016.81 ms run,  265.87 ms python,   7.44 ms fetch data, 1743.50 ms AMD * 8,  1.34 loss, 0.000236 LR, global_norm:  0.65, 1418.88 GB used, 568206.00 GFLOPS
    3064 2012.98 ms run,  265.37 ms python,   6.99 ms fetch data, 1740.61 ms AMD * 8,  1.38 loss, 0.000236 LR, global_norm:  0.74, 1418.88 GB used, 569288.34 GFLOPS
    3065 2014.07 ms run,  265.09 ms python,   6.92 ms fetch data, 1742.06 ms AMD * 8,  1.35 loss, 0.000236 LR, global_norm:  0.62, 1418.88 GB used, 568980.26 GFLOPS
    eval lm loss: 1.29, eval clsf loss: 0.02, eval lm accuracy: 0.720512,                   eval clsf accuracy: 0.51, avg eval step time: 0.63
 79%|███████████████████████████████████████████████████████████████████▌                  | 3066/3900 [2:14:47<28:20,  2.04s/it]Reference Convergence point reached after 3139584 datasamples and 2h14m47.57s.
 79%|███████████████████████████████████████████████████████████████████▌                  | 3066/3900 [2:14:47<36:39,  2.64s/it]
wandb: 
wandb: 🚀 View run 8/1024 FP8=1 TC128=1 output at: 
wandb: Find logs at: wandb/run-20251218_025758-ajs7ue27/logs

imm + out
   80 1868.34 ms run,  258.57 ms python,   5.77 ms fetch data, 1604.00 ms AMD * 8,  4.18 loss, 0.001077 LR, global_norm:  0.73, 1384.51 GB used, 613824.19 GFLOPS                                                                                    

8/1024 GPUS=8 FP8=0 1h55m29s 2025-12-11
 3060 2092.11 ms run,  224.53 ms python,   6.62 ms fetch data, 1860.96 ms AMD * 8,  1.37 loss, 0.000237 LR, global_norm:  0.69, 1492.04 GB used, 548517.55 GFLOPS

FP8=0 8/1024 mi350  3h29m3s  2025-12-08 2773 steps
 2773 4341.79 ms run,  226.56 ms python,   6.42 ms fetch data, 4108.81 ms AMD * 8,  1.35 loss, 0.000318 LR, global_norm:  0.70, 1388.96 GB used, 215149.50 GFLOPS







TC128=0 8/8

  127 1870.27 ms run,  299.58 ms python,   6.19 ms fetch data, 1564.51 ms AMD * 8,  3.97 loss, 0.001064 LR, global_norm:  0.71, 1307.01 GB used, 589866.24 GFLOPS                                                                                    


 3208 1850.45 ms run,  271.95 ms python,   6.89 ms fetch data, 1571.62 ms AMD * 8,  1.41 loss, 0.000195 LR, global_norm:  0.65, 1338.80 GB used, 596184.32 GFLOPS                                                                                    
 3209 1867.34 ms run,  270.66 ms python,   6.71 ms fetch data, 1589.96 ms AMD * 8,  1.42 loss, 0.000195 LR, global_norm:  0.76, 1338.80 GB used, 590792.86 GFLOPS                                                                                    
 3210 1853.31 ms run,  271.09 ms python,   6.43 ms fetch data, 1575.79 ms AMD * 8,  1.43 loss, 0.000195 LR, global_norm:  0.67, 1338.80 GB used, 595263.93 GFLOPS                                                                                    
 3211 1870.42 ms run,  271.19 ms python,   6.30 ms fetch data, 1592.93 ms AMD * 8,  1.39 loss, 0.000194 LR, global_norm:  0.59, 1338.80 GB used, 589819.90 GFLOPS                                                                                    
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.14it/s]
eval lm loss: 1.29, eval clsf loss: 0.02, eval lm accuracy: 0.720069,                   eval clsf accuracy: 0.49, avg eval step time: 0.46                                                                                                           
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                   | 3212/3900 [1:51:16<21:32,  1.88s/it]Reference Convergence point reached after 3289088 datasamples and 1h51m16.88s.
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                   | 3212/3900 [1:51:16<23:50,  2.08s/it]
wandb: 
wandb: 🚀 View run TC128=0 allCON+output+im 8/1024 FP8=1 at: 

row
  TC128=0
  124 2030.50 ms run,  257.58 ms python,   5.75 ms fetch data, 1767.17 ms AMD * 8,  3.92 loss, 0.001065 LR, global_norm:  0.88, 1563.11 GB used, 543778.22 GFLOPS                                                                                    

  TC128=1
   75 1950.86 ms run,  234.41 ms python,   5.64 ms fetch data, 1710.81 ms AMD * 8,  4.02 loss, 0.001079 LR, global_norm:  0.87, 1563.11 GB used, 565976.25 GFLOPS                                                                                    

  only out
  TC128=1
     18 2056.62 ms run,  241.16 ms python,   5.68 ms fetch data, 1809.78 ms AMD * 8,  4.15 loss, 0.001095 LR, global_norm:  0.85, 1667.26 GB used, 536922.99 GFLOPS  

    remove other con
         6 2113.86 ms run,  235.56 ms python,   5.62 ms fetch data, 1872.68 ms AMD * 8,  4.43 loss, 0.001098 LR, global_norm:  2.24, 1738.79 GB used, 540830.71 GFLOPS                                                                                    
  TC128=0
      7 1933.77 ms run,  238.88 ms python,   5.38 ms fetch data, 1689.52 ms AMD * 8,  4.18 loss, 0.001098 LR, global_norm:  1.84, 1667.26 GB used, 571033.04 GFLOPS                                                                                                                                                                      
      remove other con
      20 2024.18 ms run,  237.61 ms python,   5.27 ms fetch data, 1781.30 ms AMD * 8,  4.37 loss, 0.001094 LR, global_norm:  1.35, 1738.79 GB used, 564792.51 GFLOPS                                                                                    


all con + per tensor + not (1,23)

