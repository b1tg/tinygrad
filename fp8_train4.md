
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


# 1221 amax custom kernel比想象中复杂，但不加的话虽然loss能降，但是貌似内存占用不太好，整体loss偏低


# 1222 found fast

TC128=0 detach-amax 1-22 allCON+output+im
   66 2051.50 ms run,  240.32 ms python,   5.63 ms fetch data, 1805.55 ms AMD * 8,  4.08 loss, 0.001081 LR, global_norm:  0.77, 1400.25 GB used, 539347.56 GFLOPS                                                                                                     

 
TC128=1 detach-amax 1-22 allCON+output+im
    9 1980.75 ms run,  242.62 ms python,   5.38 ms fetch data, 1732.75 ms AMD * 8,  4.19 loss, 0.001097 LR, global_norm:  1.23, 1400.25 GB used, 558610.82 GFLOPS                                     


remove cons in qkv | NO
   20 2135.41 ms run,  240.42 ms python,   5.35 ms fetch data, 1889.65 ms AMD * 8,  4.21 loss, 0.001094 LR, global_norm:  0.96, 1476.83 GB used, 536726.76 GFLOPS                                     


TC128=1 detach-amax allCON+output+im
   10 1980.54 ms run,  247.29 ms python,   5.49 ms fetch data, 1727.76 ms AMD * 8,  4.29 loss, 0.001097 LR, global_norm:  1.96, 1406.16 GB used, 558226.14 GFLOPS                                     




TC128=1 max1 allCON+output+im
   10 1974.69 ms run,  255.04 ms python,   5.35 ms fetch data, 1714.31 ms AMD * 8,  4.28 loss, 0.001097 LR, global_norm:  2.07, 1406.16 GB used, 559880.18 GFLOPS                                     
TC128=0 max1 allCON+output+im
   10 2014.91 ms run,  248.13 ms python,   5.22 ms fetch data, 1761.56 ms AMD * 8,  4.31 loss, 0.001097 LR, global_norm:  1.83, 1406.16 GB used, 548704.62 GFLOPS                                                                                    


TC128=1 max1+nround allCON+output+im
  : why not work, (some search timeout)
   10 2017.53 ms run,  247.02 ms python,   5.58 ms fetch data, 1764.93 ms AMD * 8,  4.48 loss, 0.001097 LR, global_norm:  2.50, 1410.46 GB used, 550332.25 GFLOPS                                                                                    


TC128=1 max1+nround allCON+output+im (forgot? cons in im) 内存减少但runstime还是增高，好像是我故意去的？
   10 1966.19 ms run,  253.20 ms python,   5.54 ms fetch data, 1707.45 ms AMD * 8,  4.34 loss, 0.001097 LR, global_norm:  2.09, 1307.32 GB used, 566654.58 GFLOPS                                                                                    
   111 2018.00 ms run,  240.78 ms python,   5.58 ms fetch data, 1771.63 ms AMD * 8,  4.00 loss, 0.001069 LR, global_norm:  0.78, 1307.32 GB used, 552107.46 GFLOPS                                                                                    
   > TC128=0
   22 2030.03 ms run,  237.43 ms python,   5.61 ms fetch data, 1786.99 ms AMD * 8,  4.33 loss, 0.001094 LR, global_norm:  1.08, 1307.32 GB used, 548835.22 GFLOPS                                                                                    


TC128=1 max1+nround allCON+output+im (forgot? cons in BertSelfOutput) (error in search; this raise memory)
  152 2007.02 ms run,  238.45 ms python,   5.69 ms fetch data, 1762.89 ms AMD * 8,  3.95 loss, 0.001057 LR, global_norm:  0.79, 1338.13 GB used, 555124.36 GFLOPS                                                                                    
  (not very stable)
  101 1964.84 ms run,  243.72 ms python,   5.59 ms fetch data, 1715.53 ms AMD * 8,  3.99 loss, 0.001072 LR, global_norm:  0.88, 1307.41 GB used, 567042.92 GFLOPS                                                                                    
   71 2010.85 ms run,  241.70 ms python,   5.83 ms fetch data, 1763.32 ms AMD * 8,  3.94 loss, 0.001080 LR, global_norm:  0.82, 1307.41 GB used, 554066.28 GFLOPS                                                                                    



 [RUN] max1+nround allCON+output+im 1h51m

 3061 1963.90 ms run,  277.94 ms python,   6.55 ms fetch data, 1679.41 ms AMD * 8,  1.38 loss, 0.000237 LR, global_norm:  0.59, 1338.13 GB used, 567314.35 GFLOPS                                                                                    
 


 remove back
  1034 2138.90 ms run,  246.20 ms python,   6.50 ms fetch data, 1886.20 ms AMD * 8,  1.53 loss, 0.000808 LR, global_norm:  1.38, 1345.10 GB used, 517250.77 GFLOPS                                                                                    
 1035 2136.74 ms run,  247.34 ms python,   6.07 ms fetch data, 1883.32 ms AMD * 8,  1.56 loss, 0.000808 LR, global_norm:  1.41, 1345.10 GB used, 517773.85 GFLOPS                                                                                    
 1036 2131.90 ms run,  247.59 ms python,   6.20 ms fetch data, 1878.11 ms AMD * 8,  1.53 loss, 0.000808 LR, global_norm:  1.36, 1345.10 GB used, 518948.23 GFLOPS                                                                                    
 1037 2135.75 ms run,  246.22 ms python,   6.25 ms fetch data, 1883.28 ms AMD * 8,  1.55 loss, 0.000808 LR, global_norm:  1.32, 1345.10 GB used, 518012.77 GFLOPS                                                                                    
 1038 2138.63 ms run,  246.71 ms python,   6.43 ms fetch data, 1885.48 ms AMD * 8,  1.54 loss, 0.000807 LR, global_norm:  1.06, 1345.10 GB used, 517316.28 GFLOPS                                                                                    
 1039 2138.00 ms run,  247.73 ms python,   6.60 ms fetch data, 1883.67 ms AMD * 8,  1.58 loss, 0.000807 LR, global_norm:  1.24, 1345.10 GB used, 517468.48 GFLOPS       




 # 1222 

 FP8=0 RUNMLPERF=0 BASEDIR="/raid/datasets/wiki" AMD_LLVM=0 PYTHONPATH=. DEFAULT_FLOAT=HALF BENCHMARK=6 BERT_LAYERS=10 BS=66 GPUS=2  MODEL=bert python3 examples/mlperf/model_train.py
     5  561.66 ms run,    9.03 ms python,   0.63 ms fetch data,  551.99 ms AMD * 2,  0.69 loss, 0.000145 LR, global_norm:  0.35, 45.75 GB used,  59175.76 GFLOPS


FP8=1
    5  585.03 ms run,   10.85 ms python,   0.55 ms fetch data,  573.63 ms AMD * 2,   nan loss, 0.000145 LR, global_norm:   nan, 44.48 GB used,  56681.59 GFLOPS

TC128=1
    5  588.50 ms run,   10.81 ms python,   0.57 ms fetch data,  577.13 ms AMD * 2,   nan loss, 0.000145 LR, global_norm:   nan, 44.48 GB used,  56346.96 GFLOPS


after tuning 1224
FP8=0
    5  538.29 ms run,    8.95 ms python,   0.62 ms fetch data,  528.71 ms AMD * 2,  0.69 loss, 0.000145 LR, global_norm:  0.35, 45.54 GB used,  61390.16 GFLOPS
    BertOutput -> hidden_states = self.dense(hidden_states.contiguous().contiguous_backward()).contiguous().contiguous_backward() (FP8=1 提升小)
    5  526.66 ms run,    9.04 ms python,   0.56 ms fetch data,  517.06 ms AMD * 2,  0.69 loss, 0.000145 LR, global_norm:  0.35, 45.50 GB used,  62491.50 GFLOPS
FP8=1
    5  559.30 ms run,   11.06 ms python,   0.55 ms fetch data,  547.70 ms AMD * 2,   nan loss, 0.000145 LR, global_norm:   nan, 44.39 GB used,  58948.27 GFLOPS

FP8=0 BENCHMARK=6 BERT_LAYERS=10 BS=66 GPUS=1
    5  972.47 ms run,    3.22 ms python,   0.55 ms fetch data,  968.70 ms AMD,  0.69 loss, 0.000145 LR, global_norm:  0.35, 40.08 GB used,  33182.39 GFLOPS

FP8=1 BENCHMARK=6 BERT_LAYERS=10 BS=66 GPUS=1
    5 1006.01 ms run,    3.56 ms python,   0.66 ms fetch data, 1001.79 ms AMD,   nan loss, 0.000145 LR, global_norm:   nan, 38.97 GB used,  32130.72 GFLOPS


after tuning 1224
FP8=0 BENCHMARK=6 BERT_LAYERS=2 BS=66 GPUS=1
    5  549.97 ms run,    2.25 ms python,   2.27 ms fetch data,  545.45 ms AMD,   nan loss, 0.000145 LR, global_norm:   nan, 10.39 GB used,  17207.67 GFLOPS

FP8=1 BENCHMARK=6 BERT_LAYERS=2 BS=66 GPUS=1
    5  545.29 ms run,    1.61 ms python,   0.54 ms fetch data,  543.14 ms AMD,  0.69 loss, 0.000145 LR, global_norm:  0.35, 10.13 GB used,  17335.02 GFLOPS


real run
 FP8=0
 1970 1961.33 ms run,  217.43 ms python,   6.16 ms fetch data, 1737.74 ms AMD * 8,  1.41 loss, 0.000544 LR, global_norm:  0.71, 1466.00 GB used, 581481.40 GFLOPS


 FP8=1 ste TC128=0
 1970 1962.68 ms run,  263.90 ms python,   6.03 ms fetch data, 1692.75 ms AMD * 8,  1.38 loss, 0.000544 LR, global_norm:  0.65, 1417.07 GB used, 582248.38 GFLOPS

