# Subsampling 50000 Experiment

## File/Folder Size

This is based on 211 raw fastq files.
```
(microbiome) root@97b0edf71e22:/project/workflow# du -sh results/*
7.8G	results/kneadeddata
876K	results/metaphlan_bowtie2out
868K	results/metaphlan_profiles
4.5G	results/raw_subsampled
```

## Subsampling 50000 Reads
Takes 5 hours and 21 minutes and 54 seconds to subsample 211 fastq files using 2 threads

## Kneadding Subsampled Data
Takes 7 minutes and 14 seconds to knead 211 subsampled fastq files using 6 threads

## Metaphlan Kneadded Data
Takes 3 hours, 52 minutes and 5 seconds to run metaphlan on 211 kneadded data using 6 threads.

### Number of lines in metaphlan profile output

Each profile file contains 5 header (rowwise) meta information, the lines following are the actual abundance/species data. There are only 4 files that have more than 5 lines, this may mean we need to subsample at a high depth.

```
(microbiome) root@97b0edf71e22:/project/workflow# wc -l results/metaphlan_profiles/*
     5 results/metaphlan_profiles/ERR2017411_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017412_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017413_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017414_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017415_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017416_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017417_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017418_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017419_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017420_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017421_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017422_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017423_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017424_seqt.subsampled_kneaddata.trimmed_profile.txt
    11 results/metaphlan_profiles/ERR2017426_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017427_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017428_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017429_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017430_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017431_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017432_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017433_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017434_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017435_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017436_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017437_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017438_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017439_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017440_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017441_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017442_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017443_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017444_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017445_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017446_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017447_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017448_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017449_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017450_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017451_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017452_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017453_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017454_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017455_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017456_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017457_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017458_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017459_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017460_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017461_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017462_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017463_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017464_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017465_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017466_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017467_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017468_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017469_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017470_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017471_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017472_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017473_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017474_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017475_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017476_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017477_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017478_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017479_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017481_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017482_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017483_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017484_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017485_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017486_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017487_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017489_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017490_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017491_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017494_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017495_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017499_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017500_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017503_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017504_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017506_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017507_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017508_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017510_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017512_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017513_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017514_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017515_seqt.subsampled_kneaddata.trimmed_profile.txt
    24 results/metaphlan_profiles/ERR2017516_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017517_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017518_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017519_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017520_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017521_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017522_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017523_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017524_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017525_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017526_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017527_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017528_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017529_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017530_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017531_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017532_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017533_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017534_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017535_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017536_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017537_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017538_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017539_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017540_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017541_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017542_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017543_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017544_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017545_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017546_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017547_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017548_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017549_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017550_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017551_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017552_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017553_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017554_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017555_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017556_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017557_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017558_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017559_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017560_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017561_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017562_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017563_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017564_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017565_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017566_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017567_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017568_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017569_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017570_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017571_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017572_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017573_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017574_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017575_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017576_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017577_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017578_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017579_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017580_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017581_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017584_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017586_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017587_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017588_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017589_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017590_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017591_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017592_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017593_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017594_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017595_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017596_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017597_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017598_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017599_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017600_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017601_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017602_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017603_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017604_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017605_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017606_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017607_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017608_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017609_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017610_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017611_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017612_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017613_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017614_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017615_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017616_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017617_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017618_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017619_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017620_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017621_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017622_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017623_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017624_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017625_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017626_seqt.subsampled_kneaddata.trimmed_profile.txt
    12 results/metaphlan_profiles/ERR2017627_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017628_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017629_seqt.subsampled_kneaddata.trimmed_profile.txt
    11 results/metaphlan_profiles/ERR2017630_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017631_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017632_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017633_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017634_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017635_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017636_seqt.subsampled_kneaddata.trimmed_profile.txt
     5 results/metaphlan_profiles/ERR2017637_seqt.subsampled_kneaddata.trimmed_profile.txt
  1093 total
```


