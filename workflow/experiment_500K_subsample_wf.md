# Subsampling 500k reads
## File/Folder Size
This is based on 385 fastq files
```
(microbiome) root@97b0edf71e22:/project/workflow# du -sh results/*
4.0K	results/experiment_500K_subsample_wf.md
145G	results/kneadeddata
2.0M	results/metaphlan_bowtie2out
1.7M	results/metaphlan_profiles
83G	results/raw_subsampled
```

## Seqtk Subsampling
Takes 8 hours, 41 minutes and 18 seconds using 6 threads to subsample 500k reads for 385 fastq files
## Kneading Subsampled data
Takes 2 hours, 2 minutes and 10 seconds using 6 threads to kneade the subsampled 500k reads for 385 fastq files
## Metaphlan
Takes 9 hours, 29 minutes and 5 secionds using 6 threads
