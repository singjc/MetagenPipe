#!/bin/bash
working_dir=${1:-`pwd`}
# docker run -d microbiome_ubuntu20 /bin/bash  /project/workflow/cmd.sh
docker run --rm -w /project/workflow -v $working_dir:/project whitleyo/microbiome_ubuntu20:latest /bin/bash /project/workflow/cmd_sradownload_identifiers.sh
