#!/bin/sh
success_down=$(grep -o "'[[:alnum:]]*' was downloaded successfully" logs/sra_download.log | sort | uniq | wc -l)

fail_down=$(grep -o "'[[:alnum:]]*' - no data" logs/sra_download.log | sort | uniq | wc -l)
fail_down_list=$(grep -o "'[[:alnum:]]*' - no data" logs/sra_download.log | sort | uniq)
echo "INFO: $success_down accessions were successfully downloaded"
echo "INFO: $fail_down accessions failed to download (no data)"
echo "INFO: Failed accessions list:\n$fail_down_list"

