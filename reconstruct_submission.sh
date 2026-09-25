#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/output"
echo "Reconstructing matching_results.tsv..."
if [ -f "matching_results.tsv.gz" ]; then
    gunzip -k -f matching_results.tsv.gz
elif [ -f "matching_results.tsv.part_aa" ]; then
    cat matching_results.tsv.part_* > matching_results.tsv
fi
echo "Successfully reconstructed matching_results.tsv! Line count: $(wc -l < matching_results.tsv) rows."
