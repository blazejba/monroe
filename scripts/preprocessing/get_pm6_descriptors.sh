#! /bin/bash
mkdir -p data/pm6
for i in {1..20}; do
    idx=$(printf "%02d" $i)
    if [ -f "data/pm6/pm6_processed_${idx}.parquet" ]; then
        echo "pm6_processed_${idx}.parquet already exists"
        continue
    fi
    echo "Downloading pm6_processed_${idx}.parquet"
    wget https://zenodo.org/records/8427533/files/pm6_processed_${idx}.parquet?download=1 -O data/pm6/pm6_processed_${idx}.parquet
done
echo "Downloading pm6_random_splits.pt"
wget https://zenodo.org/records/8427533/files/pm6_random_splits.pt?download=1 -O data/pm6/pm6_random_splits.pt