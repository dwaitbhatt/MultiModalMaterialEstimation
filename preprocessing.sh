apt-get update
apt-get install wget unzip -y

if [ -d "vis-data-256" ] || [ -f "vis-data-256.zip" ]; then
    echo "vis-data-256 directory or zip file already exists"
else
    wget https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip --no-check-certificate
    unzip vis-data-256.zip
    rm vis-data-256.zip
fi

if [ -d "vis-data-256-processed" ]; then
    echo "vis-data-256 dataset is already cached"
else
    python process_greatest_hits.py
fi