echo "### Download, unzip, and preprocess GreatestHits dataset"
./preprocessing.sh

echo "### Testing Material Estimation model on GreatestHits"
python test.py --config config_test.json
