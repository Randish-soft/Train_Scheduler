
# Generate:

poetry run python -m Model --mode generate --input input/lebanon_cities.csv --country lebanon --optimize "cost,ridership" --output-dir output/lebanon

# Learn:
Train_Scheduler % python -m Model --mode learn --country Country_Name --train-types IC,OTHERS
