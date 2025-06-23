# Spark Environments

Each version has its own folder with everything you need.

## Spark 4.0
```bash
cd docker/spark-4.0
./launch.bat
```

## Spark 2.4  
```bash
cd docker/spark-2.4
./launch.bat
```

Or just go into the folder and run `docker-compose run --rm spark` directly.

The datasets are already mounted, you can find them at `/workspace/data/sourced/HIGGS.csv` once you're inside the container.

Both environments are configured with enough memory to handle the 7GB+ datasets, so you should be good to go.

Clean up when you're done:
```bash
docker-compose down
``` 