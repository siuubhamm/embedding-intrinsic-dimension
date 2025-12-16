# B565-Final-Project

This is the repository for our final project, which is about computing intrinsic dimensions within token embeddings.

Link to our paper: [here](https://github.iu.edu/B565-Fall-2025/B565-Final-Project/blob/main/Data%20Mining%20Final%20Technical%20Report.pdf)

Below are step-by-step instructions for how to run the code within the repo.

*Required packages for this program*
* numpy
* pandas
* matplotlib
* faiss-cpu

## Step 1: Install requirements

1. Using a virtual environment is recommended, ensure that you have required packages installed by using pip or conda.

   `pip install -r requirements.txt`

2. Then find the data you would like to have processed in JSON form, 
we recommend using abstracts taken from the arXiv dataset, available: https://www.kaggle.com/datasets/Cornell-University/arxiv

   It is recommended that you place this data in a directory named `data/raw/`

3. Then, find an embedding model that you would like to use, we recommend the GloVe model for this program: https://www.kaggle.com/datasets/gargvg/glovewordembeddings/versions/1

   place the files from the model in a folder named `embeddings`

## Step 2: Extract abstract texts

run extract_arxiv_cs_abstracts.py to create a .txt file compiling abstract data. This step is optional if you already have the text data.

example: `python src/extract_arxiv_cs_abstracts.py --input_jsonl data/raw/arxiv-metadata-oai-snapshot.json --output_txt data/processed/arxiv_cs/all_cs_abstracts.txt`

## Step 3: Create token samples

run make_samples.py

example: `python src/make_samples.py --input_txt data/processed/arxiv_cs/all_cs_abstracts.txt --out_dir data/processed/arxiv_cs --seed 2025 --sizes 5000 20000 50000`

## Step 4: Generate dataset summary

run dataset_stats.py

example: `python src/dataset_stats.py --samples data/processed/arxiv_cs/sample_5000.txt data/processed/arxiv_cs/sample_20000.txt data/processed/arxiv_cs/sample_50000.txt --out_csv runs/dataset_stats.csv`

## Step 5: Compute intrinsic dimensions

run compute_id.py

example: `python src/compute_id.py --experiments configs/experiments.json --samples data/processed/arxiv_cs/sample_5000.txt data/processed/arxiv_cs/sample_20000.txt data/processed/arxiv_cs/sample_50000.txt --out_dir runs --k 10 --repeats 3`

## Step 6: Summarize and plot results

run summarize_results.py

example: `python src/summarize_results.py --runs_dir runs --stats_csv runs/dataset_stats.csv`

then run plot_ratio_vs_ed.py

example: `python src/plot_ratio_vs_ed.py --summary_csv runs/id_summary.csv --model glove --level token --sample_file sample_50000.txt`

## Step 7: Run Retrieval Benchmark

run retrieval_benchmark.py

example: `python src\retrieval_benchmark.py --sample data\processed\arxiv_cs\sample_50000.txt --glove300 embeddings\glove.6B.300d.txt --num_queries 1000000 --topk 10 --R_true 100 --pca_dim 15`