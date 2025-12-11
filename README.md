# B565-Final-Project

This is the repository for our final project, which is about computing intrinsic dimensions within token embeddings.

Link to our paper: [link to paper goes here] 

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

example: `py .\src\extract_arxiv_cs_abstracts.py --input_jsonl data/raw/arxiv-raw.json --output_txt abstracts.txt --max_papers 10`

## Step 3: Create token samples

run make_samples.py

example: `py .\src\make_samples.py --input_txt abstracts.txt --out_dir data/processed/`

## Step 4: Generate dataset summary

run dataset_stats.py

example: `py .\src\dataset_stats.py --samples data/processed/sample_5000.txt  --out_csv stats.csv`

## Step 5: Compute intrinsic dimensions

run compute_id.py

example: `py .\src\compute_id.py --experiments .\configs\experiments.json --samples data/processed/sample_5000.txt`

## Step 6: Summarize and plot results

run summarize_results.py

example: `py .\src\summarize_results.py --runs_dir .\runs\`

then run plot_ratio_vs_ed.py

example: `py .\src\plot_ratio_vs_ed.py --summary_csv .\runs\id_summary.csv --out_png plot.png`