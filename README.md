#COVID NETWORK BRIDGES PAPER CODE
##hdsr_runner.py
###do_corpus_clean(fname):
fname=jsonl file containing document texts, metadata
creates and saves json file for each document in fname, containing scispacy tokenized document list.
###create_dict(fname, min_count=0, daterange=None)
fname=jsonl file with document information
min_count=minimum word count from search terms associated with fname for document to be included in processing
daterange=2 item list of python dates, for inclusion of document in processing
creates and saves gensim dictionary from supplied jsonl document file. Filters documents based on min_count and daterange parameters
###do_mlmom_models(fname, min_count=0, daterange=None)
fname=jsonl file with document information
min_count=minimum word count from search terms associated with fname for document to be included in processing
daterange=2 item list of python dates, for inclusion of document in processing
runs and saves six gensim models with random seeds from 0-500, at set hyperparameters.
###do_eval(fname, min_count=None, daterange=None)
fname=jsonl file with document information
min_count=minimum word count from search terms associated with fname for document to be included in processing
daterange=2 item list of python dates, for inclusion of document in processing
cycles through set alpha, beta, and topic number settings and runs model with hyperparameters at each setting, saving c_v coherence and perplexity to csv.
###create_mlmom_format(fname, min_count=None, daterange=None)
fname=jsonl file with document information
min_count=minimum word count from search terms associated with fname for document to be included in processing
daterange=2 item list of python dates, for inclusion of document in processing
runs clustering, pca, and document-topic information for final display
creates formatted json data file, uploads to s3 for site access and save locally

##do_hdsr_clusters.py
A script to assign from the generated clustering to one of the three general subjects identified: 
Bench Science, Treatment, Public Health

##hdsr_betweenness.py
A script to generate the betweenness centrality for each of the topics given the generated visualization output json.

