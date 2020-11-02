from gensim.models.phrases import Phrases, Phraser, original_scorer
import gensim.corpora as corpora
import string
import re
from stop_words import get_stop_words
import nltk
from nltk.tokenize import RegexpTokenizer
import gensim
import json
import tqdm
import numpy as np
from gensim.models import CoherenceModel
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import scispacy
import spacy
import datetime


# supporting function
def compute_model(corpus, dictionary, raw_corpus, topics=20, a=0.91, b=.1, seed=100, savefile=None):
	full_raw_corpus = []
	count = 0
	for x in corpus:
		#print(count)
		full_raw_corpus.append(x)
		count += 1
	lda_model = gensim.models.LdaMulticore(
		corpus=full_raw_corpus,
		id2word=dictionary,
		num_topics=topics, 
		random_state=seed,
		chunksize=100,
		passes=5,
		alpha=a,
		eta=b,
		eval_every=5,
		per_word_topics=True)

	topics = lda_model.show_topics(num_topics=topics, num_words=10, formatted=True)
	
	if savefile:
		lda_model.save(savefile)
	else:
		lda_model.save('test_model')


	perp = lda_model.log_perplexity(full_raw_corpus)
	full_raw_corpus = []
	for x in raw_corpus:
		full_raw_corpus.append(x)
	coherence_model_lda = CoherenceModel(model=lda_model, texts=full_raw_corpus, dictionary=dictionary, coherence='c_v')
	with np.errstate(invalid='ignore'):
		lda_score = coherence_model_lda.get_coherence()
		print(lda_score)
		print(perp)
		return [lda_score, perp]


rm_chars = string.punctuation.replace('-', '').replace("'", "")
pattern = r"[{}]".format(rm_chars) # create the pattern



spacy_nlp = spacy.load("en_core_sci_lg")


def remove_stop_words(text, customized_stops=[]):
	stops = get_stop_words('en')
	tokens = [x for x in text if x and x not in stops and len(x) > 2 and x not in customized_stops]
	return tokens


def rm_cruft(w):
	if w.find('\\') >= 0:
		print(w)
		return False
	return True


def clean_text(t):
	try:
		doc = spacy_nlp(t.lower().replace("\n", " "))
		cleaned_t = list(doc.ents)
		cleaned_t = list(map(lambda x: x.text, cleaned_t))
		cleaned_t = list(filter(rm_cruft, cleaned_t))
		if len(cleaned_t) > 100000:
			return cleaned_t[:100000]
		return cleaned_t
	except ValueError:
		return [""]

def process_ngrams(texts):
	cleaned_corpus = []
	for x in texts:
		cleaned_corpus.append(clean_text(x))


	return cleaned_corpus[0]

def do_corpus_clean(fname):
	f = open('hdsr-searches/' + fname,'r')
	

	#cleaned_d = []
	index = 0

	#wf = open('cleaned_hdsr_searches/' + fname, 'w+')
	for doc_str in f:

		doc = json.loads(doc_str)
		print(doc[0])
		wf = open('cleaned_hdsr_searches/all_docs/' + doc[0] + '.json', 'w')

		cleaned_doc = process_ngrams([doc[8]])
		if index % 1000 == 0:
			print(fname)
			print("cleaned " + str(index))
		#print(cleaned_doc)
		wf.write(json.dumps(cleaned_doc))
		#cleaned_d.append(cleaned_doc)
		index += 1
		


def do_eval(fname, min_count=None, daterange=None):
	min_topics = 15
	max_topics = 75
	step_size = 5
	topics_range = range(min_topics, max_topics, step_size)
	# Alpha parameter
	alpha = list(np.arange(0.01, 1, 0.3))
	alpha.append('symmetric')
	alpha.append('asymmetric')
	# Beta parameter
	beta = list(np.arange(0.01, 1, 0.3))

	#alpha = [0.31]
	#beta = [0.91]

	beta.append('symmetric')
	beta.append('auto')


	#alpha = [0.01]
	#beta = [0.01]
	#alpha = ['symmetric'] 
	#beta= [0.01]



	id2word = corpora.Dictionary().load('hdsr_dicts/' + fname[:-5])

	corpus_sets = [""]

	corpus_title = ['100% Corpus']

	model_results = {'Validation_Set': [],
				 'Topics': [],
				 'Alpha': [],
				 'Beta': [],
				 'Coherence': [],
				 'Perplexity' : []
				}

	if 1 == 1:
		pbar = tqdm.tqdm(total=270)
		# iterate through validation corpuses
		for i in range(len(corpus_sets)):
			# iterate through number of topics
			for k in topics_range:
				# iterate through alpha values
				for a in alpha:
					# iterare through beta values
					for b in beta:
						# get the coherence score for the given parameters


						corpus_iterator = iter(DictionaryIter(fname, id2word, raw=False, min_count=min_count, date=daterange))
						text_iterator = iter(DictionaryIter(fname, id2word, raw=True, min_count=min_count, date=daterange))
						count = 0
						thing = []

						cv = compute_model(corpus=corpus_iterator, dictionary=id2word, raw_corpus=text_iterator, topics=k, a=a, b=b)
						model_results['Validation_Set'].append(corpus_title[i])
						model_results['Topics'].append(k)
						model_results['Alpha'].append(a)
						model_results['Beta'].append(b)
						model_results['Coherence'].append(cv[0])
						model_results['Perplexity'].append(cv[1])
						pbar.update(1)
	pd.DataFrame(model_results).to_csv('ebola_m2_topics_tuning.csv', index=False)
	pbar.close()
	return 

class DictionaryIter:
	def __init__(self, fname, corpus_dict, raw=False, min_count=0, date=None):
		#self.rf = open("cleaned_hdsr_searches/" + fname, "r")
		self.rf = open("hdsr-searches/" + fname, "r")
		self.corpus_dict =  corpus_dict
		self.raw = raw
		self.name = fname
		self.min_count = min_count
		self.date = date

	def __iter__(self):
		return self

	def _in_date(self, doc_obj):
		dateformat = "%Y-%m-%d"
		if self.date is None:
			return True
		else:
			doc_date = datetime.datetime.strptime(doc_obj[7], dateformat)
			if doc_date >= self.date[0] and doc_date < self.date[1]:
				return True
			return False


	def _minimum_count(self, doc_obj, filter_by_cov=False):
		if self.min_count == 0:
			return True
		name_searches = {
			"H1N1-OR-Influenza_A_m2.json" : ["H1N1","Influenza", "Influenza-a"],
			"Ebola-OR-EVD-OR-Ebola_Virus_Disease_m2.json" : ["Ebola", "EVD", "Ebola Virus Disease"],
			"Zika.json" : ['Zika'],
			"Zika_m2.json" : ['Zika'],
			"MERS-OR-Middle_East_Respiratory_Syndrome_m2.json" : ["MERS", "Middle East Respiratory Syndrome"],
			"SARS-OR-Severe_Acute_Respiratory_Syndrome_m2.json" : ["SARS", "Severe Acute Respiratory Syndrome"], 
			"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
			"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q1.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
			"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q2.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
			"HIV-OR-human_immunodeficiency_virus-OR-AIDS-OR-acquired_immunodeficiency_syndrome_m2.json" : ["HIV", "human immunodeficiency virus","AIDS","acquired immunodeficiency syndrome"]
		}

		doc_text = doc_obj[8].lower()
		doc_term_count = 0

		cov_term_count = 0
		if filter_by_cov:
			for w in name_searches['severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q1.json']:
				term_count = doc_text.count(w.lower())
				cov_term_count += term_count
			if cov_term_count > 0:
				return False

		for w in name_searches[self.name]:
			term_count = doc_text.count(w.lower())
			doc_term_count += term_count

		if doc_term_count >= self.min_count:
			return True
		return False

	def __next__(self):
		#print(self.rf.readline())

		doc_info_str = self.rf.readline()
		if doc_info_str == "":
			raise StopIteration
		doc_info = json.loads(doc_info_str)
		if self._in_date(doc_info):
			f = open('cleaned_hdsr_searches/all_docs/' + doc_info[0] + '.json', 'r')
			doc_tokens = f.read()			
			if len(doc_tokens) == 0:
				doc_tokens = []
			else:
				doc_tokens = json.loads(doc_tokens)
			#print(doc_tokens[0])

			if self._minimum_count(doc_info) is False:
				return self.__next__()
			
			if self.raw:
				return doc_tokens if len(doc_tokens) < 100000 else doc_tokens[:100000]
			else:
				return self.corpus_dict.doc2bow(doc_tokens) if len(doc_tokens) < 100000 else self.corpus_dict.doc2bow(doc_tokens[:100000])
		else:
			return self.__next__()

def min_count_func(doc_obj, name, min_count=0, filter_by_cov=False):
	if min_count == 0:
			return True
	name_searches = {
		"H1N1-OR-Influenza_A_m2.json" : ["H1N1","Influenza", "Influenza-a"],
		"Ebola-OR-EVD-OR-Ebola_Virus_Disease_m2.json" : ["Ebola", "EVD", "Ebola Virus Disease"],
		"Zika.json" : ['Zika'],
		"Zika_m2.json" : ['Zika'],
		"MERS-OR-Middle_East_Respiratory_Syndrome_m2.json" : ["MERS", "Middle East Respiratory Syndrome"],
		"SARS-OR-Severe_Acute_Respiratory_Syndrome_m2.json" : ["SARS", "Severe Acute Respiratory Syndrome"], 
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q1.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q2.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
		"HIV-OR-human_immunodeficiency_virus-OR-AIDS-OR-acquired_immunodeficiency_syndrome_m2.json" : ["HIV", "human immunodeficiency virus","AIDS","acquired immunodeficiency syndrome"]
	}

	doc_text = doc_obj[8].lower()
	doc_term_count = 0

	cov_term_count = 0
	if filter_by_cov:
		for w in name_searches['severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q1.json']:
			term_count = doc_text.count(w.lower())
			cov_term_count += term_count
		if cov_term_count > 0:
			return False

	for w in name_searches[name]:
		term_count = doc_text.count(w.lower())
		doc_term_count += term_count

	if doc_term_count >= min_count:
		return True
	return False

def in_date(doc_obj, daterange):
	dateformat = "%Y-%m-%d"
	if daterange is None:
		return True
	else:
		doc_date = datetime.datetime.strptime(doc_obj[7], dateformat)
		if doc_date >= daterange[0] and doc_date < daterange[1]:
			return True
		return False



def create_dict(fname, min_count=0, daterange=None):
	rf = open("hdsr-searches/" + fname, "r")
	corpus_dict = corpora.Dictionary()
	count = 0
	for doc_list in rf:
		doc_info = json.loads(doc_list)
		print(doc_info[0])
		f = open('cleaned_hdsr_searches/all_docs/' + doc_info[0] + '.json', 'r')
		if min_count_func(doc_info, fname, min_count=min_count) is False:
			continue
		if in_date(doc_info, daterange) is False:
			continue
		t = f.read()
		if len(t) == 0:
			continue
		doc_tokens = json.loads(t)
		print("document!" + str(count))
		print(len(doc_tokens))
		corpus_dict.add_documents([doc_tokens])
		print(len(corpus_dict))
		if len(corpus_dict) > 1990000:
			corpus_dict.filter_extremes(no_below=5, no_above=0.75, keep_n=1900000)
		count+=1


	print("filtering extremes of dictionary")
	print("DOC COUNT " + str(count))
	corpus_dict.filter_extremes(no_below=5, no_above=0.75, keep_n=500000)
	corpus_dict.save("hdsr_dicts/" + fname[:-5])
	return corpus_dict

def doc_iterator(fname, corpus_dict, raw=True, min_count=0, daterange=None):
	#rf = open("cleaned_hdsr_searches/" + fname, "r")
	rf = open("hdsr-searches/" + fname, "r")
	for doc_list in rf:
		doc_info = json.loads(doc_list)
		if min_count_func(doc_info, fname, min_count=min_count) is False:
			continue
		if in_date(doc_info, daterange) is False:
			continue
		f = open('cleaned_hdsr_searches/all_docs/' + doc_info[0] + '.json', 'r')
		t = f.read()
		if len(t) == 0:
			print("Empty Doc")
			yield []
		else:
			text = json.loads(t)
			if raw == True:
				yield text
			else:
				yield corpus_dict.doc2bow(text)


def do_mlmom_models(fname, min_count=0, date=None):
	
	id2word = corpora.Dictionary().load('hdsr_dicts/' + fname[:-5])
	for x in range(0, 600, 100):
		print("model1")
		corpus_iterator = iter(DictionaryIter(fname, id2word, raw=False, min_count=min_count, date=date))
		text_iterator = iter(DictionaryIter(fname, id2word, raw=True, min_count=min_count, date=date))
		count = 0
		print(count)
		
		
		thing = []

		save_fp = 'hdsr_models/' + fname[:-5] + '_' + str(x) + "_special"
		compute_model(corpus=corpus_iterator, dictionary=id2word, raw_corpus=text_iterator, topics=15, a='symmetric', b=.01, seed=x, savefile=save_fp)
		
		



def upload_str(string_data, file_path):
    import io
    import boto3
    
    

    client = boto3.client(
        's3',
        aws_access_key_id=AWS_PROFILE['ACCESS_KEY'],
        aws_secret_access_key=AWS_PROFILE['SECRET_KEY'],
        region_name='us-east-2'
    )
    prefix = os.path.join('LDA-models', file_path)
    print(prefix)
    string_data = string_data.encode('utf-8')
    f = io.BytesIO(string_data)
    client.upload_fileobj(f, 'rnlp-data', prefix)
    return 1

def create_mlmom_format(fname, min_count=0, daterange=None):
	id2word = id2word = corpora.Dictionary().load('hdsr_dicts/' + fname[:-5])
	corpus_iterator = doc_iterator(fname, id2word, raw=False, min_count=min_count, daterange=daterange)
	text_iterator = doc_iterator(fname, id2word,raw=True, min_count=min_count, daterange=daterange)
	count = 0

	ntopics = 30
	formatted_m = []
	for x in range(0, 600, 100):
		x1 = int(x / 100)
		save_fp = 'hdsr_models/' + fname[:-5] + '_' 
		model_info_json = {
			"name" : "m" + str(x1),
			"diffs" : []
			}
		lda1 = gensim.models.LdaMulticore.load(save_fp + str(x) )

		for y in range(x, 600, 100):
			y1 = int(y / 100)
			lda2 = gensim.models.LdaMulticore.load(save_fp + str(y) )
			mdiff, annotation = lda1.diff(lda2, distance='hellinger')
			model_info_json["diffs"].append({
					"name" : "m" + str(y1),
					"scores": mdiff.tolist()
				})
		topics = lda1.show_topics(num_topics = ntopics)
		t1_a = []
		for x in topics:
			try:
				t1_a.append((x[0].item(), x[1]))
			except:
				t1_a.append((x[0], x[1]))

		topic_weights = []
		for t in lda1.state.sstats:
			score = 0
			for w in t: 
				score = score + w
			topic_weights.append(score)
		model_info_json["topic_weights"] = topic_weights
		model_info_json["topics"] = t1_a
		model_info_json["level"] = '1'
		formatted_m.append(model_info_json)

	graph = {"nodes" : [], "links" : []}
	doc_graphlinks = {}
	linksdict = {}
	model_index = 0
	for x in range(0, 600, 100):
		
		save_fp = 'hdsr_models/' + fname[:-5] + '_' 
		lda1 = gensim.models.LdaMulticore.load(save_fp + str(x))
		doc_index = 0
		corpus_iterator = doc_iterator(fname, id2word, raw=False, min_count=min_count, daterange=daterange)
		for doc in corpus_iterator:
			topics_in_doc = lda1.get_document_topics(doc, minimum_probability=0.01)
			for t in topics_in_doc:
				model_topic_name = formatted_m[model_index]["name"] + ":" + str(t[0])
				if doc_index not in doc_graphlinks:
					doc_graphlinks[doc_index] = []

				doc_graphlinks[doc_index].append([model_topic_name, t[1]])
			doc_index+=1
		model_index += 1


	#print(doc_graphlinks)
	for x in doc_graphlinks:
		for i in range(0, len(doc_graphlinks[x]) - 1):

			mname = doc_graphlinks[x][i][0]
			for j in range(i + 1, len(doc_graphlinks[x])):
				dname = doc_graphlinks[x][j][0]

				linkname = mname + "&" + dname
				reverse_linkname = dname + "&" + mname
				if linkname in linksdict:
					linksdict[linkname]["weight"] += 1
					#linksdict[linkname]["docs"].append(x)
				elif reverse_linkname in linksdict:
					linksdict[reverse_linkname]["weight"] += 1
					#linksdict[reverse_linkname]["docs"].append(x)
				else:
					linksdict[linkname] = {
						"id" : linkname,
						"source" : mname,
						"target" : dname,
						"weight" : 1,
						#"docs" : [x]
					}
	graph["links"] = list(linksdict.values())

	base_nps = []
	for x in range(0, 600, 100):
		lda1 = gensim.models.LdaMulticore.load(save_fp + str(x))
		if x == 0:
			base_nps = lda1.get_topics()
		else:
			base_nps = np.concatenate((base_nps, lda1.get_topics()), axis=0)

	scaler = Normalizer()
	base_nps = scaler.fit_transform(base_nps)
	pca = PCA(n_components=2)
	pca_result = pca.fit_transform(base_nps)


	clustering = KMeans(n_clusters=ntopics, max_iter=10000, random_state=100).fit(base_nps)
	print("AP finished")
	
	data = []

	print_dict = dict()
	for t in clustering.labels_:
		print_dict[t] = []

	mcount = 0
	tcount = 0

	for x in range(0, 600, 100):
		lda1 = gensim.models.LdaMulticore.load(save_fp + str(x))
		model_obj = {}
		corpus_obj = {}

		model_obj["level"] = 1
		model_obj["top_words"] = []
		model_obj["locations"] = []
		model_obj["clusters"] = []
		model_obj["topic_weights"] = []
		model_obj["docs"] = []

		topics = lda1.show_topics(num_topics=ntopics, num_words=10, formatted=True)
		for t in lda1.state.sstats:
			score = 0
			for w in t: 
				score = score + w
			model_obj["topic_weights"].append(score)


		for t in topics:
			model_obj["docs"].append([])
			model_obj["top_words"].append(t[1])
			print_dict[clustering.labels_[tcount]].append(t[1])
			model_obj["locations"].append([pca_result[tcount][0].item(), pca_result[tcount][1].item()])
			model_obj["clusters"].append(clustering.labels_[tcount].item())
			tcount += 1


		mcount += 1
		dcount = 0
		corpus_iterator = doc_iterator(fname, id2word, raw=False, min_count=min_count, daterange=daterange)
		for doc in corpus_iterator:
			doc_topics = lda1.get_document_topics(doc)
			for topic_score in doc_topics:
				if topic_score[1] > .01:
					model_obj["docs"][topic_score[0]].append([[dcount, -1], topic_score[1].item()])

			dcount += 1
		print(dcount)

		data.append(model_obj)

	meta_data = open("hdsr-searches/" + fname, 'r')
	#the_meta = json.load(meta_data)

	meta_str = ''
	for the_meta in meta_data:
		meta = json.loads(the_meta)
		if in_date(meta, daterange=daterange) and min_count_func(meta, fname, min_count=min_count):
			the_str = '"","{}","{}","","","","{}","","{}"\n'.format(meta[2], meta[3][0], meta[7], meta[0])
			meta_str = meta_str + the_str

	multi_data = {
		"formatted" : graph,
		"formatted_proj" : data,
		"metadata" : meta_str
	}
	#fname = "q2_test.json"
	wf = open("hdsr_formatted/" + fname , 'w')

	json.dump(multi_data, wf)

	upload_str(json.dumps(multi_data), fname[:-5] +  "/multilevel_lda_formatted.json")


def count_occurrences(name):
	name_searches = {
		"H1N1-OR-Influenza_A.json" : ["H1N1","Influenza", "Influenza-a"],
		"Ebola-OR-EVD-OR-Ebola_Virus_Disease.json" : ["Ebola", "EVD", "Ebola Virus Disease"],
		"Zika.json" : ['Zika'],
		"Zika_m2.json" : ['Zika'],
		"MERS-OR-Middle_East_Respiratory_Syndrome.json" : ["MERS", "Middle East Respiratory Syndrome"],
		"SARS-OR-Severe_Acute_Respiratory_Syndrome.json" : ["SARS", "Severe Acute Respiratory Syndrome"], 
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2.json" : ["severe acute respiratory syndrome coronavirus 2", "covid-19", "sars-cov-2", "sars‐cov‐2", "sars cov‐2"],
		"HIV-OR-human_immunodeficiency_virus-OR-AIDS-OR-acquired_immunodeficiency_syndrome.json" : ["HIV", "human immunodeficiency virus","AIDS","acquired immunodeficiency syndrome"]
	}

	f = open('hdsr-searches/' + name ,'r')
	
	search_arr = []
	for w in name_searches[name]:
		search_arr
		print(w.lower())

	index = 0

	term_occurrences = {}
	for doc_str in f:
		doc = json.loads(doc_str)
		doc_text = doc[8].lower()
		doc_term_count = 0

		for w in name_searches[name]:
			term_count = doc_text.count(w.lower())
			doc_term_count += term_count

		if doc_term_count == 0:
			print(doc[0])
			print("0 OCCURRED")
			print(doc_text)
		
		if doc_term_count in term_occurrences:
			term_occurrences[doc_term_count] += 1
		else:
			term_occurrences[doc_term_count] = 1
		if index % 1000 == 0:
			print("counted " + str(index))
		index += 1

	terms_count_arr = []
	for x in term_occurrences:
		terms_count_arr.append([x, term_occurrences[x]])

		
	
	terms_count_arr = sorted(terms_count_arr, key=lambda a: a[0])

	print(name)
	thing = ""
	for x in terms_count_arr:
		thing = thing + "occurrences: {}, {}\n".format(x[0], x[1])

	wf = open('hdsr_counts/' +name[:-5] + '.csv', 'w')
	wf.write(thing)


def prep_date(the_date):
	ar = the_date.split("-")
	r = ar[0] + "-" + ar[1]
	return r
def count_dates(name, min_count=2, date_range=None):

	f = open('hdsr-searches/' + name ,'r')

	date_occurrences = {}
	count = 0
	for doc_str in f:
		doc = json.loads(doc_str)
		countmet = min_count_func(doc, name, min_count=min_count, filter_by_cov=True)
		in_range = in_date(doc, date_range) if date_range is not None else True
		formed_d = prep_date(doc[7])
		if countmet and in_range:
			if formed_d in date_occurrences:
				date_occurrences[formed_d] += 1
			else:
				date_occurrences[formed_d] = 1
			count += 1
	print(count)

if __name__ == '__main__':

	daterange = [datetime.datetime(1600, 1, 1), datetime.datetime(2030, 8, 1)]
	min_count=2

	daterange=None
	min_count=0

	#h1n1=10469
	#ebola=3115
	#mers=4566
	#sars=13762
	#zika=872
	#hiv=10236
	names = {
		"H1N1-OR-Influenza_A_m2.json" : "h1n1",
		"Ebola-OR-EVD-OR-Ebola_Virus_Disease_m2.json" : 'ebola',
		"10k_rand.json" : '10k_r',
		"Zika.json" : 'zika',
		"MERS-OR-Middle_East_Respiratory_Syndrome_m2.json" : 'mers',
		"SARS-OR-Severe_Acute_Respiratory_Syndrome_m2.json" : 'sars', 
		"Zika_m2.json" : 'zika',
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2.json" : 'covid',
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q1.json" : 'covid',
		"severe_acute_respiratory_syndrome_coronavirus_2-OR-covid-19-OR-sars_cov_2_q2.json" : 'covid',
		"HIV-OR-human_immunodeficiency_virus-OR-AIDS-OR-acquired_immunodeficiency_syndrome_m2.json" : 'hiv',
		"full_corpus.json" : "all"
	}

	for x in names:

		#count_dates(x, min_count=2, date_range=daterange)
		create_dict(x, min_count=min_count, daterange=daterange)
		do_eval(x, min_count=min_count, daterange=daterange)

		do_mlmom_models(x, date=daterange, min_count=min_count)
		create_mlmom_format(x, min_count=min_count, daterange=daterange)

		#count_occurrences(x)




	



