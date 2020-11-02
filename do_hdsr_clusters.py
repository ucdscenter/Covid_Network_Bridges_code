import json
import pandas as pd



def create_mlmom_clusters(rfile="multilevel_lda_formatted.json", wfile="multilevel_clusters.csv"):
	f = open(rfile, "r")
	mlmom_formatted = json.load(f)


	data = mlmom_formatted["formatted_proj"]
	clusters = {}

	maxclust = 0

	metadata_str = open("10k_meta_2.csv", "r").read()
	metadata_str = metadata_str.replace("\xa0\xa0\n", "").split("\n")
	metadata_str = [x.replace('"', "").replace("**", "").split(",") for x in metadata_str]
	metadata_str = metadata_str[:len(metadata_str) - 1]

	for i, x in enumerate(metadata_str):
		if len(x) > 9:
			fixed_auths = ";".join(x[2:len(x)- 6])
			final_row = x[:2]
			final_row.append(fixed_auths)
			final_row = final_row + x[len(x) - 6:]
			metadata_str[i] = final_row

			
	
	model_i = 0
	for model in data:
		index_val = 0

		for tcluster in model["clusters"]:
			if tcluster not in clusters:
				clusters[tcluster] = {}
				if tcluster > maxclust:
					maxclust = tcluster
			for doc in model["docs"][index_val]:
				print(metadata_str[doc[0][0]])
				doc_id = metadata_str[doc[0][0]][8]
				if doc_id not in clusters[tcluster]:
					clusters[tcluster][doc_id] = doc[1]
				else:
					clusters[tcluster][doc_id] += doc[1]
			index_val += 1

	print("number of clusters:")
	print(maxclust)

	csv_string = "id, title, clusters\n"
	no_cluster_count = 0
	for doc_row in metadata_str:
		doc_id = doc_row[8]
		max_score = [0, -1]
		doc_scores = []
		for c in clusters:
			if doc_id in clusters[c]:
				"""if clusters[c][doc_id] > max_score[0]:
					max_score[1] = c
					max_score[0] = clusters[c][doc_id]
				"""
				doc_scores.append(str(c) + "&" + str(clusters[c][doc_id]))
		#row_string = doc_id + "," + doc_row[1] + "," + str(max_score[1]) + "\n"
		if len(doc_scores) == 0:
			no_cluster_count += 1
			doc_scores.append("-1&-1")
		row_string = doc_id + "," + doc_row[1] + "," + ";".join(doc_scores) + "\n"
		csv_string = csv_string + row_string
	csv_string = csv_string[:len(csv_string) - 1]
	
	csv_file = open(wfile, "w")
	csv_file.write(csv_string)
	print(no_cluster_count)
	return
	

def do_hdsr_clusters(rfile="10k_doc_clusters.csv", wfile="10k_3_clusters_hdsr.csv"):
	df = pd.read_csv("10k_doc_clusters.csv")
	ref_df = pd.read_csv('10k_Random_Cluster_Labels_Sheet1.csv')
	csv_header = "id, title, cluster1, cluster2\n"
	for x in range(0, df.shape[0]):
		scores = df.iloc[x]["clusters"].split(";")
		the_scores = []
		for thething in scores:
			the_arr = thething.split('&')
			the_arr[0] = int(the_arr[0])
			the_arr[1] = float(the_arr[1])
			the_scores.append(the_arr)
		the_scores.sort(key=lambda x: x[1], reverse=True)
		score_1 = the_scores[0][0] if len(the_scores) > 0 else -1
		score_2 = the_scores[1][0] if len(the_scores) > 1 else -1
		print(score_1)
		print(score_2)
		if score_1 is not -1:
			score_1 = ref_df[ref_df["cluster"] == int(score_1)]["NETWORK LOCATION"]
			print(score_1)
			score_1 = score_1.values[0]
		else:
			score_1 = "NONE"

		if score_2 is not -1:
			score_2 = ref_df[ref_df["cluster"] == int(score_2)]["NETWORK LOCATION"]
			score_2 = score_2.values[0]
		else:
			score_2 = "NONE"
		
		the_str = str(df.iloc[x]["id"]) + "," + str(df.iloc[x]["title"]) + "," + str(score_1) + "," + str(score_2) + "\n"
		csv_header = csv_header + the_str
	print(type(csv_header))
	wf = open(wfile, "w")
	wf.write(csv_header)
	return

if __name__ == "__main__":

	do_hdsr_clusters()

