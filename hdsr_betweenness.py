import networkx as nx
import os
import json
import numpy as np


def return_link_obj(edge_obj):
	return (edge_obj["source"], edge_obj["target"], {
			"score" : 1 / edge_obj["weight"],
			"id" : edge_obj["id"]
		})

def get_edge_iqrs(data):
	edge_scores = []
	for edge in data['formatted']["links"]:
		edge_scores.append(edge["weight"])

	percentiles = np.percentile(edge_scores, [0, 50, 99])
	return [10000000000000]
	return percentiles


def create_graph(data, limit_score=1000):
	g = nx.Graph()
	for model_no in range(0,len(data["formatted_proj"])):
		for i in range(0, len(data["formatted_proj"][model_no]["clusters"])):
			g.add_node("m" + str(model_no) + ":" + str(i), 
				cluster=data["formatted_proj"][model_no]["clusters"][i], 
				top_words=data["formatted_proj"][model_no]["top_words"][i], 
				weight=data["formatted_proj"][model_no]["topic_weights"][i])
	#filtered_edges = list(filter(lambda x: False if x["weight"] < limit_score else True, data["formatted"]["links"]))
	filtered_edges = data["formatted"]["links"]
	formatted_edges = list(map(lambda x: return_link_obj(x), filtered_edges))
	
	g.add_edges_from(formatted_edges)
	return g

def get_betweeness(graph_data):
	btw_centrality = nx.betweenness_centrality(graph_data, weight="score")
	return btw_centrality


def write_cluster_csv(btw_data, graph_data, file_pre):
	clusters = {}
	iqr_vals = []
	for q in btw_data.keys():
		iqr_vals.append(q)

	for n in graph_data[iqr_vals[0]].nodes:
		if graph_data[iqr_vals[0]].nodes[n]["cluster"] not in clusters:
			clusters[graph_data[iqr_vals[0]].nodes[n]["cluster"]] = 1
		else:
			clusters[graph_data[iqr_vals[0]].nodes[n]["cluster"]] += 1
	retstr = 'cluster, size, group_betweenness_centrality_25, group_betweenness_centrality_50, group_betweenness_centrality_75\n'
	for c in clusters:
		cnodes = [x for x,y in graph_data[iqr_vals[0]].nodes(data=True) if y['cluster']== c]
		c_obj = {
			"cluster" : c,
			"size" : clusters[c],
			"group_betweenness_25" : nx.group_betweenness_centrality(graph_data[iqr_vals[0]], cnodes, weight="score"),
			"group_betweenness_50" : 0,#nx.group_betweenness_centrality(graph_data[iqr_vals[1]], cnodes, weight="score"),
			"group_betweenness_75" : 0#nx.group_betweenness_centrality(graph_data[iqr_vals[2]], cnodes, weight="score")
		}
		lstr = '{}, {}, {}, {}, {}\n'.format(c_obj["cluster"], c_obj["size"], c_obj["group_betweenness_25"], c_obj["group_betweenness_50"], c_obj["group_betweenness_75"])
		retstr = retstr + lstr

	with open('centrality_files/' + file_pre + "_clusters.csv", 'w') as wf:
		wf.write(retstr)
	return

def write_topic_csv(btw_data, graph_data, file_pre):
	retlist = []
	iqr_vals = []
	for q in btw_data.keys():
		iqr_vals.append(q)
	for key in btw_data[iqr_vals[0]]:
		node_obj = {
			"id" : key,
			"top_words" : graph_data[iqr_vals[0]].nodes[key]["top_words"],
			"weight" : graph_data[iqr_vals[0]].nodes[key]["weight"],
			"cluster" : graph_data[iqr_vals[0]].nodes[key]["cluster"],
			"centrality_25" : btw_data[iqr_vals[0]][key],
			"centrality_50" : 0,#btw_data[iqr_vals[1]][key],
			"centrality_75" : 0#btw_data[iqr_vals[2]][key]
		}
		retlist.append(node_obj)


	retlist = sorted(retlist, key=lambda a: a["centrality_25"], reverse=True)
	rstr = 'topic_id, top_words, weight, cluster, centrality\n'

	for n in retlist:
		lstr = '{}, {}, {}, {}, {}\n'.format(n["id"], n["top_words"], n["weight"], n["cluster"], n["centrality_25"])
		rstr = rstr + lstr
	with open('centrality_files/' + file_pre + "_topics.csv", 'w') as wf:
		wf.write(rstr)
	return


if __name__ == '__main__':
	#f = 'SARS-OR-Severe_Acute_Respiratory_Syndrome_m2.json'
	#f = 'Zika_m2.json'
	#fp = 'hdsr_used_formatted/' + f
	for f in os.listdir('hdsr_used_formatted'):
		fp = 'hdsr_used_formatted/' + f
		print(fp)
		fp_d = json.load(open(fp, 'r'))
		iqrs = get_edge_iqrs(fp_d)
		print(iqrs)
		graphs_scores = {}
		graphs_data = {}
		index = 0
		for q in iqrs:
			graphs_data[q] = create_graph(fp_d, limit_score=q)
			graphs_scores[q] = get_betweeness(graphs_data[q])
			index += 1
		write_topic_csv(graphs_scores, graphs_data, f[:-5])
		#write_cluster_csv(graphs_scores, graphs_data, f[:-5])