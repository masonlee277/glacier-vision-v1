import argparse
from math import sqrt
import matplotlib.pyplot as plt
import networkx 
import numpy as np
import rasterio
from scipy.ndimage import label
from skimage.morphology import skeletonize
from skimage.util import invert

# Defaults if __name__ != '__main__'
args = argparse.Namespace(file='art.png', verbosity=1, invert=None)

def main():
	"""
	Main function to execute the connectivity calculation and print the results.
	"""
	# Path to the binary image
	binary_image_path = args.file
	
	# Load the binary image
	binary_image = load_binary_image(binary_image_path)
	if args.invert is None:
		image = binary_image
	else:
		image = invert(binary_image)
	skeleton = skeletonize(image)

	# display results
	if args.verbosity is not None:
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
		
		ax = axes.ravel()
		
		ax[0].imshow(image, cmap=plt.cm.gray)
		ax[0].axis('off')
		ax[0].set_title('original', fontsize=20)
		
		ax[1].imshow(skeleton, cmap=plt.cm.gray)
		ax[1].axis('off')
		ax[1].set_title('skeleton', fontsize=20)
		
		fig.tight_layout()
		plt.imsave(f"sk{args.file}", skeleton)  # save scikit skeleton
	
	# Convert the binary image to a graph
	Graph, labeled_array = binary_image_to_graph(skeleton, False)

	# Calculate connectivity metrics
	metrics = calculate_connectivity_metrics(Graph, skeleton)
	
	# Print metrics
	print("---Metrics---")
	print(f"Number of Connected Components (b0): {metrics['num_components']}")
	print(f"Mean River Segment Length: {metrics['mean_length']}")
	print(f"Total Network Length: {metrics['total_length']}")
	print(f"Number of Endpoints: {metrics['num_endpoints']}")
	"""
	Old:
	print(f"Number of Edges: {metrics['num_edges']}")
	print(f"Largest Component Size: {metrics['largest_component_size']}")
	print(f"Number of Nodes: {metrics['num_nodes']}")
	print(f"Average Number of Node Connections: {metrics['avg_node_connections']:.2f}")
	print(f"Number of Components: {metrics['num_components']}")
	print(f"Connectivity Metric: {metrics['connectivity']:.2f}")
	"""

def load_binary_image(binary_image_path):
	"""Load binary image from a GeoTIFF file."""
	try:
		with rasterio.open(binary_image_path) as src:
			binary_image = src.read(1)  # Read the first band
		return binary_image
	except Exception as e:
		print(f"Error loading image: {e}")
		raise

def binary_image_to_graph(binary_image, cardinal_adjacency=True):
	"""Convert binary image to a graph representation."""
	if args.verbosity is not None:
		print("Converting binary image to graph...")
	
	# Label connected components
	labeled_array, num_features = label(binary_image == 0)
	#pdb.set_trace(header="labeled_array?")
	if args.verbosity is not None:
		print(f"Labeled array shape: {labeled_array.shape}")
		print(f"Number of features: {num_features}")
	
	# Create a graph
	Graph = networkx.Graph()
	if cardinal_adjacency:
		relative_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	else:
		relative_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
	
	# Add nodes and edges to the graph
	for y in range(binary_image.shape[0]):
		for x in range(binary_image.shape[1]):
			if binary_image[y, x] == 1:  # Only consider water pixels
				Graph.add_node((y, x))
				# Add edges to adjacent nodes
				for dy, dx in relative_neighbors:
					ny, nx = y + dy, x + dx
					if 0 <= ny < binary_image.shape[0] and 0 <= nx < binary_image.shape[1]:
						if binary_image[ny, nx] == 1:
						
							Graph.add_edge((y, x), (ny, nx))
				
	if args.verbosity is not None:
		print(f"Number of nodes in graph: {Graph.number_of_nodes()}")
		print(f"Number of edges in graph: {Graph.number_of_edges()}")
	
	return Graph, labeled_array

def get_component_lengths_and_endpoints(Graph, skeleton):
	"""
	Returns list of estimated component lengths (sum of all connected segments)
		Cardinally adjacent pixels add 1 to the length
		Diagonally adjacent pixels add sqrt(2) to the length
	Graph: generated from skeletonized image, cardinal_adjacency=False
	"""
	component_lengths = []
	components_endpoints = []
	nodes = set(Graph.nodes)  # Unvisited water pixels
	next_edges = []  # Connections to adjacent water pixels

	while True:
		if next_edges:
			# Continue with one of the next nodes in the same component
			prev_node, curr_node = next_edges.pop()
			new_neighbor_cnt = 0  # How many new pixels it connects to
		else:
			try:
				# Start somewhere in a new component
				prev_node = nodes.pop()
				curr_node = prev_node  # Distance of 0
				component_lengths.append(0)
				components_endpoints.append([])
				new_neighbor_cnt = -1  # For endpoint classification
			except KeyError:
				break

		# Add the distance from the previous node to the current node
		if ((curr_node[0] - prev_node[0] != 0) and
			(curr_node[1] - prev_node[1] != 0)):  # If bordering on corner
			component_lengths[-1] += sqrt(2)
		elif ((curr_node[0] - prev_node[0] != 0) or
			  (curr_node[1] - prev_node[1] != 0)):  # If bordering on side
			component_lengths[-1] += 1
		else:
			pass  # If in the same place, add 0 distance

		# Queue the next edges to all neighboring unvisited water pixels 
		for next_node in Graph.neighbors(curr_node):
			try:
				nodes.remove(next_node)
			except KeyError:
				continue
			next_edges.append((curr_node, next_node))
			new_neighbor_cnt += 1

		# Track endpoints not on the image edge
		# Mask 2 pixels inward b/c scikit skeleton sometimes shrinks from edge
		# Check skeleton to disqualify bifurcation sites and cut corners
		if (new_neighbor_cnt <= 0 and  # possible endpts
			1 < curr_node[0] < skeleton.shape[0] - 2 and  # away from edge
			1 < curr_node[1] < skeleton.shape[1] - 2 and
			np.sum(skeleton[curr_node[0]-1:curr_node[0]+2,  # define endpt
							curr_node[1]-1:curr_node[1]+2]) <= 2):
			# e.g. sum([[False, False, False],
            #           [False, True,  True],
			#		    [True,  False, False]]) returns 3: not an endpt
			components_endpoints[-1].append(curr_node)

	return component_lengths, components_endpoints

def calculate_connectivity_metrics(Graph, skeleton):
	"""Calculate connectivity metrics for the graph."""
	try:
		num_edges = Graph.number_of_edges()
		num_nodes = Graph.number_of_nodes()
		num_components = len(list(networkx.connected_components(Graph)))
		
		# Calculate largest component size
		largest_component_size = max(len(c) for c in networkx.connected_components(Graph)) if num_nodes > 0 else 0
		#comps = list(networkx.connected_components(Graph))
		#print([len(c) for c in comps])
		#pdb.set_trace()
		
		# Calculate average number of node connections
		avg_node_connections = sum(deg for node, deg in Graph.degree()) / num_nodes if num_nodes > 0 else 0
		
		# Calculate connectivity metric
		connectivity = (num_edges - largest_component_size**2) / (num_nodes) * (1 + avg_node_connections / num_components) if num_nodes > 0 else 0

		# Calculate component length metrics
		component_lengths, components_endpoints = get_component_lengths_and_endpoints(Graph, skeleton)
		total_length = sum(component_lengths)
		mean_length = total_length / len(component_lengths)
		num_endpoints = sum((len(endpts) for endpts in components_endpoints))
		if args.verbosity is not None and args.verbosity >= 2:
			print(f"Component lengths: {component_lengths}")
			print(f"Components_endpoints: {components_endpoints}")

		return {
			'num_edges': num_edges,
			'largest_component_size': largest_component_size,
			'num_nodes': num_nodes,
			'avg_node_connections': avg_node_connections,
			'num_components': num_components,
			'connectivity': connectivity,
			'mean_length': mean_length,
			'total_length': total_length,
			'num_endpoints': num_endpoints
		}
	except Exception as e:
		print(f"Error calculating metrics: {e}")
		raise


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="ice-shelf-connectivity.py",
		description="An evaluation script for segconnector.")
	add = parser.add_argument
	add("file", help="input image path")
	add("-v", "--verbosity", action="count")
	add("-i", "--invert", nargs="?")
	args = parser.parse_args()

	main()

