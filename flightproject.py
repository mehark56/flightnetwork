import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms import community

# Load the datasets
def load_data(routes_path, airports_path, airlines_path):
    routes = pd.read_csv(routes_path)
    airports = pd.read_csv(airports_path)
    airlines = pd.read_csv(airlines_path)
    return routes, airports, airlines

# Check the column names
def check_columns(routes, airports, airlines):
    print("Routes Columns:", routes.columns)
    print("Airports Columns:", airports.columns)
    print("Airlines Columns:", airlines.columns)

# Rename columns
def rename_columns(df, columns_mapping):
    df.rename(columns=columns_mapping, inplace=True)
    return df

# Preprocess the data
def preprocess_data(routes, airports, source_col, dest_col, iata_col):
    routes = routes.dropna(subset=[source_col, dest_col])
    airports = airports.dropna(subset=[iata_col])
    return routes, airports

# Create the graph
def create_graph(routes, airports, source_col, dest_col, iata_col):
    G = nx.DiGraph()
    
    for _, row in airports.iterrows():
        G.add_node(row[iata_col], name=row['Airport Name'], city=row['City'], country=row['Country'])
    
    for _, row in routes.iterrows():
        if row[source_col] in G and row[dest_col] in G:
            G.add_edge(row[source_col], row[dest_col], airline=row['Airline'])
    
    return G

# Perform network analysis
def network_analysis(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    return {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality
    }

# Calculate additional metrics
def calculate_metrics(G):
    network_density = nx.density(G)
    avg_shortest_path_length = nx.average_shortest_path_length(G.to_undirected())
    clustering_coefficient = nx.average_clustering(G.to_undirected())
    diameter = nx.diameter(G.to_undirected())

    return {
        'density': network_density,
        'avg_shortest_path': avg_shortest_path_length,
        'clustering_coefficient': clustering_coefficient,
        'diameter': diameter
    }

# Visualize the network with different layouts
def visualize_network(G, layout='spring', centrality=None):
    plt.figure(figsize=(15, 10))
    
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.1)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    
    if centrality:
        node_color = [centrality[n] for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, node_color=node_color, cmap=plt.cm.viridis)
    else:
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=10)
    
    plt.title(f"Flight Network ({layout.capitalize()} Layout)")
    plt.show()

# Detect and visualize communities
def detect_communities(G):
    communities = community.greedy_modularity_communities(G.to_undirected())
    modularity = community.modularity(G.to_undirected(), communities)
    
    # Assign community numbers to nodes
    community_map = {}
    for i, community_set in enumerate(communities):
        for node in community_set:
            community_map[node] = i
    
    # Visualization
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, node_color=[community_map[n] for n in G.nodes()], cmap=plt.cm.rainbow)
    plt.title(f"Communities in Flight Network (Modularity: {modularity:.4f})")
    plt.show()

# Main function to run the analysis
def main():
    routes_path = 'path_to_routes_dataset.csv'
    airports_path = 'path_to_airports_dataset.csv'
    airlines_path = 'path_to_airlines_dataset.csv'
    
    routes, airports, airlines = load_data(routes_path, airports_path, airlines_path)
    
    # Check the column names
    check_columns(routes, airports, airlines)
    
    # Define the new column names mapping
    routes_columns_mapping = {
        'source airport': 'Source Airport',
        'destination airport': 'Destination Airport',
        'airline': 'Airline'
    }
    
    airports_columns_mapping = {
        'name': 'Airport Name',
        'city': 'City',
        'country': 'Country',
        'iata': 'IATA Code'
    }
    
    airlines_columns_mapping = {
        'name': 'Airline Name',
        'iata': 'Airline IATA Code',
        'country': 'Airline Country'
    }
    
    # Rename columns
    routes = rename_columns(routes, routes_columns_mapping)
    airports = rename_columns(airports, airports_columns_mapping)
    airlines = rename_columns(airlines, airlines_columns_mapping)
    
    # Update these variables with the correct column names after renaming
    source_col = 'Source Airport'
    dest_col = 'Destination Airport'
    iata_col = 'IATA Code'
    
    # Preprocess the data with correct column names
    routes, airports = preprocess_data(routes, airports, source_col, dest_col, iata_col)
    
    # Create graph
    G = create_graph(routes, airports, source_col, dest_col, iata_col)
    
    # Network analysis
    centrality_metrics = network_analysis(G)
    metrics = calculate_metrics(G)
    
    # Visualization
    visualize_network(G, layout='spring', centrality=centrality_metrics['degree'])
    visualize_network(G, layout='circular', centrality=centrality_metrics['betweenness'])
    visualize_network(G, layout='shell', centrality=centrality_metrics['closeness'])
    
    # Detect communities
    detect_communities(G)
    
    # Display the results
    print("Network Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value}")
    
    print("\nTop 10 Airports by Eigenvector Centrality:")
    top_10_airports = sorted(centrality_metrics['eigenvector'].items(), key=lambda x: x[1], reverse=True)[:10]
    for airport, centrality in top_10_airports:
        print(f"{airport}: {centrality:.4f}")

if __name__ == "__main__":
    main()

