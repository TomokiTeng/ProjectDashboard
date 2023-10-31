import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pickle
import networkx as nx
import ast

# Dashboard vibe
st.set_page_config(layout="wide")

# Load the data
@st.cache_data 
def load_data():
    return pd.read_csv('dblp_data.csv'), pd.read_csv('profile_data.csv'), pd.read_csv('sim_data.csv', index_col=0) 
final_df, profile_df, sim_df = load_data()

@st.cache_data
def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

G_full = load_graph('Tomoki_Teng_graph_data.pkl')

def parse_coauthors(coauthors_str):
    # Using ast.literal_eval safely
    return ast.literal_eval(coauthors_str)
# Convert the 'List of Coauthors' to lists
final_df['List of Coauthors'] = final_df['List of Coauthors'].apply(parse_coauthors)

def plot_graph_for_professor(prof_name):
    subset = final_df[final_df['Full Name'] == prof_name]
    G = nx.Graph()
    
    # Add nodes
    for _, row in subset.iterrows():
        G.add_node(row['Full Name'], color='blue')  # NTU professor (e.g., from SCSE)
        for coauthor in row['List of Coauthors']:
            if coauthor in final_df['Full Name'].values:
                G.add_node(coauthor, color='blue')
            else:  # If coauthor is outside NTU
                G.add_node(coauthor, color='red')
            G.add_edge(row['Full Name'], coauthor)
    
    # Draw graph
    pos = nx.spring_layout(G)
    colors = [G.nodes[node]['color'] for node in G.nodes]
    
    fig = plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
    nx.draw_networkx_edges(G, pos)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10)
        
    return fig


# Set up the layout for 4 columns (5 columns is too small)
cols = st.columns(4)

# First column
with cols[0].container():
    cols[0].header('Profile Details')
    # Select a professor
    selected_prof = cols[0].selectbox('Select a Professor:', profile_df['Full Name'].unique())

    st.title(f'Profile of {selected_prof}')
    prof_info = profile_df[profile_df['Full Name'] == selected_prof].iloc[0]

    for label, value in [("Email", prof_info["Email"]),
                         ("Website", prof_info["Website URL"]),
                         ("Total Publications", len(final_df[final_df["Full Name"] == selected_prof])),
                         ("Citations", prof_info["Citations"]),
                         ("Education", prof_info["Education"])]:
        cols[0].markdown(
            f'<div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin-bottom:10px"><b>{label}:</b> {value}</div>',
            unsafe_allow_html=True)

    # Expander Experience section
    experience_expander = cols[0].expander('Experience')
    experience_expander.write(prof_info["Research_Interests"])

    # Expander Biography section
    biography_expander = cols[0].expander('Biography')
    biography_expander.write(prof_info["Biography"])

    # Conferences and CORE Rank
    cols[0].header('Conferences')
    # Filtering the data properly
    selected_prof_conferences = final_df[(final_df['Full Name'] == selected_prof) & (final_df['CORE Rank'].isin(['A*', 'A']))]
    # Creating a table for Venue and CORE Rank
    conference_table = selected_prof_conferences[['Venue', 'CORE Rank']].drop_duplicates().reset_index(drop=True)
    cols[0].table(conference_table)

# Second column
cols[1].header('Common Research Interests')
interests_text = ' '.join(profile_df['Research_Interests'].dropna())
wordcloud = WordCloud(width=800, height=400).generate(interests_text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
cols[1].pyplot(fig)

# Publications Breakdown
cols[1].header('Publications Type Breakdown')
# Filter the data for the selected professor
selected_prof_data = final_df[final_df['Full Name'] == selected_prof]
# Create a pivot table for the filtered data
pivot_df = selected_prof_data.pivot_table(index='Year', columns='Type of Publication', aggfunc='size', fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
pivot_df.plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Number of Publications')
cols[1].pyplot(fig)

# Conference CORE Rank Breakdown
selected_prof_conferences = final_df[final_df['Full Name'] == selected_prof]
# Replace other CORE ranks with 'Others'
selected_prof_conferences = selected_prof_conferences.replace(
    to_replace=[rank for rank in selected_prof_conferences['CORE Rank'].unique() if rank not in ['A*', 'A', 'B', 'C']],
    value='Others'
)
# Pivot table for the filtered data
pivot_df = selected_prof_conferences.pivot_table(index='CORE Rank', columns='Type of Publication', aggfunc='size', fill_value=0)
# Reorder the index and columns properly
core_order = ['Others', 'C', 'A', 'B', 'A*']
type_order = pivot_df.columns.tolist()  # Get the order of publication types prev plot
pivot_df = pivot_df.reindex(core_order).reindex(type_order, axis=1).fillna(0)
# Get the colors prev plot
colors = plt.cm.tab10.colors
# CORE Rank breakdown
fig, ax = plt.subplots(figsize=(8, 5))
pivot_df.plot(kind='barh', stacked=True, ax=ax, color=colors[:len(type_order)])  # Use the first N colors from the color cycle
ax.set_xlabel('Number of Publications')
ax.set_ylabel('CORE Rank')
cols[1].pyplot(fig)



# Third column
cols[2].header('Top Co-authors')
selected_prof_coauthors = final_df[final_df['Full Name'] == selected_prof]['List of Coauthors']
coauthors_counter = Counter()
for coauthors_list in selected_prof_coauthors:
    coauthors_counter.update(coauthors_list)
# Remove the selected professor's own name from the counter
coauthors_counter.pop(selected_prof, None)
# Getting top 10 co-authors
top_coauthors = dict(coauthors_counter.most_common(10))
# Determine colors based on whether the co-author is inside or outside NTU
colors = ['blue' if name in final_df['Full Name'].values else 'red' for name in top_coauthors.keys()]
fig, ax = plt.subplots(figsize=(10, 5.6))
sns.barplot(x=list(top_coauthors.values()), y=list(top_coauthors.keys()), ax=ax, palette=colors)
ax.set_xlabel('Number of Collaborations')
cols[2].pyplot(fig)

# Coauthor Relationship Graph
with cols[2].container():
    cols[2].header('Coauthor Relationship')
    fig = plot_graph_for_professor(selected_prof)
    cols[2].pyplot(fig)


# Fourth column
with cols[3].container():
    cols[3].header('Similarity Network Graph')
    # Initialize graph
    G = nx.Graph()
    # Add nodes
    for name in sim_df.index:
        G.add_node(name)
    # Add edges with cosine similarity as weight
    threshold = 0.7
    for i, name1 in enumerate(sim_df.index):
        for j, name2 in enumerate(sim_df.columns):
            similarity = sim_df.iloc[i, j]

            if similarity > threshold and name1 != name2 and similarity < 1:  # Added condition to avoid self-similarity of 1
                G.add_edge(name1, name2, weight=similarity)
    # Remove nodes with low degree
    degree_threshold = 2
    low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree <= degree_threshold]
    G.remove_nodes_from(low_degree_nodes)
    # Draw the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.5)
    node_sizes = [G.degree(node) * 200 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_sizes, font_size=10, alpha=0.7, width=[G[u][v]['weight'] for u,v in G.edges()], ax=ax)
    cols[3].pyplot(fig)
    
    # Similar Profiles
    nested_cols = cols[3].columns(2)
    # User input for X top similar profiles to display
    top_x = nested_cols[0].number_input('Select top X similar profiles:', min_value=1, max_value=len(sim_df)-1, value=5)
    # Filtering the top X most similar profiles
    selected_prof_similarities = sim_df.loc[selected_prof].sort_values(ascending=False).drop(selected_prof)[0:top_x]
    # Display the top X most similar profiles
    nested_cols[1].write(f'Top {top_x} profiles most similar to {selected_prof}:')
    nested_cols[1].write(selected_prof_similarities)

    # Venue Distribution
    cols[3].header('Venue Distribution')
    # Filter the data
    selected_prof_venue_data = final_df[final_df['Full Name'] == selected_prof]
    # Create a pivot table for the filtered data
    pivot_venue_df = selected_prof_venue_data.pivot_table(
        index='Venue', 
        columns='Type of Publication', 
        aggfunc='size', 
        fill_value=0
    ).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_venue_df.plot(kind='barh', stacked=True, ax=ax)
    ax.set_xlabel('Venue')  # Swap x and y labels
    ax.set_title('Number of Publications')
    cols[3].pyplot(fig)