import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pickle
import networkx as nx
import ast
import numpy as np
import plotly.express as px

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

def toggle_page():
    st.session_state.current_page = 'other' if st.session_state.current_page == 'main' else 'main'
    st.experimental_rerun()

def plot_stacked_bar_chart():
    # Convert 'Year' to numeric, if it's not already
    final_df['Year'] = pd.to_numeric(final_df['Year'], errors='coerce')
    # Group by 'Year' and 'Type of Publication', then get the size of each group
    yearly_type_counts = final_df.groupby(['Year', 'Type of Publication']).size().unstack().fillna(0)
    # Define a color mapping for the different types of publications
    color_mapping = {
        'Journal Article': 'blue',
        'Conference Paper': 'orange',
        'Book': 'green',
        'Book Chapter': 'red',
        'Informal Publication': 'purple',
        'Other': 'gray'
    }
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot each publication type in a stacked bar chart
    bottom_values = np.zeros(len(yearly_type_counts))
    for column in yearly_type_counts.columns:
        ax.bar(yearly_type_counts.index, yearly_type_counts[column], bottom=bottom_values, label=column, color=color_mapping.get(column, 'gray'))
        bottom_values += yearly_type_counts[column].values
    ax.set_title('Breakdown of Publication Types Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Publication Count', fontsize=14)
    ax.legend(title='Type of Publication')
    ax.set_xticks(yearly_type_counts.index)
    ax.set_xticklabels(yearly_type_counts.index, rotation=45)
    plt.tight_layout()
    return fig
def plot_line_chart():
    # Exclude the year 2024 from the dataframe if needed
    final_df['Year'] = pd.to_numeric(final_df['Year'], errors='coerce')
    yearly_type_counts = final_df.groupby(['Year', 'Type of Publication']).size().unstack().fillna(0)
    yearly_type_counts = yearly_type_counts[yearly_type_counts.index != 2024]
    # Create the line chart
    fig, ax = plt.subplots(figsize=(12, 8))
    # Highlight the region from 2019 to 2025 if needed
    ax.axvspan(2019, 2025, color='yellow', alpha=0.3, label='Highlighted Region (2019-2025)')
    # Plot each publication type
    for column in yearly_type_counts.columns:
        ax.plot(yearly_type_counts.index, yearly_type_counts[column], label=column, marker='o')
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Publication Count', fontsize=14)
    ax.set_xticks(yearly_type_counts.index)
    ax.set_xticklabels(yearly_type_counts.index, rotation=45)
    ax.legend(title='Type of Publication')
    plt.tight_layout()
    return fig
def plot_influence_bubble_chart():
    # Calculate total publications for each professor
    total_publications = final_df.groupby('Full Name').size().reset_index(name='Total Publications')
    # Merge this with the profile_df to get the citations for each professor
    influence_df = profile_df.merge(total_publications, on='Full Name', how='left')
    
    # Replace NaN values in 'Total Publications' with 0
    influence_df['Total Publications'] = influence_df['Total Publications'].fillna(0)
    
    # Normalize data for plotting
    max_citations = influence_df['Citations'].max()
    max_publications = influence_df['Total Publications'].max()
    influence_df['Normalized Citations'] = influence_df['Citations'] / max_citations
    influence_df['Normalized Publications'] = influence_df['Total Publications'] / max_publications
    
    # Use Plotly Express to create the bubble chart
    fig = px.scatter(
        influence_df,
        x='Normalized Publications',
        y='Normalized Citations',
        size='Total Publications',  # Ensure all values are numbers
        hover_name='Full Name',
        hover_data=['Citations', 'Total Publications'],
        title='Influence Bubble Chart',
        labels={
            'Normalized Publications': 'Normalized Total Publications',
            'Normalized Citations': 'Normalized Citations',
            'Total Publications': 'Total Publications (Size of Bubble)'
        }
    )
    fig.update_layout(
        height=1000,  # Set the height to a larger value to make the plot taller
        width=600,  # Optionally, adjust the width to maintain a certain aspect ratio
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
    
    return fig


# Initialize the session state for current_page if it does not exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'


if st.session_state.current_page == 'main':
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

    # Layout to push the button to the right
    col1, col2 = st.columns([0.95, 0.05])
    with col2:
        if st.button('Go to Other Page', key='btn_go_other'):
            toggle_page()

else:
    st.title("All Professors' Detail")
    # Split the other page into 3 columns
    col1, col_bubble_chart = st.columns([1, 2])  # Adjust the proportions as needed

    # First column with its content
    with col1:
        stacked_bar_chart = plot_stacked_bar_chart()
        st.pyplot(stacked_bar_chart)
        line_chart = plot_line_chart()
        st.pyplot(line_chart)
    
    # Bubble chart spanning the area of the second and third columns
    with col_bubble_chart:
        influence_bubble_chart = plot_influence_bubble_chart()
        st.plotly_chart(influence_bubble_chart, use_container_width=True)

    # Layout to push the button to the right
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button('Back to Main Dashboard', key='btn_go_main'):
            toggle_page()
