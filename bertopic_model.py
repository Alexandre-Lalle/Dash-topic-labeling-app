from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from umap import UMAP
from typing import List, Union
import datamapplot


def load_model():

    # Define embedding model
    embedding_model = SentenceTransformer("BAAI/bge-small-en")

    # Load model and add embedding model
    loaded_model = BERTopic.load("data/model", embedding_model=embedding_model)
    print("==> BERTopic model loaded successfully !!!")
    
    return loaded_model


def load_features():

    print("===> loading features...")
    with open('data/reduced_embeddings.pickle', 'rb') as handle:
        reduced_embeddings = pickle.load(handle)
    

    document_info = pd.read_csv('data/document_info.csv')
    # Create a new column 'ID' with the index values
    document_info['ID'] = document_info.index
    # Reorder columns with 'ID' as the first column
    column_order = ['ID'] + list(document_info.columns.difference(['ID']))
    document_info = document_info[column_order]
    # Drop unwanted columns (assuming they are case-sensitive)
    document_info.drop(columns=['Name', 'Representative_Docs', 'Top_n_words', 'Representative_document'], inplace=True)
    
    dataset = pd.read_csv('data/dataset.csv')
    dataset['index'] = range(0, len(dataset))
    order = ['index', 'title','abstract']
    dataset = dataset[order]

    abstracts = pd.read_csv('data/dataset.csv')["abstract"]
    abstracts = abstracts.astype(str).tolist()
    titles = pd.read_csv('data/dataset.csv')["title"]
    titles = titles.astype(str).tolist()

    # Add 'Title' column with titles from 'dataset.csv'
    document_info['Title'] = titles

    # Reorder columns based on your desired order
    desired_order = ['ID', 'Title', 'Document', 'Topic', 'Probability', 'Representation', 'Flan-T5', 'Llama2']
    document_info = document_info[desired_order]

    topic_info = pd.read_csv('data/topic_info.csv')
    topic_info['ID'] = range(0, len(topic_info))
    # Reorder the columns to place 'ID' first
    columns = ['ID'] + [col for col in topic_info.columns if col != 'ID']
    topic_info = topic_info[columns]

    return reduced_embeddings, topic_info, document_info, dataset,  titles, abstracts



def visualize_documents(topic_model,
                        docs: List[str],
                        topics: List[int] = None,
                        embeddings: np.ndarray = None,
                        reduced_embeddings: np.ndarray = None,
                        sample: float = None,
                        hide_annotations: bool = False,
                        hide_document_hover: bool = False,
                        custom_labels: Union[bool, str] = False,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750):
    
    """ Visualize documents and their topics in 2D"""

    
    topic_per_doc = topic_model.topics_

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    # df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    # df["doc"] = [docs[index] for index in indices]
    # df["topic"] = [topic_per_doc[index] for index in indices]

    df = pd.DataFrame({
    "index": indices,  # Add this line to store original indices
    "topic": np.array(topic_per_doc)[indices],
    "doc": [docs[index] for index in indices]
    })

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = [f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""

    # Include index in hovertext
    selection['hovertext'] = selection.apply(lambda row: f"Index: {row['index']}, Doc: {row['doc']}", axis=1)

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection['hovertext'],
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            # Include index in hovertext
            selection['hovertext'] = selection.apply(lambda row: f"Index: {row['index']}, Doc: {row['doc']}", axis=1)

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection['hovertext'],
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def datamap_plot():

    arxivml_data_map = np.load("datamapplot/arxiv_ml_data_map.npy")
    arxivml_label_layers = []
    for layer_num in range(5):
        arxivml_label_layers.append(
            np.load(f"datamapplot/arxiv_ml_layer{layer_num}_cluster_labels.npy", allow_pickle=True)
        )
    arxivml_hover_data = np.load("datamapplot/arxiv_ml_hover_data.npy", allow_pickle=True)

    plot = datamapplot.create_interactive_plot(
        arxivml_data_map,
        arxivml_label_layers[0],
        arxivml_label_layers[2],
        arxivml_label_layers[4],
        hover_text = arxivml_hover_data,
        initial_zoom_fraction=0.75,
        font_family="Playfair Display SC",
        #title="ArXiv Machine Learning Landscape",
        #sub_title="A data map of papers from the Machine Learning section of ArXiv",
        logo="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/512px-ArXiv_logo_2022.svg.png",
        logo_width=128,
        on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
        enable_search=True,
        #darkmode=True,
    )

    return plot