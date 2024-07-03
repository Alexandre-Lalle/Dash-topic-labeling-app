 # -*- coding: utf-8 -*-
 # dash==2.17.0
"""
Module doc string
"""

import dash
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import pickle
from dash.dependencies import Output, Input, State
from bertopic_model import load_model, load_features, visualize_documents


PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
PAGE_SIZE = 10

# Load the features once at startup
reduced_embeddings, topic_info, document_info, dataset, titles, abstracts = load_features()


"""
#  Somewhat helpful functions
"""

loaded_model = None
def get_model():
    global loaded_model
    if loaded_model is None:
        try:
            loaded_model = load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
    return loaded_model


def visualize_topics_():
    return get_model().visualize_topics(custom_labels=True, title="")

def visualize_documents_():
    return visualize_documents(topic_model=get_model(), docs=titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True, height=700 ,width=1050)

def visualize_barchart_():
    return get_model().visualize_barchart(custom_labels=True, title="")

def fig_datamapplot():
    # To improve loading speed, we created the datamap using bertopic_model `datamap_plot` function and saved it as an HTML file. 
    # This is because `datamapplot.create_interactive_plot` output isn't compatible with Dash and cannot be directly displayed.

    with open('datamapplot/fig_datamap_plot.html', 'r', encoding='utf-8') as file:
        iframe_content = file.read()
    fig = html.Iframe(srcDoc=iframe_content, style={"width": "100%", "height": "600px", "border": "none"})
        
    return fig


def get_topic_distribution(doc_id):

    with open('data/topic_distr.pkl', 'rb') as handle:
        topic_distr = pickle.load(handle)
    # topic_document_distribution
    #topic_distr, _ = loaded_model.approximate_distribution(abstracts[doc_id], window=8, stride=4)

    fig_visualize_distribution = loaded_model.visualize_distribution(topic_distr[doc_id], custom_labels=True)
 
    return fig_visualize_distribution

def get_token_distribution(doc_id):

    # For better performance, we saved topic_token_distr for the 10000 first abstracts
    with open('data/topic_token_distr.pickle', 'rb') as handle:
        topic_token_distr = pickle.load(handle)

    #Calculate the topic distributions on a token-level
    #topic_distr, topic_token_distr = loaded_model.approximate_distribution(abstracts[doc_id], calculate_tokens=True)

    #Visualize the token-level distributions
    df = loaded_model.visualize_approximate_distribution(abstracts[doc_id], topic_token_distr[doc_id])
    
    return df


"""
#  Page layout and contents
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px", style={'margin-left': '20px'})),
                    dbc.Col(
                        dbc.NavbarBrand("ArXiv Topic Explorer", className="ml-2",style={'text-decoration': 'none'}),
                        className="p-0",
                    ),
                ],
                align="center",
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)


LDA_PLOT = dcc.Loading(
    id="loading-lda-plot", children=[dcc.Graph(id="tsne-lda",figure=visualize_documents_())], type="default"
)

LDA_TABLE = html.Div(
    id="lda-table-block",
    children=[
        dcc.Loading(
            id="loading-lda-table",
            children=[
                dash_table.DataTable(
                    id="lda-table",
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "Text"},
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "min-width": "50%",
                        }
                    ],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(243, 246, 251)",
                        }
                    ],
                    style_cell={
                        "padding": "16px",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "max-width": "0",
                    },
                    style_header={"backgroundColor": "white", "fontWeight": "bold"},
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    filter_action="native",
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    columns=[],
                    data=[],
                )
            ],
            type="default",
        )
    ],
    style={"display": "none"},
)

LDA_PLOTS = [
    dbc.CardHeader(html.H5("Document Assignment Explorer")),
    dbc.Alert(
        "Not enough data to render BERTopic plots, please adjust the filters",
        id="no-data-alert-lda",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            html.P(
                "Click on a document point in the scatter to explore that specific document",
                className="mb-0",
            ),
            html.P(
                "(not affected by sample size or time frame selection)",
                style={"fontSize": 10, "font-weight": "lighter"},
            ),
            LDA_PLOT,
            html.Hr(),
            LDA_TABLE,
        ]
    ),
]

BARCHART_PLOT = [
    dbc.CardHeader(html.H5("Similar Topics Explorer")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(html.P("Search for topics using search terms : "), md=4),
                    dbc.Col(
                        [
                            dcc.Input(id="search-term-input", 
                                        type="text", 
                                        placeholder="Enter search terms",
                                        style={"width": "70%", "margin-right": "10px"}
                                        ),
                            dbc.Button(
                                children="Search",
                                id="search-button",
                                n_clicks=0,
                                className="ml-2",  # Add some margin to the left
                                #style={"height": "38px"}
                            ),
                        ],
                        md=6,
                        style={"display": "flex", "align-items": "center"}
                    ),
                ]
            ),
            dcc.Loading(
                    id="loading-barchart-plot", 
                    children=[dcc.Graph(id="barchart-sample",figure=visualize_barchart_())], 
                    type="default"
                )

        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOPIC_INFO = [
    dbc.CardHeader(html.H5("Data Overview")),
    dbc.CardBody(
        [
            dcc.Tabs([
                dcc.Tab(label='Raw Dataset', 
                        value='tab-1',
                        children=[
                            dcc.Loading(
                                id="loading-dataset",
                                children=[
                                    html.Br(),
                                    dash_table.DataTable(
                                        id='datatable-paging',
                                        columns=[
                                            {"name": i, "id": i} for i in dataset.columns
                                        ],
                                        page_current=0,
                                        page_size=PAGE_SIZE,
                                        page_action='custom',
                                        style_cell={
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'width': 'auto',
                                            #'maxWidth': 0,
                                            'minWidth': '50px',
                                            'maxWidth': '300px',
                                        },
                                        tooltip_data=[
                                            {
                                                column: {'value': str(value), 'type': 'markdown'}
                                                for column, value in row.items()
                                            } for row in dataset.to_dict('records')
                                        ],
                                        tooltip_duration=None,
                                        style_data_conditional=[
                                            {
                                                'if': {'row_index': 'odd'},
                                                'backgroundColor': '#f8f9fa',  # Light gray background for odd rows
                                                'color': 'black'  # Black text for odd rows
                                            },
                                            {
                                                'if': {'row_index': 'even'},
                                                'backgroundColor': 'white',  # White background for even rows
                                                'color': 'black'  # Black text for even rows
                                            }
                                        ],
                                    )
                                ],
                                type="default",
                            )
                        ],  
                    ),
                dcc.Tab(label='Topic Details', 
                        value='tab-2',
                        children=[
                            dcc.Loading(
                                id="loading-topic-info",
                                children=[
                                    html.Br(),
                                    dash_table.DataTable(
                                        id='datatable-topic-info',
                                        columns=[
                                            {"name": i, "id": i} for i in topic_info.columns
                                        ],
                                        page_current=0,
                                        page_size=PAGE_SIZE,
                                        page_action='custom',
                                        style_cell={
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'width': 'auto',
                                            'minWidth': '50px', 
                                            'maxWidth': 0,  
                                        },
                                        tooltip_data=[
                                            {
                                                column: {'value': str(value), 'type': 'markdown'}
                                                for column, value in row.items()
                                            } for row in topic_info.to_dict('records')
                                        ],
                                        tooltip_duration=None,
                                        style_data_conditional=[
                                            {
                                                'if': {'row_index': 'odd'},
                                                'backgroundColor': '#f8f9fa',  # Light gray background for odd rows
                                                'color': 'black'  # Black text for odd rows
                                            },
                                            {
                                                'if': {'row_index': 'even'},
                                                'backgroundColor': 'white',  # White background for even rows
                                                'color': 'black'  # Black text for even rows
                                            }
                                        ],
                                    )
                                ],
                                type="default",
                            )
                        ],  
                    ),
                dcc.Tab(label='Document Insights', 
                        value='tab-3',
                        children=[
                            dcc.Loading(
                                id="loading-docs-info",
                                children=[
                                    html.Br(),
                                    dash_table.DataTable(
                                        id='datatable-docs-info',
                                        columns=[
                                            {"name": i, "id": i} for i in document_info.columns
                                        ],
                                        page_current=0,
                                        page_size=PAGE_SIZE,
                                        page_action='custom',
                                        style_cell={
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'minWidth': '50px',
                                            'maxWidth': 0,
                                        },
                                        tooltip_data=[
                                            {
                                                column: {'value': str(value), 'type': 'markdown'}
                                                for column, value in row.items()
                                            } for row in document_info.to_dict('records')
                                        ],
                                        tooltip_duration=None,
                                        style_data_conditional=[
                                            {
                                                'if': {'row_index': 'odd'},
                                                'backgroundColor': '#f8f9fa',  # Light gray background for odd rows
                                                'color': 'black'  # Black text for odd rows
                                            },
                                            {
                                                'if': {'row_index': 'even'},
                                                'backgroundColor': 'white',  # White background for even rows
                                                'color': 'black'  # Black text for even rows
                                            }
                                        ],
                                    )
                                ],
                                type="default",
                            )
                        ],
                    ),
                ],
            ),   
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_PLOT = [
    dbc.CardHeader(html.H5("Interactive Topic Visualization")),
    dbc.CardBody(
        [
            dcc.Tabs([
                dcc.Tab(label='Data Map Plot', 
                        value='tab-1',
                        children=[
                            dcc.Loading(
                                id="loading-datamapplot",
                                children=[
                                    #dcc.Graph(id="bank-wordcloud"),
                                    html.Div(id="datamapplot-container", children=fig_datamapplot())
                                ],
                                type="default",
                            )
                        ],  
                    ),
                dcc.Tab(label='Intertopic Distance Map', 
                        value='tab-2',
                        children=[
                            dcc.Loading(
                                id="loading-intertopic-map",
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            dcc.Graph(id="intertopic-map-figure", figure=visualize_topics_()),
                                            #width=10,
                                            className="d-flex justify-content-center",
                                        )
                                    ),
                                ],
                                type="default",
                            )
                        ],
                    ),
                ],
            ),   
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOPIC_DISTRIBUTION = [
    dbc.CardHeader(html.H5("Topic Probability Distribution")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(html.P("Choose a document : "), md=4),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="topic_distribution_1",
                                options= [
                                    {"label": title, "value": i}  # Use i for index
                                    for i, title in enumerate(titles)
                                ],
                                value=0,
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.Br(),
            dcc.Tabs([
                dcc.Tab(label='Document level', 
                        value='tab-1',
                        children=[
                            dcc.Loading(
                                id="loading-distribution",
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            dcc.Graph(id="topic-distribution"),
                                            className="d-flex justify-content-center",
                                        )
                                    ),
                                ],
                                type="default",
                            )
                        ],  
                    ),
                dcc.Tab(label='Token level', 
                        value='tab-2',
                        children=[
                            dcc.Loading(
                                id="loading-token-distribution",
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            #html.Div(id='table-container'),
                                            html.Iframe(id='table-container', style={'width': '100%', 'height': '500px', 'border': 'none'}),
                                            #width=10,
                                            className="d-flex justify-content-center",
                                        )
                                    ),
                                ],
                                type="default",
                            )
                        ],
                    ),
                ],
            ),   
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(TOPIC_INFO)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_PLOT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(BARCHART_PLOT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOPIC_DISTRIBUTION)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 30, "marginBottom": 20}),
    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], serve_locally=True)
#server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

@app.callback(
    Output('barchart-sample', 'figure'),
    Input('search-button', 'n_clicks'),
    State('search-term-input', 'value')
)
def update_barchart(n_clicks, search_term):
    if n_clicks > 0 and search_term:
        similar_topics, _ = loaded_model.find_topics(search_term, top_n=6)
        if similar_topics:
            return loaded_model.visualize_barchart(topics=similar_topics, custom_labels=True, title="") 
        else:
            return dash.no_update 
    return visualize_barchart_() 


@app.callback(
    [Output("lda-table", "data"), Output("lda-table", "columns"),
    Output("lda-table", "filter_query"), Output("lda-table-block", "style")
    ],
    [Input("tsne-lda", "clickData")],
    [State("lda-table", "filter_query")],
)
def filter_table_on_scatter_click(doc_click, current_filter):

    columns = [{"name": i, "id": i} for i in document_info.columns]
    data = document_info.to_dict("records")


    """ TODO """
    if doc_click is not None:
        selected_doc = doc_click["points"][0]["hovertext"]

        # Extract the index as an integer
        parts = selected_doc.split(',')
        index_part = parts[0].strip()
        doc_id = int(index_part.split(':')[-1])

        #temp = selected_complaint = tsne_click["points"][0]
        #print("line 925 selected_complaint ===>",temp)

        if current_filter != "":
            filter_query = (
                "({ID} eq "
                + str(doc_id)
                + ") || ("
                + current_filter
                + ")"
            )
        else:
            filter_query = "{ID} eq " + str(doc_id)
        print("current_filter", current_filter)
        return (data, columns, filter_query, {"display": "block"})
    return [data, columns, "", {"display": "none"}]


@app.callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"))
def update_dataset_table(page_current,page_size):
    return dataset[:500].iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')

@app.callback(
    Output('datatable-topic-info', 'data'),
    Input('datatable-topic-info', "page_current"),
    Input('datatable-topic-info', "page_size"))
def update_dataset_table(page_current,page_size):
    return topic_info.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')

@app.callback(
    Output('datatable-docs-info', 'data'),
    Input('datatable-docs-info', "page_current"),
    Input('datatable-docs-info', "page_size"))
def update_dataset_table(page_current,page_size):
    return document_info[:500].iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')



@app.callback(
    [Output('topic-distribution', 'figure'),
     Output('table-container', 'srcDoc')],
    [Input('topic_distribution_1', 'value')]
)
def update_distributions(doc_id):
    
    doc_id = int(doc_id)
    topic_fig = get_topic_distribution(doc_id)
    df = get_token_distribution(doc_id)

    # Convert DataFrame to HTML
    html_df = df.to_html()

    # Create an HTML structure to embed in the iframe
    src_doc = f'''
    <html>
    <head>
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            border: 1px solid black;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    </head>
    <body>
    {html_df}
    </body>
    </html>
    '''
    return topic_fig, src_doc



if __name__ == "__main__":
    app.run_server(port=4050, debug=True) ##threaded=True

                   
                   
