# Dash-Topic-Labeling-App



## Overview

This repository contains a web application prototype that demonstrates the practical application of our LLM-based topic labeling method. Our method leverages LLMs like Llama 2 and Flan-T5 to generate more accurate and descriptive topic labels, enhancing the interpretability of topics derived from document collections. This prototype serves as a concrete example of how our proposed solution can be utilized in real-world scenarios.

## Technologies Used

- **Plotly Dash**: A powerful Python framework for building interactive web applications, known for its capability to create rich and responsive data visualizations.
- **BERTopic**: Used for topic modeling.
- **Llama 2 and Flan-T5**: Used for generating topic labels.

## Dataset

The prototype utilizes the arXiv dataset, comprising approximately 118,000 abstracts of computer science research papers from arXiv. These abstracts offer a diverse range of scientific disciplines, making them an ideal source for conducting comprehensive topic modeling. The well-structured nature of the text facilitates effective natural language processing tasks.

## Application Features

The web application integrates the BERTopic model into the Dash framework to create various interactive functionalities. It features five main components, which are detailed below.

## Components

### 3.1 Comprehensive Data Analysis

The first component of our web application, called "Data Overview", provides users with a comprehensive view of the original dataset, the topics present within the dataset, and the documents associated with each topic. This component serves as a foundational tool for users to engage with the dataset, enabling a thorough and detailed analysis of the data and the derived topics. It lays the groundwork for more detailed investigations and visualizations provided by the subsequent components of the application.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/1c89570f-8a2c-41ff-8ed9-d05f73a20868)

*Figure 1: Raw dataset visualization*

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/89ca9e72-f5fa-4d25-a4d8-bcdd63bb1ac5)

*Figure 2: Topic details visualization*

### 3.2 Interactive Topic Visualization

This component of the web application contains two tabs with interactive visualizations.

- The first tab, called "Data Map Plot," allows users to visualize document titles within topics and search for specific topics using keywords. This feature provides an intuitive way to explore the dataset by mapping out the documents according to their assigned topics, enabling users to quickly find and examine topics of interest.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/0f37cfb8-7410-4e97-bd0d-ec2f0da3cb98)

*Figure 3: Data map plot*

- The second tab, called "Intertopic Distance Map," offers a dynamic tool for exploring the relationships between topics. Users can use a slider to select a topic, which will then be highlighted in red. Hovering over a topic reveals general information about it, including the topic size and its corresponding label. This visualization helps users understand the relative distances and connections between different topics, offering insights into the broader structure and organization of the dataset.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/405ddf8f-ceef-48ed-b31b-155e27153445)

*Figure 4: Intertopic distance map*

### 3.3 Similar Topics Explorer

The "Similar Topics Explorer" component enables users to search for topics similar to a given query term and visualize the results using bar charts. By inputting a specific term, users can quickly identify and explore topics that are closely related to their query.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/8c9635f2-57e4-434d-b779-e420a7bf5a0a)

*Figure 5: Similar topics explorer*

### 3.4 Topic Probability Distribution

The "Topic Probability Distribution" component allows users to search for a document within the dataset using a dropdown menu and visualize the topic distributions within that document. Additionally, users can examine how each token contributes to a specific topic by visualizing the token-level distributions. This feature provides a detailed view of the internal topic structure of individual documents, highlighting the contribution of each word to the overall topic distribution. It enables users to gain a deeper understanding of how topics are represented at both the document and token levels.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/9128de44-dc2f-4539-abb9-71c5f914c99b)

*Figure 6: Document level distribution*

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/e0e4259f-3fc6-4cff-a7f9-bfd5a0d2f89e)

*Figure 7: Topic level distribution*

### 3.5 Document Assignment Explorer

The "Document Assignment Explorer" component enables users to visualize document titles within topics and select specific topics from a list on the right that contains all the topic labels. When users click on a particular document, a table appears below the plot, providing detailed information about the selected document. Users can interactively explore multiple documents by selecting and filtering them directly within the table, facilitating a comprehensive analysis of document assignments across various topics. This feature enhances the user's ability to navigate and understand the distribution of documents within topics, supporting detailed examinations and comparisons within the dataset.

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/3397d522-8bc4-410f-b90a-395302f28b7c)

*Figure 8: Document assignment explorer*

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/8e55c264-aed0-4af1-9dc8-d3bf335cbbf5)

*Figure 9: Topics selection*

![image](https://github.com/Alexandre-Lalle/Dash-topic-labeling-app/assets/128550546/499b903a-c8a7-4679-94c5-4d8c54822bc7)

*Figure 10: Documents visualization and filtering*

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/Alexandre-Lalle/Dash-topic-labeling-app.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Dash-topic-labeling-app
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    python app.py
    ```

## Conclusion

This application demonstrates the potential of LLM-based topic labeling in real-world applications. We encourage you to explore the features and functionalities provided in this prototype.


