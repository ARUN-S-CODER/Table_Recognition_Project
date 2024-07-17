# 📊 Table Extraction and Knowledge Graph Construction from Invoices

## 🚀 Problem Statement
 &emsp; &emsp;  &emsp; &emsp; The goal of this project is to **automatically extract table structures from invoice images** and represent the extracted data in a structured knowledge graph format. This involves detecting and interpreting table data from images using **OCR (Optical Character Recognition)** and constructing meaningful relationships between the extracted entities using graph theory.

## 📚 Table of Contents
- [Problem Statement](#-problem-statement)
- [Research Topics](#-research-topics)
- [References](#-references)
  - [GitHub Repositories](#github-repositories)
  - [Articles and Research Papers](#articles-and-research-papers)
- [Python Modules Used](#-python-modules-used)
- [Table Extraction Algorithm](#-table-extraction-algorithm)
- [Knowledge Graph Construction Algorithm](#-knowledge-graph-construction-algorithm)
- [Overview of the Code](#-overview-of-the-code)
- [Repository Structure](#-repository-structure)
- [Output and Results](#-output-and-results)
- [Conclusion](#-conclusion)

## 📖 Research Topics
 &emsp; &emsp; We explored the following topics to build our solution:<br>
 &emsp; &emsp; &emsp; &emsp; 1. **Graph Neural Network** <br>
 &emsp; &emsp; &emsp; &emsp; 2. **OCR Detection**<br>
 &emsp; &emsp; &emsp; &emsp; 3. **Table Transformer Pre-trained Model**<br>
 &emsp; &emsp; &emsp; &emsp; 4. **Paddle OCR**<br>
 &emsp; &emsp; &emsp; &emsp; 5. **Neo4j Data Model**<br>

## 🔗 References
### GitHub Repositories
1. [Extract Table Structure from Image Document](https://github.com/karndeepsingh/table_extract/blob/main/Extract_Table_Structure_from_Image_Document_.ipynb)
2. [GNN Table Extraction](https://github.com/AILab-UniFI/GNN-TableExtraction)
3. [Advanced Literate Machinery](https://github.com/AlibabaResearch/AdvancedLiterateMachinery)
4. [Table Transformer](https://github.com/microsoft/table-transformer)
5. [TIES-2.0](https://github.com/shahrukhqasim/TIES-2.0)
6. [GNN Table Recognition](https://github.com/sohaib023/gnn-table-recognition)
7. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md)

### Articles and Research Papers
1. [Information Extraction with Graph Convolutional Networks](https://www.google.com/amp/s/nanonets.com/blog/information-extraction-graph-convolutional-networks/amp/)
2. [What is Graph Neural Network](https://www.simplilearn.com/what-is-graph-neural-network-article)
3. [A Comprehensive Survey on Graph Neural Networks](https://link.springer.com/article/10.1007/s11042-021-11819-7)
4. [Dynamic Graph Convolutional Neural Networks](https://paperswithcode.com/method/dgcnn)
5. [Table Structure Detection and Data Extraction](https://arkaprava.medium.com/table-structure-detection-and-data-extraction-de18d9bb37bf)
6. [Building an Intelligent Invoice Processing Solution](https://blog.qburst.com/2022/10/building-an-intelligent-invoice-processing-solution-part-1/)

## 🐍 Python Modules Used
- **OpenCV**: Image preprocessing and manipulation.
- **NumPy**: Numerical operations.
- **PIL**: Image handling.
- **pytesseract**: OCR processing.
- **re**: Regular expressions for text cleaning.
- **spacy**: Natural language processing.
- **networkx**: Graph creation and manipulation.
- **matplotlib**: Graph visualization.
- **google.generativeai**: Gemini AI for entity and relationship extraction.

## 🧩 Table Extraction Algorithm
1. **Image Loading and Preprocessing**:
   - Convert the image to grayscale and apply median blur to reduce noise.
   - Perform edge detection using the Canny algorithm.
2. **Contour Detection**:
   - Find contours in the preprocessed image.
   - Filter contours to detect table cells based on their geometrical properties.
3. **Table Segmentation**:
   - Group detected cells into rows and columns to form a table structure.
4. **OCR Processing**:
   - Apply OCR on each cell to extract text.
   - Clean and standardize the extracted text.

## 🧠 Knowledge Graph Construction Algorithm
1. **Entity and Relationship Extraction**:
   - Use **Gemini AI** for few-shot learning to identify entities and relationships from the extracted text.
2. **Graph Creation**:
   - Create nodes for each entity and add labels.
   - Add edges between nodes to represent relationships.
3. **Graph Visualization**:
   - Use **NetworkX** to construct the graph.
   - Visualize the graph using **Matplotlib** with nodes and edges labeled appropriately.

## 🖥️ Overview of the Code
### `Table_Recognition_using_LLM_models.py`
- **Load and Preprocess Image**: Read the image and convert it to grayscale, then apply median blur to reduce noise.
- **Edge Detection and Contour Finding**: Detect edges using the Canny algorithm and find contours.
- **Table Segmentation**: Segment the detected contours into table structures.
- **OCR Extraction**: Use Tesseract to extract text from each table cell.
- **Text Cleaning**: Clean the extracted text using regular expressions.
- **Entity and Relationship Extraction**: Use Gemini AI to extract entities and relationships from the cleaned text.
- **Knowledge Graph Construction**: Create and visualize the knowledge graph using NetworkX and Matplotlib.

### `Table_Recognition_without_using_LLM.py`
- **Load and Preprocess Image**: Read the image and convert it to grayscale, then apply median blur to reduce noise.
- **Edge Detection and Contour Finding**: Detect edges using the Canny algorithm and find contours.
- **Table Segmentation**: Segment the detected contours into table structures.
- **OCR Extraction**: Use Tesseract to extract text from each table cell.
- **Text Cleaning**: Clean the extracted text using regular expressions.
- **Entity and Relationship Extraction**: Use SpaCy to extract entities and relationships from the cleaned text.
- **Knowledge Graph Construction**: Create and visualize the knowledge graph using NetworkX and Matplotlib.

## 📂 Repository Structure
      ├── Table_Recognition_without_using_LLM.py
      ├── Table_Recognition_using_LLM_model.py
      ├── OUTPUT
      |   ├── output1
      |   ├── ........
      |   ├── output8
      ├── Installation_Guide.md
      ├── Requirements.txt
      └── README.md

## 📈 Output and Results
The output of the project includes:
- Extracted table data from invoice images.
- Visual representation of the knowledge graph showing entities and their relationships.

The project demonstrates high accuracy in detecting table structures from images and successfully constructs meaningful knowledge graphs that can be used for further analysis and automation. 

### 🎬 OUTPUT VIDEO: https://github.com/user-attachments/assets/e29330ff-07c2-4ef7-bed1-f33e4ff64516

## 🏁 Conclusion
This project leverages advanced OCR techniques, graph theory, and machine learning models to extract and represent structured data from unstructured invoice images. The combination of these technologies provides a robust solution for automating data extraction and relationship mapping from complex document structures.
