import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def add_table_to_graph(G, table_id, entities):
    for entity, label in entities:
        G.add_node(entity, label=label, table_id=table_id)

def extract_relationships(doc):
    relationships = []
    for token in doc:
        if token.dep_ in ("nsubj", "dobj"):
            subject = token.head.text
            object_ = token.text
            verb = token.head.head.text if token.dep_ == "dobj" else token.head.text
            relationships.append((subject, verb, object_))
    return relationships

def process_image(image):
    
    img_array = np.array(image)
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        img=img_array.astype(np.uint8)
    elif img_array.shape[2] == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    elif img_array.shape[2] == 4:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unexpected number of channels in the input image.")
        
    threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, vertical_kernel)

    table_lines = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        table = img[y:y+h, x:x+w]
        tables.append(table)

    extracted_tables = []
    for table_img in tables:
        table_text = pytesseract.image_to_string(table_img)
        extracted_tables.append(table_text)

    return extracted_tables

def create_knowledge_graph(extracted_tables):
    G = nx.DiGraph()

    all_table_entities = []
    for idx, table_text in enumerate(extracted_tables):
        clean_text = preprocess_text(table_text)
        entities = extract_entities(clean_text)
        all_table_entities.append(entities)
        add_table_to_graph(G, idx, entities)

    for table_text in extracted_tables:
        doc = nlp(table_text)
        relationships = extract_relationships(doc)
        for subject, verb, object_ in relationships:
            G.add_edge(subject, object_, relation=verb)

    return G, all_table_entities

def plot_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color="lightblue", font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

st.title("Knowledge Graph from Invoice")

uploaded_file = st.file_uploader("Upload a PNG file", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Processing...'):
        extracted_tables = process_image(image)
        G, all_table_entities = create_knowledge_graph(extracted_tables)
    
    st.success('Processing Complete!')
    
    st.subheader('Extracted Entities and Relationships')
    for idx, entities in enumerate(all_table_entities):
        st.write(f"Table {idx + 1}")
        for entity, label in entities:
            st.write(f"{entity} ({label})")
    
    st.subheader('Knowledge Graph')
    plot_graph(G)
    st.pyplot(plt)
