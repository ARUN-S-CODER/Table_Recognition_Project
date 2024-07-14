import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image
import pytesseract
import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

nlp = spacy.load("en_core_web_sm")

''' kindly create your own api_key and use in the below configure function.
            You can Use the link in the readme file to create our own api key'''

genai.configure(api_key="PASS YOUR OWN API KEY") 

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

def extract_relationships_gemini(text):
    examples = [
    {
        "input": "Invoice ID: 62727 Date: 08/06/2006 Name: John Doe Address: 123 Elm St, Springfield, ZIP: 12345 Item: Widget A Price: $20.00",
        "output": {
            "entities": [
                {"text": "62727", "type": "INVOICE_ID"},
                {"text": "08/06/2006", "type": "DATE"},
                {"text": "John Doe", "type": "NAME"},
                {"text": "123 Elm St, Springfield", "type": "ADDRESS"},
                {"text": "12345", "type": "ZIP_CODE"},
                {"text": "Widget A", "type": "ITEM"},
                {"text": "$20.00", "type": "MONEY"}
            ],
            "relationships": [
                {"source": "62727", "type": "issued_on", "target": "08/06/2006"},
                {"source": "John Doe", "type": "located_at", "target": "123 Elm St, Springfield"},
                {"source": "John Doe", "type": "located_at", "target": "12345"},
                {"source": "Widget A", "type": "costs", "target": "$20.00"}
            ]
        }
    },
    {
        "input": "Invoice ID: M3558273 Date: 06-Nov-23 Customer ID: USHAD010 Name: Tammy Boland Address: 7915 N. Hale Ave, Suite E, Peoria, ZIP: 61615 Item: Shipment S2303219070 Price: $5,521.09",
        "output": {
            "entities": [
                {"text": "M3558273", "type": "INVOICE_ID"},
                {"text": "06-Nov-23", "type": "DATE"},
                {"text": "USHAD010", "type": "CUSTOMER_ID"},
                {"text": "Tammy Boland", "type": "NAME"},
                {"text": "7915 N. Hale Ave, Suite E, Peoria", "type": "ADDRESS"},
                {"text": "61615", "type": "ZIP_CODE"},
                {"text": "Shipment S2303219070", "type": "ITEM"},
                {"text": "$5,521.09", "type": "MONEY"}
            ],
            "relationships": [
                {"source": "M3558273", "type": "issued_on", "target": "06-Nov-23"},
                {"source": "USHAD010", "type": "belongs_to", "target": "Tammy Boland"},
                {"source": "Tammy Boland", "type": "located_at", "target": "7915 N. Hale Ave, Suite E, Peoria"},
                {"source": "Tammy Boland", "type": "located_at", "target": "61615"},
                {"source": "Shipment S2303219070", "type": "costs", "target": "$5,521.09"}
            ]
        }
    },
    {
        "input": "Invoice ID: A202076381 Date: 22-09-23 Name: Officine Meccaniche Lafranconi S.P.A. Address: Milano, Italy ZIP: 20100 Item: Ocean Bill of Lading Price: $289.82",
        "output": {
            "entities": [
                {"text": "A202076381", "type": "INVOICE_ID"},
                {"text": "22-09-23", "type": "DATE"},
                {"text": "Officine Meccaniche Lafranconi S.P.A.", "type": "NAME"},
                {"text": "Milano, Italy", "type": "ADDRESS"},
                {"text": "20100", "type": "ZIP_CODE"},
                {"text": "Ocean Bill of Lading", "type": "ITEM"},
                {"text": "$289.82", "type": "MONEY"}
            ],
            "relationships": [
                {"source": "A202076381", "type": "issued_on", "target": "22-09-23"},
                {"source": "Officine Meccaniche Lafranconi S.P.A.", "type": "located_at", "target": "Milano, Italy"},
                {"source": "Officine Meccaniche Lafranconi S.P.A.", "type": "located_at", "target": "20100"},
                {"source": "Ocean Bill of Lading", "type": "costs", "target": "$289.82"}
            ]
        }
    },
    {
        "input": "Invoice ID: 101928 Date: 14-07-24 Name: Alice Johnson Address: 456 Oak St, Metropolis, ZIP: 67890 Item: Service Charge Price: $150.00",
        "output": {
            "entities": [
                {"text": "101928", "type": "INVOICE_ID"},
                {"text": "14-07-24", "type": "DATE"},
                {"text": "Alice Johnson", "type": "NAME"},
                {"text": "456 Oak St, Metropolis", "type": "ADDRESS"},
                {"text": "67890", "type": "ZIP_CODE"},
                {"text": "Service Charge", "type": "ITEM"},
                {"text": "$150.00", "type": "MONEY"}
            ],
            "relationships": [
                {"source": "101928", "type": "issued_on", "target": "14-07-24"},
                {"source": "Alice Johnson", "type": "located_at", "target": "456 Oak St, Metropolis"},
                {"source": "Alice Johnson", "type": "located_at", "target": "67890"},
                {"source": "Service Charge", "type": "costs", "target": "$150.00"}
            ]
        }
    },
    {
        "input": "Invoice ID: C2301308301 Date: 18-08-23 Name: DHL Global Forwarding Address: 7915 N. Hale Ave, Suite E, Peoria, ZIP: 61615 Item: Foreign Customs Duties Price: $261.00",
        "output": {
            "entities": [
                {"text": "C2301308301", "type": "INVOICE_ID"},
                {"text": "18-08-23", "type": "DATE"},
                {"text": "DHL Global Forwarding", "type": "NAME"},
                {"text": "7915 N. Hale Ave, Suite E, Peoria", "type": "ADDRESS"},
                {"text": "61615", "type": "ZIP_CODE"},
                {"text": "Foreign Customs Duties", "type": "ITEM"},
                {"text": "$261.00", "type": "MONEY"}
            ],
            "relationships": [
                {"source": "C2301308301", "type": "issued_on", "target": "18-08-23"},
                {"source": "DHL Global Forwarding", "type": "located_at", "target": "7915 N. Hale Ave, Suite E, Peoria"},
                {"source": "DHL Global Forwarding", "type": "located_at", "target": "61615"},
                {"source": "Foreign Customs Duties", "type": "costs", "target": "$261.00"}
            ]
        }
    }
]



    model = genai.GenerativeModel('gemini-1.0-pro-latest')
    prompt = f"Extract entities and relationships from the following text:\n{text}\n\nProvide the output in the format: entities and relationships.\n\nExamples:\n\n{examples[0]['input']}\n\nOutput:\nEntities:\n{examples[0]['output']['entities']}\n\nRelationships:\n{examples[0]['output']['relationships']}\n\n"
    response = model.generate_content(prompt)

    navigat_var = response.candidates[0].content.parts[0].text

    parsedData = parse_data(navigat_var)
    entities = parsedData['entities']
    relationships = parsedData['relationships']
 
    return entities, relationships

def parse_data(data_str):
    entities_str, relationships_str = data_str.strip().split("\n\nRelationships:\n")
    
    entities_str = entities_str.replace("Entities:\n", "")
    entities = json.loads(entities_str.replace("'", '"'))

    relationships = json.loads(relationships_str.replace("'", '"'))

    return {"entities": entities, "relationships": relationships}



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
    threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 5)

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
        entities, relationships = extract_relationships_gemini(table_text)
        print(entities, "\n", relationships, "\n\n")
        for i in entities:
            entity, label = i['text'], i['type']
            print(entity, label)
            G.add_node(entity, label=label)
        for i in relationships:
            source, relation, target = i['source'], i['type'], i['target']
            print(source, relation, target)
            G.add_edge(source, target, relation=relation)

    return G, all_table_entities

def plot_graph(G):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=3)
    nx.draw(G, pos, with_labels=True, node_size=7000, node_color="lightblue", font_size=14, font_weight="bold")
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
    
            
    
    st.subheader('Knowledge Graph')
    plot_graph(G)
    st.pyplot(plt)
