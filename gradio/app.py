import gradio as gr
import xml.etree.ElementTree as ET
import re
import urllib
import torch


from transformers import pipeline

classifier = pipeline(model="Yozhikoff/arxiv-topics-distilbert-base-cased")


import re
import urllib.request
import xml.etree.ElementTree as ET

def get_arxiv_title_and_abstract(link):
    try:
        # Validate the arxiv link
        pattern = r'^https?://arxiv.org/(abs|pdf)/(\d{4}\.\d{4,5})(\.pdf)?$'
        match = re.match(pattern, link)
        if not match:
            raise ValueError('Invalid arxiv link')
        
        # Construct the arxiv API URL
        arxiv_id = match.group(2)
        api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
        
        # Send a request to the arxiv API
        response = urllib.request.urlopen(api_url)
        xml_data = response.read()
        
        # Parse the XML data
        root = ET.fromstring(xml_data)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        
        return title, summary
    except:
        raise gr.Error('Invalid arXiv URL!')


def classify_paper(title, abstract):
    if title == '' and abstract == '':
        raise gr.Error('Fill Title or/and Abstract')
    
    text = f"TITLE\n{title}\n\nABSTRACT\n{abstract}"
    item = classifier.tokenizer(text)
    input_tensor = torch.tensor(item['input_ids'])[None]
    logits = classifier.model(input_tensor).logits[0]
    preds = torch.sigmoid(logits).detach().cpu().numpy()
    result = {classifier.model.config.id2label[num]: float(prob) for num, prob in enumerate(preds) if prob > 0.25}
    return result
    

with gr.Blocks(title='Paper classifier') as demo:
    gr.Markdown('# Paper Topic Classifier')
    with gr.Row():
        with gr.Column():
            gr.Markdown('## Inputs')
            gr.Markdown('#### Please enter an arXiv link **OR** fill title and abstract manually')
            arxiv_link = gr.Textbox(label="Arxiv link", placeholder="https://arxiv.org/abs/1706.03762")

            b1 = gr.Button("Parse Link")

            title = gr.Textbox(label="Paper title", placeholder="Title text")
            abstract = gr.Textbox(label="Paper abstract", placeholder="Abstract text")

            b2 = gr.Button("Classify Paper", variant='primary')
            
            b1.click(fn=get_arxiv_title_and_abstract, inputs=arxiv_link, outputs=[title, abstract], api_name="parse")


        with gr.Column():
            gr.Markdown('## Topics')
            gr.Markdown('## ')
            gr.Markdown('## ')
            out = gr.Label(label="Topics")
            b2.click(classify_paper, inputs=[title, abstract], outputs=out)
    
    gr.Markdown('## Examples')
    gr.Examples(
        examples=[['https://arxiv.org/abs/1706.03762'], ['https://arxiv.org/abs/2304.06718'], ['https://arxiv.org/abs/1307.0058']],
        inputs=arxiv_link,
        outputs=[title, abstract],
        fn=get_arxiv_title_and_abstract,
        cache_examples=True,
    )

demo.launch()