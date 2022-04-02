import gradio as gr
import torch

import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

model.load_state_dict(torch.load('model_after_train.pt', map_location=torch.device('cpu')), strict=False)
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def preprocess_text(text):
    parts = []

    text_len = len(text.split(' '))
    delta = 300
    max_parts = 5
    nb_cuts = int(text_len / delta)
    nb_cuts = min(nb_cuts, max_parts)
    
    
    for i in range(nb_cuts + 1):
        text_part = ' '.join(text.split(' ')[i * delta: (i + 1) * delta])
        parts.append(tokenizer.encode(text_part, return_tensors="pt", max_length=500).to(device))

    return parts

def test(text):
    text_parts = preprocess_text(text)
    overall_output = torch.zeros((1,2)).to(device)
    try:
        for part in text_parts:
            if len(part) > 0:
                overall_output += model(part.reshape(1, -1))[0]
    except RuntimeError:
        print("GPU out of memory, skipping this entry.")

    overall_output = F.softmax(overall_output[0], dim=-1)

    value, result = overall_output.max(0)

    term = "fake"
    if result.item() == 0:
        term = "real"

    return term + " at " + str(int(value.item()*100)) + " %"


description = "Fake news detector trained using pre-trained model bert-base-uncased, fine-tuned on https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset dataset"
title = "Fake News Detector"

examples = ["BRUSSELS (Reuters) - Germany is urging the European Union to add up to four more Russian nationals and companies to the bloc’s sanctions blacklist over Siemens (SIEGn.DE) gas turbines delivered to Moscow-annexed Crimea, two sources in Brussels said. The EU has barred its firms from doing business with Crimea since the 2014 annexation, imposed sanctions on Russian individuals and entities, and curbed cooperation with Russia in energy, arms and finance over its role in the crisis in Ukraine. After it annexed Crimea from Kiev, Moscow threw its support behind a separatist rebellion in eastern Ukraine, which has killed more than 10,000 people and is still simmering. The EU’s blacklist comprises 150 people and 37 entities subject to an asset freeze and a travel ban. The restrictions are in place until Sept. 15. “The regular review would normally be the moment to look at who is on the list. In the past, when there were good grounds, we’ve added entries to the list,” an EU official said. Siemens, trying to distance itself from the scandal, last week said it was halting deliveries of power equipment to Russian state-controlled customers and reviewing supply deals. Russia’s Energy Minister Alexander Novak played down the potential consequences of a halt. “What Siemens supplies can be delivered by other companies,” Novak told reporters in St Petersburg. “As for electricity generation, we ... have now learnt to produce the necessary equipment,” he said, without referring to the prospect of additional sanctions. Siemens says it has evidence that all four turbines it delivered for a project in southern Russia had been illegally moved to Crimea. German government spokeswoman Ulrike Demmer said on Monday the turbines were delivered to Crimea against the terms of the contract and despite high-ranking assurances from Russian officials that this would not happen. Berlin was consulting on what consequences this “unacceptable” operation might have, she said, adding, however, that the onus was on companies to ensure they did not violate the sanctions regime. The proposed additions to the blacklist could include Russian Energy Ministry officials and the Russian company that moved the turbines to the Black Sea peninsula, one senior diplomatic source in Brussels said. Another source said representatives of all 28 EU member states could discuss the matter for the first time in Brussels as soon as Wednesday. The EU needs unanimity to impose or extend any sanctions. Hungary, Bulgaria, Italy and Cyprus are among EU states which are usually skeptical of Russia sanctions. They take the line that punitive measures have failed to force a change of course by Moscow while hurting European business. Reuters first reported a year ago on the Siemens case, which has exposed the difficulties of imposing EU sanctions."]

iface = gr.Interface(fn=test, inputs="text", outputs="text", title=title,description=description, examples=examples)
iface.launch()
