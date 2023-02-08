import gradio as gr
import pandas as pd
import plotly.graph_objs as go
import torch
from torch import nn

# define classification model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_1 = nn.Conv1d(1, 40, 6)
        self.hidden_2 = nn.Conv1d(40, 20, 6)
        self.hidden_3 = nn.Conv1d(20, 8, 6)
        self.hidden_4 = nn.Conv1d(8, 4, 6)
        self.hidden_5 = nn.Conv1d(4, 2, 6)
        self.hidden_6 = nn.Linear(106, 50)
        self.output = nn.Linear(50, 4)

        self.pool = nn.MaxPool1d(6, stride=2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        
        x = self.hidden_1(x)
        x = self.pool(x)
        x = self.hidden_2(x)
        x = self.pool(x)
        x = self.hidden_3(x)
        x = self.pool(x)
        x = self.hidden_4(x)
        x = self.pool(x)
        x = self.hidden_5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.hidden_6(x))
        x = self.dropout(x) 
        x = self.output(x)

        return x

def visual_next(file):
    '''This function show visualization next example ECG'''
    global row
    row += 1
    df = pd.read_csv(file.name) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.iloc[row, :]))
    fig.update_layout(height=300)
    return fig

def visual_prev(file):
    '''This function show visualization previous example ECG'''
    global row
    row -= 1
    df = pd.read_csv(file.name) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.iloc[row, :]))
    fig.update_layout(height=300)
    return fig

def pred_class(file):
    '''This function classify current example ECG'''
    
    df = pd.read_csv(file.name) 
    features = torch.Tensor(list(df.iloc[row, :].values[1:]))

    # load the model
    model = CNN()
    state_dict = torch.load('model.pt', map_location='cpu')
    model.load_state_dict(state_dict)

    pred = torch.argmax(model(features.view(-1, 1, 2000))[0])

    info_dict = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}
    return info_dict[pred.item()]


# construct web-site
with gr.Blocks() as demo:
    title = gr.Markdown('# <p style="text-align: center;">ECG Analyzer</p>')

    with gr.Tab('Analyzer'):
        with gr.Row():
            file_box = gr.File(label="File")
            image_box = gr.Plot(label='Input ECG')
            row = -1
            file_box.change(visual_next, file_box, image_box)
        with gr.Row():
            output_box = gr.Textbox(label='Result')  
        with gr.Row():    
            prev_button = gr.Button('Previous')
            prev_button.click(visual_prev, file_box, image_box)
        with gr.Row():    
            next_button = gr.Button('Next')
            next_button.click(visual_next, file_box, image_box)
        with gr.Row():
            run_button = gr.Button('Analyze')
            run_button.click(pred_class, file_box, output_box)

demo.launch()
