import gradio as gr
import pandas as pd
import plotly.graph_objs as go
from catboost import CatBoostClassifier
from tsfresh import extract_features

def visual_next(file):
    global row
    row += 1
    df = pd.read_csv(file.name) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.iloc[row, :]))
    fig.update_layout(height=300)
    return fig

def visual_prev(file):
    global row
    row -= 1
    df = pd.read_csv(file.name) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.iloc[row, :]))
    fig.update_layout(height=300)
    return fig

def pred_class(file):
    
    global row
    df = pd.read_csv(file.name) 
    features = df.iloc[row, :].values[1:]
    feauters_dict = {'0': {'minimum': None,
    'median': None,
    'root_mean_square': None,
    'absolute_maximum': None,
    'standard_deviation': None,
    'variance': None,
    'sum_values': None,
    'mean': None}}
    data_long = pd.DataFrame()
    data_long['0'] = features
    data_long[1] = 2
    extracted_features = extract_features(data_long, column_id=1, kind_to_fc_parameters=feauters_dict)
    
    model = CatBoostClassifier()
    model.load_model('model.onnx')

    info_dict = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}
    return info_dict[model.predict(extracted_features)[0][0]]



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
