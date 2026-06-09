# code from Mahé DUVAL
import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from Demos.Utilities.theme import *
from Demos.sub_demos.demo_nifty import demo_nifty
from Demos.sub_demos.demo_unet import demo_unet

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    with gr.Tabs(selected="a"):
        with gr.Tab("Nifty", id="a"):
            demo_nifty()
        with gr.Tab("Nifty - Unet", id="b"):
            demo_unet()
    gr.HTML(HTML_FOOTER)

demo.queue().launch(css=CUSTOM_CSS,head=HTML_CUSTOM_HEAD, theme=theme)