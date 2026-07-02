# code from Mahé DUVAL
# Adapted version of demo.py for CPU only servers

import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from Demos.Utilities.theme import *

import Demos.sub_demos.demo_nifty as dnfty
from Demos.sub_demos.demo_nifty import demo_nifty

dnfty.runs_on_server = True

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    demo_nifty()
    gr.HTML(HTML_FOOTER)

demo.queue().launch(css=CUSTOM_CSS,head=HTML_CUSTOM_HEAD, theme=theme)