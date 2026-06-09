# code from Mahé DUVAL
import sys
import os
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Nifty.method import *
import Nifty.method as _nifty_method
from Nifty.networks import *
import warnings; warnings.filterwarnings('ignore')
import gradio as gr


seed = 0
torch.manual_seed(seed)

# Check for existing model at startup to handle UI button state
MODEL_PATH = "./training/UNet_peppers.pth"
has_initial_model = os.path.exists(MODEL_PATH)
device_used = None
# Training settings
training_path = ''
training_img = None

def train_unet(img_path, save_path, progress=gr.Progress(track_tqdm=True)):
	if img_path is None:
		raise gr.Error("Please upload a training image first.")
	if not save_path:
		raise gr.Error("Please provide a valid save path.")
	
	# Load new training image and update mu/sigma
	img = Tensor_load(img_path)
	mu, sigma = img.mean(), img.std()
	
	# Train the flow model
	flow_model = UNet(
			dim=64,
			dim_mults=(1, 2)).to(device_used)
 
	progress(0, desc="Starting training loop")
	train_flow_net((img-mu)/sigma, flow_model, load=False, epochs=10000, show=False, save_name=save_path, device=device_used)
	progress(1.0, desc="Training complete")

	# Re-enable the Generate button once trained
	return gr.update(interactive=True)

def load_unet(file_obj):
	global flow_model
	if file_obj is None:
		return gr.update()
		
	flow_model = UNet(
			dim=64,
			dim_mults=(1, 2)).to(device_used)
	
	flow_model.load_state_dict(torch.load(file_obj.name, map_location=device_used))
	flow_model.eval().to(device_used)
	
	# Re-enable the Generate button once loaded
	return gr.update(interactive=True)

# Parameters
def fresh_noise():
	return torch.randn(1, 3, 256, 256).to(device_used)

# NN flow
def flow_nn(image_path, T):
    if "cuda" in str(_nifty_method.device):
        torch.cuda.empty_cache()
        print("cleared cache")
    start = time.time()
    torch.manual_seed(seed)
    img = Tensor_load(image_path).clone().to(device_used)
    mu, sigma = img.mean(), img.std()
    if flow_model is None:
        raise gr.Error("No Neural Network loaded")
        
    noise = fresh_noise()
    with torch.no_grad():
        x = noise*1.
        times=torch.linspace(0, 1, steps=T+1).to(device_used)
        for it in range(T):
            t=times[it]
            t = t.to(device_used).unsqueeze(0)
            flow = flow_model(x,t.view(1))
            x=x+flow*(times[it+1]-times[it])

            synth_nn_x= x*sigma+mu
            yield get_synthesized_image(synth_nn_x[...,64:64+128,64:64+128])

    synth_nn = x*sigma+mu
    synth_nn=synth_nn[...,64:64+128,64:64+128]
    print(time.time()-start)
    yield get_synthesized_image(synth_nn)

def flow_nifty(input_img_path, T, k, rs=1, octaves=1, renoise=0.5):
	if "cuda" in str(_nifty_method.device):
		torch.cuda.empty_cache()
		print("cleared cache")
	start = time.time()
	if input_img_path is None:
		raise gr.Error("No input image loaded")
	torch.manual_seed(seed)
	im1 = Tensor_load(input_img_path).clone().to(device_used)
	
	noise = fresh_noise()
	img= get_synthesized_image(Nifty(im1,rs=rs,T=T,k=k,patchsize=16,stride=4,octaves=octaves,size=(128,128),renoise=renoise,warmup=0,memory=False,noise=noise,show=False,spotsize=1/4,seed=seed,blend=0.,blend_alpha=0.5,blend_map=None))
	print(time.time()-start)
	return img

def get_synthesized_image(synth):
	return (synth[0].permute(1, 2, 0).cpu().detach().numpy() * .5 + .5).clip(0, 1).astype(np.float32)

def restore_nn_controls(input_img):
	return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=input_img is not None)

def restore_nifty_controls():
	return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=flow_model is not None)

def update_pytorch_seed(input_seed):
	global seed
	seed = input_seed

def validate_training():
	global training_img, training_path
	if Path(training_path).suffix == '.pth' and training_img is not None :
		return gr.update(interactive=True) 
	return gr.update(interactive=False)    

def update_training_path(x):
	global training_path
	training_path = x 

def update_training_img(img):
	global training_img
	training_img = img
	
def update_processing_unit_selection(choice:str):
	global device_used
	_nifty_method.device = manually_select_device(try_gpu=(choice == "GPU"))
	device_used = _nifty_method.device
 
selected_processing_unit_type = "GPU" if "cuda" in str(_nifty_method.device) else "CPU"
device_used = _nifty_method.device

# Initialize conditionally so it doesn't freeze the UI waiting to train if missing
if has_initial_model:
	flow_model = UNet(dim=64, dim_mults=(1, 2)).to(device_used)
	flow_model.load_state_dict(torch.load(MODEL_PATH, map_location=device_used))
	flow_model.eval().to(device_used)
else:
	flow_model = None
 
# Interface
from Demos.Utilities.theme import *
def demo_unet():
    
	with gr.Blocks(title="Nifty") as nifty_demo:
		with gr.Row(equal_height=True):
			in_processing_unit_choice = gr.Radio(
				label="Mode",
				choices=["GPU", "CPU"],
				value=selected_processing_unit_type,
				type="value",
				elem_classes="radio_group",
				elem_id="processing_unit_radio_group"
			)
		# Inputs and outputs
		with gr.Row(equal_height=True):
			# Input column
			with gr.Column(scale=1, elem_id="input_column"):
				in_img1 = gr.Image(
					label="Input Image",
					value="./results/red_peppers.jpg",
					type="filepath",
					elem_classes="full_size_image",
					elem_id="input_image",
					width=256,
					height=256 
				) 
			with gr.Column(scale=3, elem_id="input_general_settings"):
				in_rs = gr.Slider(
					minimum=0.,
					maximum=1.,
					value=1., 
					step=0.01,
					label="Ratio Sample",
					info="Ratio of the reference patches to sample at each step"
					)
				in_T = gr.Slider(
					minimum=0.,
					maximum=100.,
					value=50.,
					step=0.,
					label="Discretization steps",
					info="Number of (linear) distretization steps between 0 and 1 to solve the flow ODE"
					)
				in_k = gr.Slider(
					minimum=0.,
					maximum=50.,
					value=5.,
					step=0.,
					label="K Neighbors",
					info="Number of top closest patch used to approximate the velocity field"
					)
				in_octaves = gr.Slider(
					minimum=0.,
					maximum=20.,
					value=4.,
					step=1.,
					label="Octaves",
					info="Number of diadic scales used for the synthesis"
					)
				in_renoise = gr.Slider(
					minimum=0.,
					maximum=1.,
					value=0.3,
					step=0.01,
					label="Renoise",
					info="Time used renoise the smooth upsampled image at each resolution"
					)
				in_seed = gr.Slider(
					minimum=0.,
					maximum=99999.,
					value=0.,
					step=1.,
					label="Seed",
					info="PyTorch manual seed"
					)
				
				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						generate_btn_nn = gr.Button("Generate NN", variant="primary", interactive=has_initial_model)
						cancel_btn_nn = gr.Button("Cancel NN", variant="stop", visible=False)
					with gr.Column(scale=1):
						generate_btn_nifty = gr.Button("Generate Nifty", variant="primary")
						cancel_btn_nifty = gr.Button("Cancel Nifty", variant="stop", visible=False)
			with gr.Column(scale=1, elem_classes="full_height filled_flex_display shrink"): gr.HTML(HTML_V_SEPARATOR, elem_classes="full_height")			
			with gr.Column(scale=1):
				with gr.Tabs(elem_classes="full_height"):
					with gr.Tab("Train Model", elem_id="train_tab", elem_classes="full_height"):
						with gr.Row(equal_height=True):
							in_img_training = gr.Image(
							label="Input Image",
							value="./results/red_peppers.jpg",
							type="filepath",
							elem_classes="full_width",
							elem_id="input_image_training",
							width=270,
							height=270 
						) 
						with gr.Row(equal_height=True):
							in_save_name = gr.Textbox(
								label="Save Path & Filename",
								value="",
								placeholder="./training/MODEL_NAME.pth"
							)
						with gr.Row(equal_height=True):
								training_status = gr.Textbox(placeholder="Training status will appear here...",label="Training Status")
						with gr.Row(equal_height=True):
								train_btn_nn = gr.Button("Train NN", variant="primary", interactive=False)
								cancel_btn_train = gr.Button("Cancel Training", variant="stop", visible=False)
					with gr.Tab("Load Model", elem_id="load_tab", elem_classes="full_height"):
						in_path_model = gr.File(elem_classes="full_size_image", file_count="single", file_types=[".pth"], label="Model file", value=MODEL_PATH if has_initial_model else None)
		
		gr.HTML(HTML_SEPARATOR)
		
		with gr.Row(equal_height=True,elem_id="output_row"):  
			with gr.Column(scale=1, elem_classes="output_column"):
				out_img_nn = gr.Image(
					label="NN Output",
					type="numpy",
					elem_classes="full_size_image"
					)
			with gr.Column(scale=1, elem_classes="output_column"):
				out_img_nifty = gr.Image(
					label="Nifty Output",
					type="numpy",
					elem_classes="full_size_image"
					)

		# Event Binding
		in_seed.change(
			fn = update_pytorch_seed,
			inputs=[in_seed],
			outputs=[]
		)
		
		# Disable/Enable Generate Nifty button if the input image is removed/added
		in_img1.change(
			fn=lambda img: gr.update(interactive=img is not None),
			inputs=[in_img1],
			outputs=[generate_btn_nifty]
		)

		# Training conditions
		in_save_name.change(
			fn= update_training_path,
			inputs=[in_save_name],
		).then(
			fn=validate_training,
			outputs=[train_btn_nn]
		)
		in_img_training.change(
			fn = update_training_img,
			inputs=[in_img_training]
		).then(
			fn=validate_training,
			outputs=[train_btn_nn]
		)
		# Processing unit
		in_processing_unit_choice.input(
			fn = update_processing_unit_selection,
			inputs=in_processing_unit_choice,
			outputs=[]
		)
		# Train model : Lock UI + Run Training... After : Unlock UI & Show Status
		start_train = train_btn_nn.click(
			fn=lambda: (
				gr.update(interactive=False, value="Training...", visible=False),
				gr.update(visible=True)
				),
			outputs=[train_btn_nn, cancel_btn_train],
			queue=False
		).then(
			fn=train_unet,
			inputs=[in_img_training, in_save_name],
			outputs=[generate_btn_nn,training_status]
		)
		start_train.then(
			fn=lambda: (
					gr.update(interactive=True, value="Train NN", visible=True),
					gr.update(visible=False)),
			outputs=[train_btn_nn, cancel_btn_train],
			queue=False
		)
		cancel_btn_train.click(
			fn=lambda: (
					gr.update(interactive=True, value="Train NN", visible=True),
					gr.update(visible=False)),
			outputs=[train_btn_nn, cancel_btn_train],
			cancels=[start_train],
			queue=False
		)
	
		# Load model from UI, update UI button state and display status
		in_path_model.change(
			fn=load_unet,
			inputs=[in_path_model],
			outputs=[generate_btn_nn]
		)
	

		# NN Generation sequence (disable Nifty to avoid OOM)
		start_nn = generate_btn_nn.click(
			fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)),
			outputs=[generate_btn_nn, cancel_btn_nn, generate_btn_nifty],
			queue=False
		).then(
			fn=flow_nn,
			inputs=[in_img1,in_T],
			outputs=[out_img_nn]
		)
		start_nn.then(
			fn=restore_nn_controls,
			inputs=[in_img1],
			outputs=[generate_btn_nn, cancel_btn_nn, generate_btn_nifty],
			queue=False
		)
		cancel_btn_nn.click(
			fn=restore_nn_controls,
			inputs=[in_img1],
			outputs=[generate_btn_nn, cancel_btn_nn, generate_btn_nifty], 
			cancels=[start_nn],
			queue=False
		)

		# Nifty Generation sequence (disable NN to avoid OOM)
		start_nifty = generate_btn_nifty.click(
			fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)),
			outputs=[generate_btn_nifty, cancel_btn_nifty, generate_btn_nn],
			queue=False
		).then(
			fn=flow_nifty,
			inputs=[in_img1, in_T, in_k, in_rs, in_octaves, in_renoise],
			outputs=[out_img_nifty]
		)
		start_nifty.then(
			fn=restore_nifty_controls,
			outputs=[generate_btn_nifty, cancel_btn_nifty, generate_btn_nn],
			queue=False
		)
		cancel_btn_nifty.click(
			fn=restore_nifty_controls, 
			outputs=[generate_btn_nifty, cancel_btn_nifty, generate_btn_nn], 
			cancels=[start_nifty],
			queue=False
		)
	with gr.Accordion(label="Documentation", open=True):
		with open('Demos/sub_demos/doc/doc_unet.md','r') as f : 
			gr.Markdown(f.read(),latex_delimiters=[{ "left": "$$", "right": "$$", "display": True },{"left": "$", "right": "$", "display": False},])
if __name__ == "__main__":
    with gr.Blocks() as demo:
        demo_unet()
    demo.queue().launch()