import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid NSWindow on background threads
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Global variables for model
tokenizer = None
model = None

def load_llama_model():
    """Load Llama model if not already loaded"""
    global tokenizer, model
    model_name = "meta-llama/Llama-3.2-1B"
    
    if tokenizer is None or model is None:
        print("Loading Llama 3.2 1B model...")
        try:
            # Select device: CUDA > CPU
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            model = model.to(device)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

def simulate_1d_heat_equation(n_spatial, n_timesteps, seed, diffusion_coeff, bc_type='Periodic'):
    """Simulate 1D heat equation with random initial condition"""
    np.random.seed(seed)
    
    # Random initial condition
    initial_temp = np.random.rand(n_spatial)
    
    # Storage for all timesteps
    temperature_grid = np.zeros((n_timesteps, n_spatial))
    temperature_grid[0, :] = initial_temp
    
    # Current temperature
    current_temp = initial_temp.copy()
    
    # Parameters
    dx = 1.0
    dt = 0.05
    alpha = diffusion_coeff
    
    # Time evolution
    for t in range(1, n_timesteps):
        new_temp = current_temp.copy()
        for i in range(n_spatial):
            center = current_temp[i]
            if bc_type == 'Periodic':
                left = current_temp[(i - 1) % n_spatial]
                right = current_temp[(i + 1) % n_spatial]
            elif bc_type == 'Neumann (no flux)':
                if i == 0:
                    left = current_temp[i]        # mirror at left boundary → du/dx = 0
                    right = current_temp[i + 1]
                elif i == n_spatial - 1:
                    left = current_temp[i - 1]
                    right = current_temp[i]       # mirror at right boundary → du/dx = 0
                else:
                    left = current_temp[i - 1]
                    right = current_temp[i + 1]
            else:
                # default to periodic if unknown
                left = current_temp[(i - 1) % n_spatial]
                right = current_temp[(i + 1) % n_spatial]
            d2_dx2 = (left - 2*center + right) / (dx*dx)
            new_temp[i] = center + alpha * dt * d2_dx2
        current_temp = new_temp
        temperature_grid[t, :] = current_temp
    
    # Scale to [100, 800]
    min_val = temperature_grid.min()
    max_val = temperature_grid.max()
    
    if max_val > min_val:
        scaled_grid = ((temperature_grid - min_val) / (max_val - min_val)) * 700 + 100
    else:
        scaled_grid = np.full_like(temperature_grid, 400)
    
    return scaled_grid.astype(int)

def continue_simulation_fixed(original_data, n_additional_steps, diffusion_coeff, bc_type='Periodic'):
    """Continue heat equation simulation with proper scaling continuity"""
    n_timesteps, n_spatial = original_data.shape
    
    # Get the last timestep as starting point (back to float for physics)
    # Need to reverse the scaling to get back to physical values
    last_scaled = original_data[-1, :].astype(float)
    
    # Reverse the original scaling to get physical temperature values
    # Assuming the scaling was: scaled = ((temp - min_val) / (max_val - min_val)) * 700 + 100
    # We need to estimate the original range from all the data
    all_scaled_data = original_data.flatten()
    scaled_min = all_scaled_data.min()  # Should be ~100
    scaled_max = all_scaled_data.max()  # Should be ~800
    
    # Convert last timestep back to [0,1] range for physics continuation
    current_temp = (last_scaled - scaled_min) / (scaled_max - scaled_min)
    
    # Storage for additional timesteps in physical units
    additional_data = np.zeros((n_additional_steps, n_spatial))
    
    # Parameters (same as original simulation)
    dx = 1.0
    dt = 0.05
    alpha = diffusion_coeff
    
    # Continue simulation in physical units
    for t in range(n_additional_steps):
        new_temp = current_temp.copy()
        for i in range(n_spatial):
            center = current_temp[i]
            if bc_type == 'Periodic':
                left = current_temp[(i - 1) % n_spatial]
                right = current_temp[(i + 1) % n_spatial]
            elif bc_type == 'Neumann (no flux)':
                if i == 0:
                    left = current_temp[i]        # mirror at left boundary → du/dx = 0
                    right = current_temp[i + 1]
                elif i == n_spatial - 1:
                    left = current_temp[i - 1]
                    right = current_temp[i]       # mirror at right boundary → du/dx = 0
                else:
                    left = current_temp[i - 1]
                    right = current_temp[i + 1]
            else:
                # default to periodic if unknown
                left = current_temp[(i - 1) % n_spatial]
                right = current_temp[(i + 1) % n_spatial]
            d2_dx2 = (left - 2*center + right) / (dx*dx)
            new_temp[i] = center + alpha * dt * d2_dx2
        current_temp = new_temp
        additional_data[t, :] = current_temp
    
    # Scale the continued part to match the original range [100, 800]
    # Use the same scaling as the original data
    scaled_additional = additional_data * (scaled_max - scaled_min) + scaled_min
    
    return scaled_additional.astype(int)

def predict_with_llama(input_string, n_spatial, n_predict_steps=4):
    """Use Llama to predict next timesteps"""
    if not load_llama_model():
        return np.random.randint(100, 800, size=(n_predict_steps, n_spatial)), "[Model not loaded] Generated random continuation."
    
    if not input_string.endswith(';'):
        input_string += ';'
    
    try:
        # prompt = f"Continue this heat equation sequence with {n_predict_steps} more timesteps:\n{input_string}"
        prompt = input_string
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to the same device as the model
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        temperature = 0
        required_tokens = compute_output_tokens(n_spatial, n_predict_steps)

        with torch.inference_mode():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=required_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        llama_output = generated_text[len(prompt):].strip()

        # Parse output
        new_rows = llama_output.split(';')
        predicted_timesteps = []

        for row in new_rows[:n_predict_steps]:
            if row.strip():
                values = []
                for x in row.split(','):
                    x = x.strip()
                    if x:
                        try:
                            val = int(float(x))
                            val = max(100, min(800, val))
                            values.append(val)
                        except:
                            pass

                if len(values) == n_spatial:
                    predicted_timesteps.append(values)

        # Ensure we have enough predictions
        if len(predicted_timesteps) >= n_predict_steps:
            return np.array(predicted_timesteps[:n_predict_steps]), llama_output
        else:
            while len(predicted_timesteps) < n_predict_steps:
                if predicted_timesteps:
                    last_row = predicted_timesteps[-1]
                    new_row = [max(100, min(800, int(val + np.random.normal(0, 10)))) for val in last_row]
                else:
                    new_row = np.random.randint(100, 800, size=n_spatial).tolist()
                predicted_timesteps.append(new_row)

            return np.array(predicted_timesteps[:n_predict_steps]), llama_output if 'llama_output' in locals() else "[Parsed partial output] Filled remainder heuristically."

    except Exception as e:
        print(f"Error in Llama prediction: {e}")
        return np.random.randint(100, 800, size=(n_predict_steps, n_spatial)), f"[Error during generation] {e}"

def create_original_plot(n_spatial, n_timesteps, seed, diffusion_coeff, bc_type):
    """Create original simulation plot and input string (Plotly for hover)"""
    data = simulate_1d_heat_equation(n_spatial, n_timesteps, seed, diffusion_coeff, bc_type=bc_type)

    # Build Plotly heatmap with hover showing indices and value
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            colorscale='Hot',
            colorbar=dict(title='Temperature'),
            hovertemplate='Time step: %{y}<br>Position: %{x}<br>Temp: %{z}<extra></extra>'
        )
    )
    fig.update_layout(
        title=f'Heat Equation: {n_spatial}×{n_timesteps} Grid  •  BC: {bc_type}',
        xaxis_title=f'Spatial Position (N = {n_spatial})',
        yaxis_title=f'Time Step (T = {n_timesteps})',
        xaxis=dict(dtick=max(1, n_spatial // 8)),
        yaxis=dict(dtick=max(1, n_timesteps // 8)),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_yaxes(autorange='reversed')

    # Create input string
    rows = []
    for t in range(n_timesteps):
        row_values = [str(val) for val in data[t, :]]
        rows.append(','.join(row_values))
    input_string = ';'.join(rows) + ';'

    return fig, input_string, data

def compute_output_tokens(n_spatial, n_out_steps):
    """Compute estimated number of output tokens: 2 * N * steps"""
    return int(2 * n_spatial * n_out_steps)

def update_token_display(n_spatial, n_out_steps):
    return compute_output_tokens(n_spatial, n_out_steps)

def generate_predictions(n_spatial, n_timesteps, seed, diffusion_coeff, n_out_steps, bc_type, input_string):
    """Generate true and Llama predictions with side-by-side interactive Plotly heatmaps"""
    n_predict_steps = int(n_out_steps)

    # Get original data
    original_data = simulate_1d_heat_equation(n_spatial, n_timesteps, seed, diffusion_coeff, bc_type=bc_type)

    # Get predictions
    llama_prediction, llama_output_text = predict_with_llama(input_string, n_spatial, n_predict_steps)
    true_continuation = continue_simulation_fixed(original_data, n_predict_steps, diffusion_coeff, bc_type=bc_type)

    # Combined datasets
    true_combined = np.vstack([original_data, true_continuation])
    llama_combined = np.vstack([original_data, llama_prediction])

    # Build Plotly subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f'True Physics Continuation\n{n_spatial}×{n_timesteps + n_predict_steps}\nBC: {bc_type}',
        f'Llama 3.2 1B Prediction\n{n_spatial}×{n_timesteps + n_predict_steps}\nBC: {bc_type}'
    ))

    # Shared z range for consistent color mapping
    zmin = min(true_combined.min(), llama_combined.min())
    zmax = max(true_combined.max(), llama_combined.max())

    # Left heatmap (true)
    fig.add_trace(
        go.Heatmap(
            z=true_combined,
            colorscale='Hot',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title='Temperature'),
            hovertemplate='Time step: %{y}<br>Position: %{x}<br>Temp: %{z}<extra></extra>'
        ),
        row=1, col=1
    )

    # Cyan separator line at boundary between original and continuation
    sep_y = n_timesteps - 0.5
    fig.add_shape(type='line', x0=-0.5, x1=n_spatial-0.5, y0=sep_y, y1=sep_y,
                  line=dict(color='cyan', width=2, dash='dash'), row=1, col=1)

    # Right heatmap (llama)
    fig.add_trace(
        go.Heatmap(
            z=llama_combined,
            colorscale='Hot',
            zmin=zmin,
            zmax=zmax,
            showscale=False,  # single colorbar on the left
            hovertemplate='Time step: %{y}<br>Position: %{x}<br>Temp: %{z}<extra></extra>'
        ),
        row=1, col=2
    )

    # Separator for right plot
    fig.add_shape(type='line', x0=-0.5, x1=n_spatial-0.5, y0=sep_y, y1=sep_y,
                  line=dict(color='cyan', width=2, dash='dash'), row=1, col=2)

    # Annotations "Original" and "Physics"/"Llama"
    fig.add_annotation(text='Original', x=0.02*n_spatial, y=(n_timesteps//2), showarrow=False,
                       font=dict(color='white', size=12), xref='x1', yref='y1',
                       bgcolor='black', opacity=0.7)
    fig.add_annotation(text='Physics', x=0.02*n_spatial, y=n_timesteps + n_predict_steps//2,
                       showarrow=False, font=dict(color='white', size=12), xref='x1', yref='y1',
                       bgcolor='blue', opacity=0.7)
    fig.add_annotation(text='Original', x=0.02*n_spatial, y=(n_timesteps//2), showarrow=False,
                       font=dict(color='white', size=12), xref='x2', yref='y2',
                       bgcolor='black', opacity=0.7)
    fig.add_annotation(text='Llama', x=0.02*n_spatial, y=n_timesteps + n_predict_steps//2,
                       showarrow=False, font=dict(color='white', size=12), xref='x2', yref='y2',
                       bgcolor='red', opacity=0.7)

    fig.update_xaxes(title_text=f'Spatial Position (N = {n_spatial})', dtick=max(1, n_spatial // 8), row=1, col=1)
    fig.update_yaxes(title_text='Time Step', dtick=max(1, (n_timesteps + n_predict_steps) // 10), row=1, col=1)
    fig.update_xaxes(title_text=f'Spatial Position (N = {n_spatial})', dtick=max(1, n_spatial // 8), row=1, col=2)
    fig.update_yaxes(title_text='Time Step', dtick=max(1, (n_timesteps + n_predict_steps) // 10), row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_layout(margin=dict(l=60, r=40, t=80, b=60))

    return fig, llama_output_text

# Create Gradio interface
with gr.Blocks() as demo:
    
    gr.Markdown("# Heat Equation with LLM continuation")
    
    with gr.Row():
        with gr.Column():
            n_spatial_slider = gr.Slider(8, 64, 32, step=4, label="N (Spatial Points)")
            n_timesteps_slider = gr.Slider(4, 32, 16, step=2, label="T (Timesteps)")
            seed_slider = gr.Slider(0, 100, 42, step=1, label="Random Seed")
            diffusion_slider = gr.Slider(0.1, 2.0, 0.5, step=0.1, label="Diffusion Coefficient")
            bc_radio = gr.Radio(choices=['Periodic', 'Neumann (no flux)'], value='Periodic', label='Boundary Condition')
            output_steps_slider = gr.Slider(1, 64, 4, step=1, label="Output Steps (Extrapolated)")
            tokens_number = gr.Number(label="Estimated Output Tokens (2×N×steps)", value=2*32*4, precision=0, interactive=False)
        
        with gr.Column():
            original_plot = gr.Plot(label="Original Simulation")
    
    with gr.Row():
        input_string_box = gr.Textbox(label="LLM Input String", lines=3, interactive=False)
    
    with gr.Row():
        predict_button = gr.Button("Generate Predictions with Llama-3.2-1B", variant="primary")
    with gr.Row():
        llm_output_box = gr.Textbox(label="LLM Output String", lines=3, interactive=False)
    
    with gr.Row():
        prediction_plot = gr.Plot(label="True vs LLM Predictions (Side by Side)")
    
    # Store original data for predictions
    original_data_state = gr.State()
    
    # Update original plot and input string
    inputs = [n_spatial_slider, n_timesteps_slider, seed_slider, diffusion_slider, bc_radio]
    
    for slider in inputs:
        slider.change(
            fn=create_original_plot,
            inputs=inputs,
            outputs=[original_plot, input_string_box, original_data_state]
        )
    
    n_spatial_slider.change(
        fn=update_token_display,
        inputs=[n_spatial_slider, output_steps_slider],
        outputs=[tokens_number]
    )
    output_steps_slider.change(
        fn=update_token_display,
        inputs=[n_spatial_slider, output_steps_slider],
        outputs=[tokens_number]
    )
    
    # Generate predictions on button click
    predict_button.click(
        fn=generate_predictions,
        inputs=[n_spatial_slider, n_timesteps_slider, seed_slider, diffusion_slider, output_steps_slider, bc_radio, input_string_box],
        outputs=[prediction_plot, llm_output_box]
    )
    
    # Initialize
    demo.load(
        fn=create_original_plot,
        inputs=inputs,
        outputs=[original_plot, input_string_box, original_data_state]
    )
    demo.load(
        fn=update_token_display,
        inputs=[n_spatial_slider, output_steps_slider],
        outputs=[tokens_number]
    )

if __name__ == "__main__":
    demo.launch()