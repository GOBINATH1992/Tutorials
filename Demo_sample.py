from openai import OpenAI
'''
pip install transformers[serving]  
transformers serve  google/gemma-4-E2B-it
This code demonstrate use of popular AI models using API and transformers library. 
It includes two main classes, Gemma4_OpenAI and Gemma4_Transformers, which interact with the models through different interfaces.
The code also includes a demonstration of ONNX Runtime inference for the Qwen 3.5 model.
Each class contains methods to sample text, images, audio, and video, as well as a tool calling mechanism for executing external functions based on model prompts.
'''

from openai import OpenAI
import json
import re
import torch
import httpx
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.image_utils import load_image

import onnxruntime
import numpy as np
import json
import os
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download


# --- Gemma4_OpenAI Class ---

class Gemma4_OpenAI:
    """
    A client class designed to interact with a local Gemma model API 
    (simulated via OpenAI compatibility) for various modalities (text, image, audio, video) 
    and tool calling.
    """
    def __init__(self):
        """
        Initializes the Gemma4_OpenAI client.

        Sets up the base URL for the local API and initializes the OpenAI client.
        """
        # Base URL for the local API endpoint
        self.base_url = "http://localhost:8000/v1"
        # Initialize the OpenAI client pointing to the local endpoint
        self.client = OpenAI(base_url= self.base_url, api_key="<random_string>")
        # Flag to determine if the response should be streamed
        self.streaming_response = True
    
    def sample_text(self):
        """
        Samples a text response from the Gemma model.

        Args:
            None

        Prints the generated text (either directly or via streaming).
        """
        completion = self.client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=[
                {
                    "role": "user",
                    "content": "What is the Transformers library known for?"
                }
            ],
            stream=self.streaming_response,
        )
        
        if self.streaming_response is False:
            # Non-streaming call
            print(completion.choices[0].message.content)
        else:
            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def sample_image(self):
        """
        Samples a text description based on a provided image URL.

        Args:
            None

        Performs the image-to-text task using the API.
        """
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        completion = self.client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                    ],
                }
            ],
            stream=self.streaming_response,
        )

        if self.streaming_response is False:
            # Non-streaming call
            print(completion.choices[0].message.content)
    
        else:
            # Streaming call
            
            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def sample_video(self):
        """
        Samples a description based on a provided video URL.

        Args:
            None

        Performs the video-to-text task using the API.
        """
        video_url = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"
        
        completion = self.client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": "What is happening in the video and what is the song about?"},
                    ],
                }
            ],
            stream=self.streaming_response,
        )


        if self.streaming_response is False:
            # Non-streaming call
            
            print(completion.choices[0].message.content)
    
        else:
            # Streaming call
            
            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def sample_audio(self):
        """
        Samples a transcription of an audio file.

        This function downloads an audio file, encodes it to base64, and sends 
        it to the model for transcription.

        Args:
            None

        Performs audio transcription using the API.
        """
        # Setup for audio processing
        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
        # Download and encode the audio file in base64 format
        audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()
        completion = self.client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio."},
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                    ],
                }
            ],
            stream=self.streaming_response,
        )
    
        if self.streaming_response is False:
            # Non-streaming call
            
            print(completion.choices[0].message.content)
        else:
            # Streaming call

            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def tool_call(self):
        """
        Demonstrates the tool calling mechanism of the model.

        This function sets up a function definition, prompts the model to use it,
        executes the tool locally, and then sends the result back to the model 
        for a final answer.

        Returns:
            None

        Prints the final response after the tool execution loop.
        """
        # Initialize client for tool call API interaction
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="<KEY>")

        # Define the available tools the model can use
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. San Francisco"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        # 1. Initial User Prompt
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        
        # 2. Model returns a tool call request
        response = client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=messages,
            tools=tools,
        )
        assistant_message = response.choices[0].message

        # 3. Execute the tool locally (Simulated execution)
        tool_call = assistant_message.tool_calls[0]
        # In a real application, you would call the actual weather API here.
        result = {"temperature": 22, "condition": "sunny"}  

        # 4. Send the tool result back to the model
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })
        
        # 5. Final response generation
        final_response = client.chat.completions.create(
            model="google/gemma-4-E2B-it",
            messages=messages,
            tools=tools,
        )
        print("\n--- Final Response After Tool Execution ---")
        print(final_response.choices[0].message.content)


# --- Gemma4_Transformers Class ---

class Gemma4_Transformers:
    """
    A client class for interacting with Gemma models using the Hugging Face 
    `transformers` library directly.
    """
    def __init__(self):
        """
        Initializes the transformer models and processor.
        """
        self.use_draft_model = False
        TARGET_MODEL_ID = "google/gemma-4-E2B-it"
        ASSISTANT_MODEL_ID = "google/gemma-4-E2B-it-assistant"
        
        # Initialize the processor for tokenization and chat template application
        self.processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
        
        # Initialize the main target model
        self.target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
        
        if self.use_draft_model:
            # Initialize the assistant model (used for instruction following/drafting)
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL_ID,
                dtype="auto",
                device_map="auto",
            )
        else:
             self.assistant_model = None


    def sample_text(self):
        """
        Generates a text response using the chat template and transformer pipeline.

        Prints the generated text.
        """
        # Define the conversation prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the Transformers library known for?"},
        ]

        # Process input using the chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.target_model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.target_model.generate(
            **inputs,
            assistant_model=self.assistant_model,
            max_new_tokens=256,
        )
        # Decode the generated text, skipping the input prompt tokens
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse and print the result
        result = self.processor.parse_response(response)
        print(result["content"])


    def sample_image(self):
        """
        Generates a text response based on an image provided via URL.

        Prints the generated text.
        """
        # Define the prompt including an image URL
        messages = [
            {
                "role": "user", "content": [
                    {"type": "image", "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"},
                    {"type": "text", "text": "What is shown in this image?"}
                ]
            }
        ]

        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.target_model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.target_model.generate(**inputs, max_new_tokens=512, assistant_model=self.assistant_model,)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse and print the result
        result = self.processor.parse_response(response)
        print(result["content"])


    def sample_audio(self):
        """
        Generates a transcription of an audio file provided via URL.

        Prints the transcribed text.
        """
        # Define the prompt including an audio URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "https://github.com/google-gemma/cookbook/raw/refs/heads/main/apps/sample-data/journal1.wav"},
                    {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
                ]
            }
        ]

        # Process input
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.target_model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.target_model.generate(
            **inputs,
            assistant_model=self.assistant_model,
            max_new_tokens=256,
        )
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse and print the result
        result = self.processor.parse_response(response)
        print(result["content"])


    def sample_video(self):
        """
        Generates a description of a video provided via URL.

        Prints the generated description.
        """
        # Define the prompt including a video URL
        messages = [
            {
                'role': 'user',
                'content': [
                    {"type": "video", "video": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4"},
                    {'type': 'text', 'text': 'Describe this video.'}
                ]
            }
        ]

        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.target_model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.target_model.generate(**inputs, max_new_tokens=512, assistant_model=self.assistant_model,)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse and print the result
        result = self.processor.parse_response(response)
        print(result["content"])
    
    def resize_to_48_multiple(self, image: Image.Image):
        """
        Resizes an image so its dimensions are multiples of 48.

        Args:
            image: The PIL Image object to resize.

        Returns:
            A cropped PIL Image object.
        """
        w, h = image.size
        new_w = (w // 48) * 48
        new_h = (h // 48) * 48
        return image.crop((0, 0, new_w, new_h))

    def extract_json_(self, text: str) -> dict:
        """
        Attempts to extract a JSON object from a string, handling markdown fences.

        Args:
            text: The raw text string potentially containing JSON.

        Returns:
            A parsed JSON dictionary.

        Raises:
            ValueError: If no valid JSON can be found.
        """
        text = text.strip()

        # Remove markdown formatting (e.g., ```json ... ```)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first potential JSON object or array
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            candidate = match.group(1)
            # print(f"Found candidate JSON: {candidate}") # Optional: debug output
            return json.loads(candidate)
    
        raise ValueError("No valid JSON found in the provided text.")


    def extract_json(self, text: str) -> dict:
        """
        A simplified function to extract JSON, specifically looking for content 
        inside the first markdown block.

        Args:
            text: The raw text string.

        Returns:
            A parsed JSON dictionary.
        """
        text = text.strip()

        # Extract content from the first code block (assuming standard markdown format)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = text.split('``')[0]
        
        return json.loads(text)


    def object_detection(self):
        """
        Performs object detection on an image using the multimodal model.
        It asks the model for bounding boxes and then draws the detected area 
        on the image.
        """
        image_url = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bike.png"
        image = load_image(image_url)
        
        # Resize the image to fit the 48-multiple constraint
        image = self.resize_to_48_multiple(image)
        
        messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's the bounding box for the bike in the image?. Return the output in json"}
            ]
        }
        ]

        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False, # Disable internal reasoning for faster detection
        )
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate output (using greedy decoding for structured output)
        generated_outputs = self.target_model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        generated = self.processor.decode(generated_outputs[0, input_len:])
        print(f"\n--- Object Detection Output ---\n{generated}")
        
        # Parse the JSON output
        detection = self.extract_json(generated)[0]
        box = detection["box_2d"]
        label = detection.get("label", "bike")
        print(f"Detected Box (xmin, ymin, xmax, ymax): {box}, Label: {label}")
        
        # Draw the bounding box on the image
        width, height = image.size
        ymin, xmin, ymax, xmax = box
        
        # Calculate normalized coordinates and map them to the original image size
        resize_shape=(1000,1000)
        re_h, re_w = resize_shape if resize_shape is not None else (height, width)
        
        # Re-scale coordinates based on the target dimensions
        xmin = (xmin / re_w) * width
        ymin = (ymin / re_h) * height
        xmax = (xmax / re_w) * width
        ymax = (ymax / re_h) * height
        
        w = xmax - xmin
        h = ymax - ymin
        
        # Draw the rectangle
        draw = ImageDraw.Draw(image)
        draw.rectangle(((xmin, ymin), (xmin + w, ymin + h)),width= 5)
        image.save('out.png', "PNG")
        print("Bounding box drawn and saved to out.png")


# --- Qwen3_5 OpenAI Class ---

class Qwen3_5_OpenAI:
    """
    A client class designed to interact with a local Qwen model API 
    (simulated via OpenAI compatibility) for various modalities and tool calling.
    """
    def __init__(self):
        """
        Initializes the Qwen3_5 client.

        Sets up the base URL, API key, and model name.
        """
        self.base_url = "http://localhost:8000/v1"
        self.client = OpenAI(base_url= self.base_url, api_key="<random_string>")
        self.model_name = "Qwen/Qwen3.5-0.8B" 
        self.streaming_response = True
    
    def sample_text(self):
        """
        Samples a text response from the Qwen model.

        Prints the generated text (either directly or via streaming).
        """
        completion = self.client.chat.completions.create(
            model= self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What is the Transformers library known for?"
                }
            ],
            stream=self.streaming_response,
        )
        if self.streaming_response is False:
            # Non-streaming call
            
            print(completion.choices[0].message.content)
        else:
            # Streaming call

            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def sample_image(self):
        """
        Samples a text description based on a provided image URL.

        Prints the generated text.
        """
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"

        completion = self.client.chat.completions.create(
                model= self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                        ],
                    }
                ],
                stream=self.streaming_response,
            )


        if self.streaming_response is False:
            # Non-streaming call
            
            print(completion.choices[0].message.content)
    
        else:
            # Streaming call
            
            # Stream the response token by token
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="")


    def tool_call(self):
        """
        Demonstrates the tool calling mechanism of the Qwen model.

        This function sets up a function definition, prompts the model to use it,
        executes the tool locally, and then sends the result back to the model 
        for a final answer.
        """
        # Initialize client for tool call API interaction
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="<KEY>")

        # Define the available tools the model can use
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. San Francisco"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        # 1. Initial User Prompt
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        
        # 2. Model returns a tool call request
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
        )
        assistant_message = response.choices[0].message

        # 3. Execute the tool locally (Simulated execution)
        tool_call = assistant_message.tool_calls[0]
        # In a real application, you would call the actual weather API here.
        result = {"temperature": 22, "condition": "sunny"}  

        # 4. Send the tool result back to the model
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })
        
        # 5. Final response generation
        final_response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
        )
        print("\n--- Final Response After Tool Execution ---")
        print(final_response.choices[0].message.content)

# ==============================================================================
# 4. Qwen 3.5 ONNX Runtime Inference
# ==============================================================================

def Qwen3_5_ONNX_Inference():
    """
     Performs text and image inference using the Qwen 3.5 ONNX model via ONNX Runtime.
    """
    print("\n" + "="*70)
    print("--- Qwen 3.5 ONNX Runtime Inference Demo ---")
    print("This method runs the model natively using ONNX Runtime.")
    print("="*70)

    # 1. Load models
    ## Load config and processor
    model_id = "onnx-community/Qwen3.5-0.8B-ONNX"
    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)

    ## Load sessions
    vision_model = "onnx/vision_encoder_q4.onnx"
    embed_model = "onnx/embed_tokens_q4.onnx"
    decoder_model = "onnx/decoder_model_merged_q4.onnx"
    model_dir = snapshot_download(model_id, allow_patterns=[ f"{vision_model}*", f"{embed_model}*", f"{decoder_model}*"])
    vision_model_path  = os.path.join(model_dir, vision_model)
    embed_model_path   = os.path.join(model_dir, embed_model)
    decoder_model_path = os.path.join(model_dir, decoder_model)

    providers = ['CPUExecutionProvider']
    vision_session  = onnxruntime.InferenceSession(vision_model_path, providers=providers)
    embed_session   = onnxruntime.InferenceSession(embed_model_path, providers=providers)
    decoder_session = onnxruntime.InferenceSession(decoder_model_path, providers=providers)

    ## Set config values
    eos_token_id = generation_config.eos_token_id
    image_token_id = config.image_token_id
    

    # 2. Prepare inputs
    ## Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "In detail, describe the following image."},
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].numpy() # shape [1, 2774]
    attention_mask = inputs["attention_mask"].numpy()  # shape [1, 2774] # mm_token_type_ids [1, 2774]  
    pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None #[11008, 1536]
    image_grid_thw = inputs["image_grid_thw"].numpy() if "image_grid_thw" in inputs else None

    ## Prepare decoder inputs
    batch_size = input_ids.shape[0]
    num_logits_to_keep = np.array(1, dtype=np.int64)
     
    past_cache_values = {}
    for inp in decoder_session.get_inputs():
        name = inp.name
        shape = inp.shape
        dtype = np.float32 if inp.type == "tensor(float)" else np.float16
        if name.startswith("past_key_values"):
            # Attention KV cache: shape [batch_size, num_kv_heads, 0, head_dim]
            past_cache_values[name] = np.zeros([batch_size, shape[1], 0, shape[3]], dtype=dtype)
        elif name.startswith("past_conv"):
            # Conv cache: shape [batch_size, hidden_size, conv_L_cache]   ['batch_size', 6144, 4]
            past_cache_values[name] = np.zeros([batch_size, shape[1], shape[2]], dtype=dtype)
        elif name.startswith("past_recurrent"):
            #  ['batch_size', 16, 128, 128]
            past_cache_values[name] = np.zeros([batch_size, shape[1], shape[2], shape[3]], dtype=dtype)
    
    # vision session 
    # input pixel_values ['num_patches', 1536] image_grid_thw  ['num_images', 3]
    # image_features ['num_features', 1024]
    # embed_session
    # inputs_embeds :  shape [1, 2774, 1024]
    # decoder_session
    # inputs_embeds ['batch_size', 'sequence_length', 1024]
    # attention_mask ['batch_size', 'total_sequence_length'] 
    # position_ids [3, 'batch_size', 'sequence_length']  3,1,2774 
    # past_key_values [batch_size, num_kv_heads, 0, head_dim]
    # past_conv [batch_size, hidden_size, conv_L_cache]
    # past_cache_values[name] ['batch_size', 16, 128, 128]

    position_ids = np.cumsum(attention_mask, axis=-1) - 1
    position_ids = np.repeat(position_ids[np.newaxis, :, :], 3, axis=0)
    # 3. Generation loop
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)
    image_features = None
    for i in range(max_new_tokens):
        inputs_embeds = embed_session.run(None, {"input_ids": input_ids})[0] # inputs_embeds :  shape [1, 2774, 1024]
        if image_features is None and pixel_values is not None:
            image_features = vision_session.run(["image_features"], {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})
            mask = (input_ids == image_token_id).reshape(-1)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_embeds[mask] = image_features
            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

        logits, *present_key_values = decoder_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_cache_values,
            ))
        
        ## Update values for next generation loop
        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
        position_ids = position_ids[:, -1:] + 1
        for j, key in enumerate(past_cache_values):
            past_cache_values[key] = present_key_values[j]

        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        if np.isin(input_ids, eos_token_id).any():
            break

        ## (Optional) Streaming
        ##print(processor.decode(input_ids[0]), end="", flush=True)
    print()
    # --- Output Result ---
    print("\n--- Qwen 3.5 ONNX Final Output ---")
    print(processor.batch_decode(generated_tokens, skip_special_tokens=True)[0])    
    return




# ==============================================================================
# 5. Qwen 3 ONNX Runtime Inference
# ==============================================================================


def Qwen3_ONNX_Inference():
    """
     Performs text and image inference using the Qwen 3 ONNX model via ONNX Runtime.
    """
    print("\n" + "="*70)
    print("--- Qwen 3 ONNX Runtime Inference Demo ---")
    print("This method runs the model natively using ONNX Runtime.")
    print("="*70)

    # 1. Load models
    ## Load config and processor
    model_id = "onnx-community/Qwen3-0.6B-ONNX"
    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    ## Load sessions
    model = "onnx/model_q4.onnx"
    model_dir = snapshot_download(model_id, allow_patterns=[ f"{model}*"])
    #model_dir = snapshot_download(model_id)
    model_path  = os.path.join(model_dir, model)
    print(f"Model Dir : {model_dir}")
    providers = ['CPUExecutionProvider']
    model_session  = onnxruntime.InferenceSession(model_path, providers=providers)

    ## Set config values
    eos_token_id = generation_config.eos_token_id
    
    # 2. Prepare inputs
    ## Create input messages
   
    messages = [
        { "role" : "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": "Write me a poem about Machine Learning." }
        ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    position_ids = np.cumsum(attention_mask, axis=-1) - 1

    ## Prepare decoder inputs
    batch_size = input_ids.shape[0]
    num_logits_to_keep = np.array(1, dtype=np.int64)
    past_key_values = {
        inp.name: np.zeros(
            [batch_size, inp.shape[1], 0, inp.shape[3]],
            dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
        )
        for inp in model_session.get_inputs()
        if inp.name.startswith("past_key_values")
    }
    # 3. Generation loop
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)
    image_features = None
    audio_features = None
    for i in range(max_new_tokens):
        logits, *present_key_values = model_session.run(None, dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_key_values,
        ))

        ## Update values for next generation loop
        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
        position_ids = position_ids[:, -1:] + 1
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        if np.isin(input_ids, eos_token_id).any():
            break

        ## (Optional) Streaming
        #print(processor.decode(input_ids[0]), end="", flush=True)
    print('-'*100)

    # --- Output Result ---
    print("\n--- Qwen 3 ONNX Final Output ---")

    response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(response)

   

# ==============================================================================
# 6. Gemma4 ONNX Runtime Inference
# ==============================================================================    


def Gemma4_ONNX_Inference():
    """
     Performs text and image inference using the Gemma4 ONNX model via ONNX Runtime.
    """
    print("\n" + "="*70)
    print("--- Gemma4 ONNX Runtime Inference Demo ---")
    print("This method runs the model natively using ONNX Runtime.")
    print("="*70)
            
    # 1. Load models
    ## Load config and processor
    model_id = "onnx-community/gemma-4-E2B-it-ONNX"
    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)

    ## Load sessions
    audio_model = "onnx/audio_encoder_q4.onnx"
    vision_model = "onnx/vision_encoder_q4.onnx"
    embed_model = "onnx/embed_tokens_q4.onnx"
    decoder_model = "onnx/decoder_model_merged_q4.onnx"
    model_dir = snapshot_download(model_id, allow_patterns=[f"{audio_model}*",  f"{vision_model}*", f"{embed_model}*", f"{decoder_model}*"])
    audio_model_path   = os.path.join(model_dir, audio_model)
    vision_model_path  = os.path.join(model_dir, vision_model)
    embed_model_path   = os.path.join(model_dir, embed_model)
    decoder_model_path = os.path.join(model_dir, decoder_model)

    providers = ['CPUExecutionProvider']
    vision_session  = onnxruntime.InferenceSession(vision_model_path, providers=providers)
    audio_session   = onnxruntime.InferenceSession(audio_model_path, providers=providers)
    embed_session   = onnxruntime.InferenceSession(embed_model_path, providers=providers)
    decoder_session = onnxruntime.InferenceSession(decoder_model_path, providers=providers)

    ## Set config values
    eos_token_id = generation_config.eos_token_id
    image_token_id = config.image_token_id
    audio_token_id = config.audio_token_id

    # 2. Prepare inputs
    ## Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "In detail, describe the following audio and image."},
                {"type": "audio", "audio": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"},
                {"type": "image", "image": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg"},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    position_ids = np.cumsum(attention_mask, axis=-1) - 1

    pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
    pixel_position_ids = inputs["image_position_ids"].numpy() if "image_position_ids" in inputs else None
    input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
    input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None

    ## Prepare decoder inputs
    batch_size = input_ids.shape[0]
    num_logits_to_keep = np.array(1, dtype=np.int64)
    past_key_values = {
        inp.name: np.zeros(
            [batch_size, inp.shape[1], 0, inp.shape[3]],
            dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
        )
        for inp in decoder_session.get_inputs()
        if inp.name.startswith("past_key_values")
    }

    # 3. Generation loop
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)
    image_features = None
    audio_features = None
    for i in range(max_new_tokens):
        inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})
        if image_features is None and pixel_values is not None:
            image_features = vision_session.run(["image_features"], {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids})[0]
            mask = (input_ids == image_token_id).reshape(-1)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_embeds[mask] = image_features
            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

        if audio_features is None and input_features is not None and input_features_mask is not None:
            audio_features = audio_session.run(
                ["audio_features"],
                {"input_features": input_features, "input_features_mask": input_features_mask},
            )[0]
            mask = (input_ids == audio_token_id).reshape(-1)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_embeds[mask] = audio_features
            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

        logits, *present_key_values = decoder_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            num_logits_to_keep=num_logits_to_keep,
            **past_key_values,
        ))

        ## Update values for next generation loop
        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
        position_ids = position_ids[:, -1:] + 1
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        if np.isin(input_ids, eos_token_id).any():
            break

        ## (Optional) Streaming
        print(processor.decode(input_ids[0]), end="", flush=True)
    print()

    # --- Output Result ---
    print("\n--- Gemma4 ONNX Final Output ---")
    # 4. Output result
    print(processor.batch_decode(generated_tokens, skip_special_tokens=True)[0])



# ==============================================================================
# 7. LFM2 ONNX Runtime Inference
# ==============================================================================

def LFM2_ONNX_Inference():
    from transformers import AutoConfig, AutoTokenizer
    """
    Performs text inference using the LFM2 model via ONNX Runtime.
    """
    print("\n" + "="*70)
    print("--- LFM2 ONNX Runtime Inference Demo ---")
    print("This method runs the LFM2 model natively using ONNX Runtime.")
    print("="*70)

    # --- Model Setup ---
    model_id = "onnx-community/LFM2-700M-ONNX"
    
    # Download the ONNX graph and weights
    filename = "model_q4.onnx" 
    model_path = snapshot_download(repo_id=model_id, allow_patterns=f"onnx/{filename}*") 
    
    # Initialize the ONNX session
    session = onnxruntime.InferenceSession(f"{model_path}/onnx/{filename}")

    # --- Prepare Inputs ---
    prompt = "What is C. elegans?"
    messages = [{"role": "user", "content": prompt}]
    
    # Use the tokenizer to format the prompt into model inputs
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos_token_id = config.eos_token_id
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    batch_size = input_ids.shape[0]
    num_logits_to_keep = np.array(1, dtype=np.int64)

    # Prepare KV Cache structure
    past_cache_values = {}
    for inp in session.get_inputs():
        name = inp.name
        shape = inp.shape
        dtype = np.float32 if inp.type == "tensor(float)" else np.float16
        if name.startswith("past_key_values"):
            past_cache_values[name] = np.zeros([batch_size, shape[1], 0, shape[3]], dtype=dtype)
        elif name.startswith("past_conv"):
            past_cache_values[name] = np.zeros([batch_size, shape[1], shape[2]], dtype=dtype)

    # --- Generation Loop ---
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)

    for i in range(max_new_tokens):
        # Run the model session
        logits, *present_cache_values = session.run(None, dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_logits_to_keep=num_logits_to_keep,
            **past_cache_values,
        ))
        
        # Update KV Cache and Input IDs
        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
        
        for j, key in enumerate(past_cache_values):
            past_cache_values[key] = present_cache_values[j]
            
        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        
        if np.isin(input_ids, eos_token_id).any():
            break

        # Optional Streaming
        print(tokenizer.decode(input_ids[0]), end='', flush=True)
        print()

    # --- Output Result ---
    print("\n--- LFM2 ONNX Final Output ---")
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])




# --- Execution ---
# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("STARTING MODEL INFERENCE DEMONSTRATION")
    print("="*80)

    # --- DEMO 1: Gemma 4 API (OpenAI style) ---
    print("\n\n#######################################################")
    print("### DEMO 1: Gemma 4 API (External/Cloud Inference) ###")
    print("#######################################################")

    gemma4_api = Gemma4_OpenAI()
    
    # 1. Text Inference (Streaming)
    gemma4_api.sample_text()

    # 2. Image Inference (Streaming)
    gemma4_api.sample_image()
    
    # 3. Tool Calling Demonstration
    gemma4_api.tool_call()


    # --- DEMO 2: Gemma 4 Transformers (Hugging Face Pipeline) ---
    print("\n\n#######################################################")
    print("### DEMO 2: Gemma 4 Transformers (Local Pipeline) ###")
    print("#######################################################")

    gemma4_hf = Gemma4_Transformers()

    # 1. Text Inference
    gemma4_hf.sample_text()

    # 2. Image Inference
    gemma4_hf.sample_image()

    # 3. Object Detection (Multimodal)
    gemma4_hf.object_detection()


    # --- DEMO 3: Qwen 3.5 API (External/Cloud Inference) ---
    print("\n\n#######################################################")
    print("### DEMO 3: Qwen 3.5 API (External/Cloud Inference) ###")
    print("#######################################################")

    qwen3_api = Qwen3_5_OpenAI()

    # 1. Text Inference (Streaming)
    qwen3_api.sample_text()

    # 2. Image Inference (Streaming)
    qwen3_api.sample_image()

    # 3. Tool Calling Demonstration
    qwen3_api.tool_call()


    # --- DEMO 4: Qwen 3.5 ONNX Runtime Inference ---
    print("\n\n#######################################################")
    print("### DEMO 4: Qwen 3.5 ONNX Runtime (Native Inference) ###")
    print("#######################################################")

    Qwen3_5_ONNX_Inference()
    
    # --- DEMO 5: Qwen 3 ONNX Runtime Inference ---
    print("\n\n#######################################################")
    print("### DEMO 5: Qwen 3 ONNX Runtime (Native Inference) ###")
    print("#######################################################")

    Qwen3_ONNX_Inference()
    
    # --- DEMO 6: Gemma4 ONNX Runtime Inference ---
    print("\n\n#######################################################")
    print("### DEMO 6: Gemma4 ONNX Runtime (Native Inference) ###")
    print("#######################################################")

    Gemma4_ONNX_Inference()

    # --- DEMO 7: LFM2 ONNX Runtime Inference ---
    print("\n\n#######################################################")
    print("### DEMO 6: LFM2 ONNX Runtime (Native Inference) ###")
    print("#######################################################")

    LFM2_ONNX_Inference()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
