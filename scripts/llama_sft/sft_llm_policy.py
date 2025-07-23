"""
sft_llm_policy.py

SFT LLM Policy module for interfacing with fine-tuned Llama models.
Simplified for Llama 8B evaluation on FrozenLake tasks.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# Set up module logger
logger = logging.getLogger(__name__)

class SFTLLMPolicy:
    """Policy class for SFT Llama 8B model evaluation."""
    
    def __init__(self, model_path: str, temperature: float = 0.1, 
                 use_quantization: bool = True, max_tokens: int = 200) -> None:
        """
        Initialize SFT Llama Policy.
        
        Args:
            model_path: Path to the fine-tuned Llama model
            temperature: Temperature for sampling (low for deterministic reasoning)
            use_quantization: Whether to use 4-bit quantization
            max_tokens: Maximum tokens to generate
        """
        
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            self._initialize_model(use_quantization)
            logger.info(f"Initialized SFT Llama model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SFT model: {e}")
            raise
    
    def _initialize_model(self, use_quantization: bool) -> None:
        """Initialize the fine-tuned Llama model."""
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded on device: {device_map}")
    
    def respond(self, prompt: str) -> str:
        """
        Generate response from SFT Llama model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response string
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response (exclude input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error getting SFT model response: {e}")
            return ""
    
    def get_model_info(self) -> dict:
        """Return information about the current SFT model."""
        return {
            "model_path": self.model_path,
            "model_type": "SFT Llama 8B",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
            del self.tokenizer
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model resources cleaned up")