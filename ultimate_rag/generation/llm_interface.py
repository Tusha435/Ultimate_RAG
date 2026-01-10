"""LLM interface for multiple providers."""

from typing import Optional, Any
from abc import ABC, abstractmethod
from loguru import logger

from ultimate_rag.config import GenerationConfig


class BaseLLM(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        """Generate response with streaming."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI API interface."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()

        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.llm_model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.config.max_response_tokens),
            temperature=kwargs.get("temperature", self.config.temperature)
        )

        return response.choices[0].message.content

    def generate_stream(self, prompt: str, **kwargs):
        client = self._get_client()

        stream = client.chat.completions.create(
            model=kwargs.get("model", self.config.llm_model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.config.max_response_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API interface."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()

        response = client.messages.create(
            model=kwargs.get("model", self.config.llm_model),
            max_tokens=kwargs.get("max_tokens", self.config.max_response_tokens),
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_stream(self, prompt: str, **kwargs):
        client = self._get_client()

        with client.messages.stream(
            model=kwargs.get("model", self.config.llm_model),
            max_tokens=kwargs.get("max_tokens", self.config.max_response_tokens),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaLLM(BaseLLM):
    """Ollama local LLM interface."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.base_url = config.ollama_base_url

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import ollama

            response = ollama.generate(
                model=kwargs.get("model", self.config.llm_model),
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_response_tokens)
                }
            )

            return response["response"]

        except ImportError:
            import httpx

            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": kwargs.get("model", self.config.llm_model),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature)
                    }
                },
                timeout=120.0
            )

            return response.json()["response"]

    def generate_stream(self, prompt: str, **kwargs):
        try:
            import ollama

            stream = ollama.generate(
                model=kwargs.get("model", self.config.llm_model),
                prompt=prompt,
                stream=True,
                options={
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
            )

            for chunk in stream:
                yield chunk["response"]

        except ImportError:
            import httpx

            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": kwargs.get("model", self.config.llm_model),
                    "prompt": prompt,
                    "stream": True
                },
                timeout=120.0
            ) as response:
                import json
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        yield data.get("response", "")


class LLMInterface:
    """Unified LLM interface supporting multiple providers."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._llm: Optional[BaseLLM] = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on config."""
        provider = self.config.llm_provider

        if provider == "openai":
            self._llm = OpenAILLM(self.config)
            logger.info("Using OpenAI LLM")

        elif provider == "anthropic":
            self._llm = AnthropicLLM(self.config)
            logger.info("Using Anthropic LLM")

        elif provider == "ollama":
            self._llm = OllamaLLM(self.config)
            logger.info(f"Using Ollama LLM at {self.config.ollama_base_url}")

        elif provider == "none":
            self._llm = None
            logger.info("LLM disabled, using structured responses only")

        else:
            logger.warning(f"Unknown LLM provider: {provider}")
            self._llm = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        if self._llm is None:
            raise RuntimeError("No LLM configured")

        return self._llm.generate(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """Generate response with streaming."""
        if self._llm is None:
            raise RuntimeError("No LLM configured")

        yield from self._llm.generate_stream(prompt, **kwargs)

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._llm is not None

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text."""
        if not self.is_available():
            return text[:max_length] + "..." if len(text) > max_length else text

        prompt = f"""Summarize the following text in {max_length} words or less:

{text}

Summary:"""

        return self.generate(prompt)

    def explain_formula(self, latex: str, context: Optional[str] = None) -> str:
        """Generate explanation for a formula."""
        if not self.is_available():
            return f"Formula: ${latex}$"

        prompt = f"""Explain the following mathematical formula:

Formula: ${latex}$

{f'Context: {context}' if context else ''}

Provide:
1. What the formula represents
2. Explanation of each variable
3. Common applications

Explanation:"""

        return self.generate(prompt)

    def compare_concepts(self, concept1: str, concept2: str, context: str = "") -> str:
        """Compare two concepts."""
        if not self.is_available():
            return f"Comparison of {concept1} and {concept2}"

        prompt = f"""Compare and contrast the following concepts:

Concept 1: {concept1}
Concept 2: {concept2}

{f'Context: {context}' if context else ''}

Provide a structured comparison covering:
1. Similarities
2. Differences
3. When to use each

Comparison:"""

        return self.generate(prompt)
