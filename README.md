# OpenAI Red-Teaming Security Assessment Framework

## Project Overview

This is a comprehensive security assessment framework designed specifically for OpenAI GPT-OSS-20B model, intended for red team competitions and security testing scenarios. The system employs a multi-agent architecture capable of identifying and testing various attack vectors, including inappropriate tool use, prompt injection, jailbreak, system prompt leakage, and RAG context leakage attacks.

## 🏗️ Project Structure

```
OpenAI-Red-Teaming/
├── main.ipynb                     # Jupyter notebook version with detailed usage examples
├── Data                           # Data for RAG
├── utils/                         # Core utility modules
│   ├── agent.py                   # Implementation of various attack agents
│   ├── gpt.py                     # GPT-OSS-20B model interface and semantic tracker
│   └── utils.py                   # Vulnerability type definitions
├── output/                        # Output results directory
│   ├── finding_jailbreak.json             # Jailbreak attack results
│   ├── finding_rag_leak.json              # RAG leakage attack results
│   ├── finding_system_prompt_leak.json    # System prompt leakage results
│   ├── finding_prompt_injection.json  # Prompt injection attack results
│   └── finding_inappropriate_tool_use.json  # Inappropriate tool use attack results
└── .gitignore                     # Git ignore file configuration
```

## 🔧 Core Components

### 1. Attack Agents (SpecializedAdversaryAgent)
- **InappropriateToolUseAgent**: Inappropriate tool use attack agent
- **PromptInjectionAgent**: Prompt injection attack agent
- **JailbreakAgent**: Jailbreak attack agent
- **SystemPromptLeakAgent**: System prompt leakage attack agent
- **RAGLeakAgent**: RAG context leakage attack agent

### 2. Model Interface (GPTOSSInterface)
- Support for Ollama API and Transformers backend
- Integrated semantic similarity computation

### 3. Semantic Tracker (UltimateSemanticTracker)
- Text embedding based on Sentence Transformers
- Semantic similarity computation and caching mechanism

## 📋 Requirements

- Python 3.12
- CUDA support (recommended for GPU acceleration)
- Ollama service (for model inference)

## 🚀 Installation and Configuration

### Configure Ollama
```bash
# Start Ollama service
ollama serve

# Pull GPT-OSS-20B model
ollama pull gpt-oss:20b
```

## 🎯 Usage

### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook main.ipynb
```

## ⚠️ Important Notice

**This framework is intended ONLY for:**
- Authorized security research and penetration testing
- Defensive security assessments
- Academic research and educational purposes

**STRICTLY PROHIBITED for:**
- Unauthorized system attacks
- Malicious purposes
- Activities violating laws and regulations

## 📄 License

This project follows the corresponding open source license. Please check the LICENSE file for details.
