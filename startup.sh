#!/bin/bash
# Script to setup env for AI work
#
export CMAKE_ARGS="-DGGML_METAL=on"
export FORCE_CMAKE=1
source deactivate
pyenv shell 3.11.0
if [ ! -d "./env" ]; then
   echo "environment doesnt exist, installing needed packages"
   python -m venv ./env
   source ./env/bin/activate
   pip install langchain langchain-community chromadb python-dotenv streamlit sentence-transformers pypdfium2
   pip install unstructured pdf2image pdfminer.six
   pip install pdf-info
   pip install "unstructured[pdf]"
   pip install -U langchain-huggingface
   pip install -U langchain-chroma
   pip install watchdog
   CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   brew install poppler tesseract libmagic
#   brew --prefix poppler ====> it shows the PATH
else
   echo "environment exists, activating environment"    
   source ./env/bin/activate
fi

