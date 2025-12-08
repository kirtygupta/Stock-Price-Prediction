#!/usr/bin/env python3
import subprocess
import sys
import time

def run_command(command):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def setup_ollama():
    print("🚀 Setting up Ollama for ChatBot...")
    print("="*60)
    
    # Check if Ollama is installed
    print("🔍 Checking Ollama installation...")
    code, stdout, stderr = run_command("ollama --version")
    
    if code != 0:
        print("❌ Ollama not found or not in PATH")
        print("Please install Ollama from: https://ollama.ai/download")
        print("\nAfter installation, run:")
        print("1. ollama serve")
        print("2. ollama pull llama2")
        return False
    
    print("✅ Ollama is installed")
    
    # Check if Ollama is running
    print("\n🔍 Checking if Ollama is running...")
    try:
        import ollama
        models = ollama.list()
        print(f"✅ Ollama is running with {len(models['models'])} model(s)")
        
        if len(models['models']) == 0:
            print("\n⚠️ No models found. Pulling a default model...")
            print("This may take a few minutes...")
            ollama.pull('llama2')
            print("✅ Model pulled successfully!")
        
    except Exception as e:
        print(f"❌ Ollama is not running: {e}")
        print("\nPlease start Ollama in a separate terminal:")
        print("$ ollama serve")
        print("\nThen pull a model:")
        print("$ ollama pull llama2")
        return False
    
    print("\n" + "="*60)
    print("✅ Ollama setup complete!")
    print("🤖 ChatBot is ready to use!")
    print("="*60)
    return True

if __name__ == "__main__":
    setup_ollama()