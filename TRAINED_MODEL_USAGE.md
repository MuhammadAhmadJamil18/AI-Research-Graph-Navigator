# Using Your Trained Model - Complete Guide

## Overview

After training your model, you have several ways to use it and test it. This guide explains the complete workflow.

## What Happens After Training

When training completes, the following happens:

1. **Model is Saved**: The fine-tuned model is saved to `models/fine-tuned/` (or your configured output directory)
2. **Model Files Created**:
   - Model weights (`adapter_model.bin` for LoRA)
   - Tokenizer files
   - Configuration files
   - Training metadata

3. **UI Updates**: The sidebar shows a new "Test Trained Model" section

## Step-by-Step Workflow

### Step 1: Training Complete

After clicking "Start Training" and training completes, you'll see:
- Success message with model path
- Training configuration summary
- Model information expandable section
- Next steps instructions

### Step 2: Load the Trained Model

1. **In the Sidebar**, scroll to the "Test Trained Model" section
2. Click **"Load Trained Model"** button
3. Wait for the model to load (may take 1-2 minutes)
4. You'll see a success message when loaded

### Step 3: Test Your Model

Once loaded, you can test the model in two ways:

#### Option A: Direct Testing (Sidebar)

1. In the "Test Trained Model" section, you'll see:
   - **Test Prompt** text area (pre-filled with example)
   - **Max Length** slider (50-500 tokens)
   - **Temperature** slider (0.1-1.0)
   - **"Generate with Trained Model"** button

2. Enter your test prompt/question
3. Adjust generation parameters
4. Click "Generate with Trained Model"
5. View the generated response

**Example Test Prompts:**
- "What are the main approaches to transformer architectures?"
- "Explain how neural networks learn from data"
- "What is the difference between supervised and unsupervised learning?"
- "Describe the applications of computer vision"

#### Option B: Use in RAG Chat (Main Interface)

1. After loading the model, check the checkbox:
   - **"Use trained model for RAG answers"**
2. Go to the **"RAG Chat"** tab
3. Ask questions about your research papers
4. The trained model will generate answers using:
   - Retrieved context from your papers
   - Your fine-tuned model's domain knowledge

### Step 4: View Results

**In Direct Testing:**
- Generated response is displayed immediately
- Previous test results are saved and can be viewed in the expandable section
- You can compare different parameter settings

**In RAG Chat:**
- Answers are generated using your trained model
- Context from retrieved papers is included
- Sources are shown with relevance scores

## Where to Test Queries

### 1. Sidebar - Direct Model Testing
- **Location**: Sidebar → "Test Trained Model" section
- **Purpose**: Test the model directly without RAG
- **Best for**: 
  - Quick model evaluation
  - Testing generation parameters
  - Comparing different prompts

### 2. RAG Chat Tab - Integrated Testing
- **Location**: Main interface → "RAG Chat" tab
- **Purpose**: Use trained model with RAG pipeline
- **Best for**:
  - Testing on actual research papers
  - Getting context-aware answers
  - Production-like usage

### 3. Evaluation Tab - Quantitative Testing
- **Location**: Main interface → "Evaluation" tab
- **Purpose**: Run RAGAS metrics on your model
- **Best for**:
  - Measuring performance
  - Comparing with baseline models
  - Research evaluation

## Output Locations

### Model Files
- **Location**: `models/fine-tuned/` (default)
- **Contents**:
  - `adapter_model.bin` - LoRA adapter weights
  - `adapter_config.json` - LoRA configuration
  - `tokenizer files` - Tokenizer and vocabulary
  - `training_args.bin` - Training arguments

### Test Results
- **Location**: Session state (in-memory during session)
- **Access**: Expandable "Previous Test Results" section
- **Note**: Results are cleared when you refresh the page

### Generated Answers
- **Location**: Chat interface (RAG Chat tab)
- **Format**: Markdown-formatted responses
- **Includes**: 
  - Generated answer
  - Source papers
  - Relevance scores

## Programmatic Usage

You can also use the trained model programmatically:

```python
from src.model_training import ModelTrainer

# Load the trained model
trainer = ModelTrainer(
    base_model="microsoft/Phi-2",
    output_dir="models/fine-tuned"
)
trainer.load_model("models/fine-tuned")

# Generate text
response = trainer.generate(
    prompt="Your prompt here",
    max_length=200,
    temperature=0.7
)
print(response)
```

## Tips for Best Results

1. **Test Different Prompts**: Try various question formats to see what works best
2. **Adjust Temperature**: 
   - Lower (0.1-0.5): More focused, deterministic
   - Higher (0.7-1.0): More creative, varied
3. **Use RAG Integration**: Combine with RAG for context-aware answers
4. **Compare Results**: Test with and without the trained model to see improvements
5. **Monitor Performance**: Use the Evaluation tab to measure metrics

## Troubleshooting

**Model won't load:**
- Check that the model directory exists
- Verify all files are present in `models/fine-tuned/`
- Check console for error messages

**Generation is slow:**
- This is normal for larger models
- Consider using GPU for faster generation
- Reduce max_length for faster responses

**Answers seem generic:**
- The model may need more training data
- Try adjusting temperature
- Use RAG integration for better context

**Model not appearing in RAG Chat:**
- Make sure you clicked "Load Trained Model"
- Check the "Use trained model for RAG answers" checkbox
- Verify model_trainer is in session state

## Next Steps

After testing your model:

1. **Evaluate Performance**: Use the Evaluation tab with RAGAS metrics
2. **Fine-tune Further**: Adjust training parameters and retrain if needed
3. **Deploy**: Use the model in your applications
4. **Compare**: Test against baseline models to measure improvement

---

**Note**: The trained model is session-specific. To use it in a new session, you'll need to load it again using the "Load Trained Model" button.

