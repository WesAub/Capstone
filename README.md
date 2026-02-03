# AIDE - Artificial Intelligence Design Engine

AIDE is a capstone project focused on AI-assisted analog circuit design, specifically targeting a **two-stage amplifier**.  
The goal of the project is to fine-tune and evaluate large language models (LLMs) to generate **sizing recommendations** from circuit netlists and target specifications (e.g., gain, bandwidth, phase margin, power).

## Project Overview

This repository contains code, notebooks, and configuration files for fine-tuning large language models (LLMs) using the **Hugging Face Transformers** framework within **Google Colab**.

- Models are loaded from Hugging Face
- Fine-tuning is performed in Google Colab using GPU acceleration
- Parameter-efficient fine-tuning methods (LoRA / QLoRA) are used to make training feasible
