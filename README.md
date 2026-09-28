# Fact and Logic: Detecting Misinformation Through Evidence and Fallacy Analysis

## Problem
Misinformation spreads through false claims and through flawed reasoning. This project checks both.

## What it does
- **Input:** a claim or short argument in text.
- **Module 1 (Fact check):** retrieves evidence from a fixed corpus and labels the claim Supported / Refuted / Not Enough Info, with sources.
- **Module 2 (Fallacy check):** detects common logical fallacies (ad hominem, false dilemma, slippery slope, appeal to authority, hasty generalization).
- **Output:** verdict, evidence snippets, and any detected fallacy.

## Stack
Python, Hugging Face Transformers, BM25 retrieval, Gradio, deployed on Hugging Face Spaces.

## Status
Week 1: setup and data exploration.
