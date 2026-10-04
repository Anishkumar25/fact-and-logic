import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import gradio as gr
 
from src.pipeline import FactChecker
 
fc = FactChecker()
 
 
def run(claim):
    if not claim.strip():
        return "", "", "", ""
    r = fc.check(claim)
    return r["label"], f"{max(r['sup'], r['con']):.2f}", r["evidence"], r["source"]
 
 
gr.Interface(
    run,
    gr.Textbox(label="Scientific claim", lines=2),
    [gr.Textbox(label="Verdict"), gr.Textbox(label="Confidence"),
     gr.Textbox(label="Best evidence sentence", lines=3), gr.Textbox(label="Source paper")],
    title="Fact & Logic",
    description="Checks a scientific claim against a corpus of research abstracts (SciFact).",
    examples=[["0-dimensional biomaterials lack inductive properties."],
              ["1 in 5 million in UK have abnormal PrP positivity."]],
    flagging_mode="never",
).launch()
 