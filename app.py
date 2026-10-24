"""Gradio demo (Hugging Face Spaces entry point).  python app.py"""
import os

try:  # present on Hugging Face Spaces; imported first, as ZeroGPU recommends
    import spaces
except ImportError:
    spaces = None

import gradio as gr

from src.pipeline import get_pipeline

if os.getenv("SPACE_ID"):  # on Spaces the GPU only exists inside @spaces.GPU calls, so stay on CPU
    os.environ.setdefault("DEVICE", "cpu")

if spaces:
    @spaces.GPU
    def _gpu_probe():
        """ZeroGPU refuses to start a Space with no @spaces.GPU function. Never called."""

PIPE = get_pipeline()
PIPE.warm()

EXAMPLES = [
    "0-dimensional biomaterials lack inductive properties.",
    "Vitamin D deficiency is associated with an increased risk of tuberculosis.",
    "Everyone I know got better after taking it, so it must work.",
    "You can't trust his research on diet, he's overweight himself.",
]
NAMES = {"SUPPORT": "Supported", "CONTRADICT": "Refuted", "NOT_ENOUGH_INFO": "Not enough info"}


def check(text: str):
    if not text or len(text.strip()) < 3:
        return "Enter a claim to check.", "", None
    r = PIPE.run(text.strip())
    v = r["verdict"]
    head = (f"## {NAMES[v['label']]}  ({v['confidence']:.0%})\n\n"
            f"Closest source: *{v['source']}* (SciFact doc {v['doc_id']})")
    fal = {f["label"]: f["prob"] for f in r["fallacies"]} or None
    if fal is None and r["fallacy_model"]:
        head += "\n\nNo logical fallacy flagged with enough confidence."
    return head, v["evidence"], fal


with gr.Blocks(title="Fact and Logic") as demo:
    gr.Markdown("# Fact and Logic\nChecks a claim against scientific abstracts and flags "
                "logical fallacies in how it is argued.\n\n*Evidence comes from the SciFact "
                "corpus (biomedical papers), so claims outside that area will mostly come back "
                "as 'Not enough info'.*")
    box = gr.Textbox(label="Claim or argument", lines=3)
    btn = gr.Button("Check", variant="primary")
    verdict = gr.Markdown()
    evidence = gr.Textbox(label="Best evidence sentence(s)", lines=4, interactive=False)
    fallacies = gr.Label(label="Possible fallacies", num_top_classes=3)
    gr.Examples(EXAMPLES, box)
    btn.click(check, box, [verdict, evidence, fallacies])
    box.submit(check, box, [verdict, evidence, fallacies])

if __name__ == "__main__":
    demo.launch()
