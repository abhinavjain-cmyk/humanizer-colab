# Humanize assignments on Google Colab

Open the notebook in Colab, pick a T4 GPU, Run all, upload your `.pdf` / `.docx`, download the rewritten PDF in the same layout.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhinavjain-cmyk/humanizer-colab/blob/main/Humanize_Assignments.ipynb)

- Model: [jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer) (Gemma 4 12B fine-tune, Apache 2.0), Q4_K_M GGUF, run with llama.cpp on the Colab GPU.
- Keeps facts, numbers, names; rebuilds A4, Times 12/16, 1 inch margins, front page, 400 to 450 words per answer.
- Nothing leaves the Colab runtime.

`hz_doc.py` holds the parse / trim / rebuild pipeline; `make_notebook.py` regenerates the notebook from it.
