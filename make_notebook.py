"""Generate Humanize_Assignments.ipynb (embeds hz_doc.py)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HZ_DOC = open(os.path.join(HERE, "hz_doc.py"), encoding="utf8").read()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src, form=False):
    c = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
    if form:
        c["metadata"]["cellView"] = "form"
    return c


cells = [
    md(
        "# Humanize assignments (PDF / DOCX in, same-layout PDF out)\n"
        "\n"
        "Rewrites every answer of an assignment with the open-weight **jialinyyzz/humanizer** model "
        "(Gemma 4 12B fine-tune, Apache 2.0) running on this Colab GPU, keeps facts, numbers and names, "
        "then rebuilds the PDF in the same layout: A4, Times 12 pt on 16 pt lines, 1 inch margins, "
        "front page, SET headings, bold questions, justified paragraphs, 400 to 450 words per answer.\n"
        "\n"
        "**Steps**\n"
        "1. Runtime, Change runtime type, **T4 GPU** (free). Then Runtime, Run all.\n"
        "2. Cell 5 asks for files: pick one or many `.pdf` / `.docx` assignments.\n"
        "3. Wait. About 4 to 6 minutes per document on a T4.\n"
        "4. The last cell downloads a zip with one PDF and one paste-ready `.txt` per document.\n"
        "\n"
        "Nothing leaves this Colab runtime. Read the output before submitting, especially numbers and names."
    ),
    code(
        "#@title 1. Settings { display-mode: \"form\" }\n"
        "MODEL_FILE = \"humanizer-12b-Q4_K_M.gguf\"  #@param [\"humanizer-12b-Q4_K_M.gguf\", \"humanizer-12b-Q6_K.gguf\", \"humanizer-12b-Q8_0.gguf\"]\n"
        "MIN_WORDS = 400  #@param {type:\"integer\"}\n"
        "MAX_WORDS = 450  #@param {type:\"integer\"}\n"
        "SAMPLES = 1  #@param {type:\"slider\", min:1, max:3, step:1}\n"
        "#@markdown Q4_K_M fits the free T4 (16 GB). Q6_K and Q8_0 need an L4/A100 runtime. More samples = better word-band fit, slower.\n"
        "import subprocess\n"
        "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout or 'NO GPU: Runtime > Change runtime type > T4 GPU')\n",
        form=True,
    ),
    code(
        "#@title 2. Install (prebuilt GPU wheel, about 1 minute)\n"
        "import sys, subprocess, time\n"
        "t0 = time.time()\n"
        "def have():\n"
        "    return subprocess.run([sys.executable, '-c', 'import llama_cpp'], capture_output=True).returncode == 0\n"
        "ok = have()\n"
        "WHEELS = 'https://github.com/abhinavjain-cmyk/humanizer-colab/releases/expanded_assets/wheels'  # prebuilt for Colab T4, built once in Colab\n"
        "if not ok:\n"
        "    print('installing prebuilt Colab wheel from GitHub ...')\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:', '--no-index', '--find-links', WHEELS, 'llama-cpp-python'], capture_output=True, text=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'diskcache', 'jinja2', 'typing-extensions'], capture_output=True)\n"
        "    ok = have()\n"
        "for cu in ('cu124', 'cu125', 'cu123', 'cu122', 'cu121'):\n"
        "    if ok: break\n"
        "    print(f'trying prebuilt wheel index {cu} ...')\n"
        "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:', 'llama-cpp-python',\n"
        "                        '--extra-index-url', f'https://abetlen.github.io/llama-cpp-python/whl/{cu}'], capture_output=True, text=True)\n"
        "    ok = have()\n"
        "if not ok:\n"
        "    print('no prebuilt wheel for this runtime, compiling (10-15 min)...')\n"
        "    import os\n"
        "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'llama-cpp-python==0.3.19', '--no-binary', 'llama-cpp-python'],\n"
        "                       env={**os.environ, 'CMAKE_ARGS': '-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75;86;89', 'FORCE_CMAKE': '1'}, capture_output=True, text=True)\n"
        "    print(r.stderr[-2000:])\n"
        "    ok = have()\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pdfplumber', 'reportlab', 'python-docx', 'huggingface_hub', 'pymupdf'], capture_output=True)\n"
        "import llama_cpp\n"
        "print(f'llama-cpp-python {llama_cpp.__version__} ready in {time.time()-t0:.0f}s, GPU offload: {llama_cpp.llama_supports_gpu_offload()}')\n"
    ),
    code(
        "#@title 3. Pipeline code (parse, trim, rebuild)\n"
        "code = r'''" + HZ_DOC.replace("'''", "\\'\\'\\'") + "'''\n"
        "open('hz_doc.py', 'w').write(code)\n"
        "import importlib, hz_doc\n"
        "importlib.reload(hz_doc)\n"
        "print('hz_doc ready')\n"
    ),
    code(
        "#@title 4. Download and load the model (first run: 2 to 4 minutes)\n"
        "import os, time, hashlib, json\n"
        "from huggingface_hub import hf_hub_download\n"
        "os.makedirs('/content/models', exist_ok=True)\n"
        "t0 = time.time()\n"
        "MODEL_PATH = hf_hub_download('jialinyyzz/humanizer-GGUF', MODEL_FILE, local_dir='/content/models')\n"
        "print(f'model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e9:.1f} GB) in {time.time()-t0:.0f}s')\n"
        "\n"
        "INSTR = ('Rewrite the text below so it reads like a person wrote it, not a language model.\\n\\n'\n"
        "         'Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\\n'\n"
        "         'throat-clearing, and any sentence that only announces what comes next.\\n'\n"
        "         'Prefer the concrete word over the abstract one. It is fine to sound uneven.\\n\\n'\n"
        "         'Every fact, number, unit, date, name and quotation must survive unchanged.')\n"
        "SEP = '\\n\\n### Rewritten:\\n\\n'\n"
        "assert hashlib.sha256((INSTR + '\\n\\n' + 'X' + SEP).encode()).hexdigest()[:16] == 'cc51d66b4c593fbe', 'prompt drifted from the model card'\n"
        "\n"
        "from llama_cpp import Llama\n"
        "t0 = time.time()\n"
        "llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_gpu_layers=-1, n_batch=512, flash_attn=True, verbose=False)\n"
        "print(f'loaded on GPU in {time.time()-t0:.0f}s')\n"
        "\n"
        "import re\n"
        "def _chunks(text, max_words=450):\n"
        "    paras = [p.strip() for p in re.split(r'\\n\\s*\\n', text.strip()) if p.strip()]\n"
        "    out, cur, n = [], [], 0\n"
        "    for p in paras:\n"
        "        w = len(p.split())\n"
        "        if cur and n + w > max_words:\n"
        "            out.append('\\n\\n'.join(cur)); cur, n = [], 0\n"
        "        cur.append(p); n += w\n"
        "    if cur: out.append('\\n\\n'.join(cur))\n"
        "    return out\n"
        "\n"
        "def rewrite(text):\n"
        "    \"\"\"Model card prompt and sampling, byte for byte. Long answers go in 450-word chunks.\"\"\"\n"
        "    outs = []\n"
        "    for ch in _chunks(text):\n"
        "        r = llm(INSTR + '\\n\\n' + ch + SEP, max_tokens=2048, temperature=1.0, top_p=0.95, top_k=0, min_p=0.0, repeat_penalty=1.0, stop=[])\n"
        "        outs.append(r['choices'][0]['text'].strip())\n"
        "    return '\\n\\n'.join(outs)\n"
        "\n"
        "t0 = time.time(); _ = rewrite('The committee met on 12 March and approved a budget of Rs 4,500 for the event. Three members voted against it. The decision was recorded in the minutes and shared with all departments the next day.')\n"
        "print(f'warm-up rewrite ok in {time.time()-t0:.0f}s:\\n{_[:300]}')\n"
    ),
    code(
        "#@title 5. Upload assignments (.pdf or .docx, several at once)\n"
        "from google.colab import files\n"
        "import shutil, os\n"
        "os.makedirs('/content/in', exist_ok=True); os.makedirs('/content/out', exist_ok=True)\n"
        "up = files.upload()\n"
        "INPUTS = []\n"
        "for name in up:\n"
        "    dst = f'/content/in/{name}'\n"
        "    shutil.move(name, dst)\n"
        "    INPUTS.append(dst)\n"
        "print(len(INPUTS), 'file(s):', *[os.path.basename(p) for p in INPUTS], sep='\\n  ')\n"
    ),
    code(
        "#@title 6. Humanize every document and rebuild the PDFs\n"
        "import time, os, json, zipfile\n"
        "import hz_doc\n"
        "results = []\n"
        "T0 = time.time()\n"
        "for path in INPUTS:\n"
        "    name = os.path.splitext(os.path.basename(path))[0]\n"
        "    print(f'\\n=== {name}')\n"
        "    t0 = time.time()\n"
        "    try:\n"
        "        doc = hz_doc.parse_document(path)\n"
        "    except Exception as e:\n"
        "        print('  cannot read:', e); continue\n"
        "    nq = sum(1 for b in doc['blocks'] if b['kind'] == 'q')\n"
        "    print(f\"  front page: {len(doc['front'])} lines, questions: {nq}, blocks: {len(doc['blocks'])}\")\n"
        "    if nq == 0:\n"
        "        print('  no numbered questions found; treating the whole body as one answer')\n"
        "        doc['blocks'] = [dict(kind='q', text=' ')] + doc['blocks']\n"
        "    new, report = hz_doc.humanize_blocks(doc, rewrite, MIN_WORDS, MAX_WORDS, SAMPLES, log=print)\n"
        "    pdf = f'/content/out/humanized_{name}.pdf'\n"
        "    hz_doc.build_pdf(new, pdf)\n"
        "    txt = f'/content/out/humanized_{name}.txt'\n"
        "    open(txt, 'w', encoding='utf8').write(hz_doc.doc_to_text(new))\n"
        "    import fitz\n"
        "    pages = len(fitz.open(pdf))\n"
        "    print(f'  -> {pdf}  ({pages} pages, {time.time()-t0:.0f}s)')\n"
        "    results.append(dict(file=name, pdf=pdf, txt=txt, pages=pages, answers=report, seconds=round(time.time()-t0)))\n"
        "\n"
        "print(f'\\nall done in {(time.time()-T0)/60:.1f} min')\n"
        "for r in results:\n"
        "    print(f\"\\n{r['file']}: {r['pages']} pages\")\n"
        "    for a in r['answers']:\n"
        "        print(f\"   {a['words_in']:4} -> {a['words_out']:4} words {a['flag'] or 'ok':6} numbers {'kept' if a['numbers_ok'] else 'CHECK'}   {a['question'][:60]}\")\n"
        "zip_path = '/content/out/humanized.zip'\n"
        "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:\n"
        "    for r in results:\n"
        "        z.write(r['pdf'], os.path.basename(r['pdf'])); z.write(r['txt'], os.path.basename(r['txt']))\n"
        "print('\\nzip:', zip_path)\n"
    ),
    code(
        "#@title 7. Download\n"
        "from google.colab import files\n"
        "files.download('/content/out/humanized.zip')\n"
        "for r in results:\n"
        "    files.download(r['pdf'])\n"
    ),
    md(
        "**Notes**\n"
        "- Word band: answers over the maximum are cut by whole sentences, least specific first (never the first sentence of a paragraph, never a sentence with a number or a name). Answers under the minimum are flagged SHORT; raise SAMPLES to 2 or 3 and rerun cell 6 for those.\n"
        "- `numbers CHECK` means a figure from the original did not survive. Open that answer and compare.\n"
        "- Input layout: the PDF parser expects bold questions numbered `1.` to `6.`, bold sub-headings, `SET 1` / `SET 2` lines and a details front page, which is the MUJ template. A `.docx` works the same way if questions and headings are bold.\n"
        "- To rerun on new files: run cell 5 then cell 6 again. The model stays loaded.\n"
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Humanize_Assignments.ipynb", "provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(HERE, "Humanize_Assignments.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(out, len(cells), "cells")
