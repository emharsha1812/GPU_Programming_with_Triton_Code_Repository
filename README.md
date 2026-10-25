# GPU Programming with Triton: code examples

Each chapter notebook includes its implementations. Open a notebook in `code/`
and run its cells in order.

| Chapter | Notebook |
|---|---|
| 2 | [Chapter 2](code/CH02_Fartale_GPU_Programming_with_Triton.ipynb) |
| 3 | [Chapter 3](code/CH03_Fartale_GPU_Programming_with_Triton.ipynb) |
| 4 | [Chapter 4](code/CH04_Fartale_GPU_Programming_with_Triton.ipynb) |
| 5 | [Chapter 5](code/CH05_Fartale_GPU_Programming_with_Triton.ipynb) |
| 6 | [Advanced matrix multiplication](code/CH06_Fartale_GPU_Programming_with_Triton.ipynb) |
| 8 | [Chapter 8](code/CH08_Fartale_GPU_Programming_with_Triton.ipynb) |

## Set up your environment

Open the notebooks in [Google Colab](https://research.google.com/colaboratory/faq.html)
with a suitable GPU runtime, or use Jupyter connected to a GPU machine. You can
access a hosted notebook from Windows, macOS, or Linux without installing Linux
on your own computer.

The machine executing the GPU cells needs a supported NVIDIA GPU, CUDA-enabled
PyTorch, and a compatible Triton build. The standard
[Triton runtime supports Linux](https://github.com/triton-lang/triton#compatibility);
a hosted or remote machine can provide that environment. Hardware and API
requirements vary by example, so check the notebook and the selected runtime.

For a local or self-managed runtime, follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/)
and [Triton installation instructions](https://triton-lang.org/main/getting-started/installation.html).
Install the notebook tools and supplementary libraries in that runtime as needed:

```sh
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab
```

Install PyTorch and Triton separately for your hardware. Chapter 5 uses NumPy,
Numba, and Matplotlib. Chapter 8 uses Hugging Face `datasets` and downloads data
when its dataset-loading cells run.

## Third-party notices

Retain [TRITON_LICENSE.txt](TRITON_LICENSE.txt) when distributing the adapted
code. See [Third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
