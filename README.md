
<p align="center">
  <h1 align="center">VoiceTrace: A Benchmark and Unified Framework for Who-Said-What Speech Retrieval</h1>
  <p align="center">
    📄 <b><a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a></b>
     | 
    🌐 <b><a href="#">Project Page</a></b>
     | 
    🤗 <b><a href="#">Benchmark</a></b>
  </p>
</p>

---
## Quick Start

To evaluate the baseline, first clone this repository:

```bash
git clone git@github.com:HumanifyAI/VoxRetrieval.git
cd VoxRetrieval
``` 

Then, download the evaluation dataset from Hugging Face:

```bash
hf download xxxx --local-dir benchmark_data
```

Then prepare the environment and run baseline evaluation:

```bash
conda create -n eval python=3.12
conda activate eval
pip install -r requirements.txt
python evaluate_baseline.py --single-dataset-path benchmark_data/benchmark_single --conv-dataset-path benchmark_data/benchmark_conv
```

## Citation
If you find this work useful, please consider contributing to this repo and cite this work:

```
@article{Yee2026voicetrace,
  title={VoiceTrace: A Benchmark and Unified Framework for Who-Said-What Speech Retrieval},
  author={Aaron Yee and Fengjie Lu and Jiarui Hai and Chenang Jiang and Helin Wang and Siwei Tu and Lingyun Sun},
  journal={arXiv preprint arXiv:TBD},
  year={2026}
}
```

## License
The Repository is licensed under CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).

## Acknowledgements

- [VoxCeleb / VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) for providing large-scale benchmark datasets for speaker-related research.
- [VoxConverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/) for providing multi-speaker conversational recordings with diarization annotations.
- [Seamless Interaction](https://github.com/facebookresearch/seamless_interaction?tab=readme-ov-file) for open-source resources that support speech and multimodal interaction research.

## 🌟 Like This Project?
If you find this repo helpful or interesting, consider dropping a ⭐ — it really helps and means a lot!