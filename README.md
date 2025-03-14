# SuperNeuroABM
A GPU-based multi-agent simulation framework for neuromorphic computing 


# Requirements
For GPU mode:
- NVIDIA GPU with compute capability 6.0+ with CUDA toolkit and drivers
- Or: AMD GPU (tested on AMD MI250X on Frontier) with ROCm 5.7.1 drivers
- `pip install git+https://github.com/ORNL/SAGESim` 

# Unit Tests
To run unit tests, `cd` to root dir and run:

`python -m unittest tests.logic_gates_test`

# Publications

Arxiv Preprint available at: 

[Date. P., Gunaratne, C., Kulkarni, S., Patton, R., Coletti, M., & Potok, T. (2023). SuperNeuro: A Fast and Scalable Simulator for Neuromorphic Computing. arXiv preprint arXiv:2305.02510.](https://arxiv.org/abs/2109.12894)
