# Arc-Agi-Attempts

A PyTorch-based implementation for solving ARC (Abstraction and Reasoning Corpus) tasks using attention-based neural networks.

## Project Structure

```
.
├── model.py           # Neural network model implementation
├── ImgTrainer.py      # Training script
├── main.py           # Main execution script
├── reOrginize.py     # Data preprocessing utility
├── requirements.txt  # Project dependencies
└── models/           # Directory for saved models
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lunaDHD/Arc-Agi-Attempts.git
cd Arc-Agi-Attempts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python ImgTrainer.py
```

2. Run inference:
```bash
python main.py
```

## Model Architecture

The model uses a multi-scale attention mechanism to process input patterns:
- Multiple window sizes (2x2, 3x3, 4x4) for pattern extraction
- Attention layers for capturing relationships between patterns
- Encoder-decoder architecture for pattern transformation

## Data Format

The input data should be formatted as follows:
```python
[
    [  # list of samples
        [  # single sample
            [  # list of examples
                [],  # 2d array input for example
                []   # 2d array output for example
            ]
        ],
        [  # pair of input and correct output
            [],  # 2d array input for problem
            []   # 2d array correct output for problem
        ]
    ]
]
```

## License

MIT License 