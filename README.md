# Henry Golding Adversarial Perturbation Generator

A specialized adversarial attack framework for analyzing deepfake detection models using the Henry Golding celebrity dataset. Leverages IBM's Adversarial Robustness Toolbox (ART) to generate comprehensive adversarial perturbations with detailed visual comparisons.

## Features

- **Multi-Attack Support**: PGD, FGSM, DeepFool, MI-FGSM implementations
- **Visual Analysis**: Automated side-by-side comparisons with perturbation heatmaps
- **Batch Processing**: Optimized for L4 GPU with configurable batch sizes
- **Comprehensive Metrics**: L2/L∞ norms, confidence drops, and success rates
- **Detailed Reporting**: JSON reports with per-image analysis

## Installation

```bash
pip install adversarial-robustness-toolbox[pytorch]
pip install torch torchvision pillow numpy matplotlib tqdm
```

## Quick Start

```python
from henry_golding_generator import HenryGoldingPerturbationGenerator
import torch.nn as nn

# Initialize your deepfake detection model
model = YourDeepfakeDetector()

# Create generator
generator = HenryGoldingPerturbationGenerator(
    model=model,
    dataset_path='./celeb-dataset/caucasian/henrygolding',
    input_shape=(3, 224, 224),
    nb_classes=2,
    device='cuda',
    batch_size=16
)

# Generate PGD attack with visual comparisons
results = generator.generate_perturbations_with_comparison(
    attack_type='pgd',
    save_dir='./henry_golding_pgd_results'
)

# Run complete multi-attack analysis
all_results = generator.run_complete_analysis(
    save_dir='./henry_golding_complete_analysis'
)
```

## Dataset Structure

```
./celeb-dataset/caucasian/henrygolding/
├── image001.jpg
├── image002.jpg
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Output Structure

```
results_directory/
├── attack_type_comparisons/
│   ├── comparison_001.png    # Individual comparisons
│   └── ...
├── attack_type_overview.png  # Grid overview
├── attack_type_detailed_report.json  # Metrics
└── attack_comparison_chart.png  # Multi-attack comparison
```

## Configuration

### Attack Parameters
```python
# Customize attack strength
generator.attacks['custom_pgd'] = ProjectedGradientDescent(
    estimator=generator.art_classifier,
    eps=16/255,      # Stronger perturbation budget
    max_iter=100     # More iterations
)
```

### Model Requirements
- Accept input shape `(batch, channels, height, width)`
- Output logits for binary classification `(batch, 2)`
- PyTorch compatible

## Metrics Tracked

### Per-Image
- Original and perturbed predictions
- Confidence scores and drops
- L2 and L∞ perturbation norms
- Attack success (prediction flip)

### Aggregate
- Overall attack success rate
- Average confidence degradation
- Perturbation magnitude distributions

## Performance

- **L4 GPU**: ~2-3 seconds per image for PGD
- **CPU fallback**: ~10-15 seconds per image
- **Batch processing**: Linear scaling with batch size

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
generator.batch_size = 8  # Reduce batch size
```

**Model Compatibility**
```python
def forward(self, x):
    if x.shape[-1] == 3:  # Convert NHWC to NCHW
        x = x.permute(0, 3, 1, 2)
    return self.model(x)  # Return raw logits
```

## Research Applications

- **Robustness evaluation** of deepfake detectors
- **Adversarial training** data generation
- **Model comparison** across architectures
- **Security assessment** for production systems

## Ethical Considerations

- Use only for research and defensive purposes
- Do not deploy against production systems without authorization
- Follow institutional guidelines for human subject research
- Consider privacy implications when analyzing celebrity datasets

## API Reference

### Main Methods
```python
# Core functionality
generate_perturbations_with_comparison(attack_type, max_images, save_dir)
run_complete_analysis(save_dir)
load_henry_golding_images(max_images)
```

### Attack Types
- `'fgsm'`: Fast Gradient Sign Method
- `'pgd'`: Projected Gradient Descent
- `'pgd_strong'`: Stronger PGD variant
- `'mi_fgsm'`: Momentum Iterative FGSM
- `'deepfool'`: Minimal perturbation attack

## Dependencies

```
adversarial-robustness-toolbox>=1.15.0
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.0.0
tqdm>=4.62.0
```

## License

MIT License - See LICENSE file for details

## Citation

```bibtex
@misc{henry_golding_perturbation_generator,
  title={Henry Golding Adversarial Perturbation Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```
