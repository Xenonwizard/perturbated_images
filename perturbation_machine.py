"""
IBM ART Batch Perturbation Generator for Henry Golding Dataset
Optimized for L4 GPU on Ubuntu with PyTorch
Generates adversarial perturbations and side-by-side comparisons
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# IBM ART imports
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    ProjectedGradientDescent,
    FastGradientMethod,
    DeepFool,
    BasicIterativeMethod,
    MomentumIterativeMethod,
    AutoAttack,
    SquareAttack,
    CarliniL2Method,
)


class HenryGoldingPerturbationGenerator:
    """
    Specialized perturbation generator for Henry Golding deepfake dataset.
    Creates side-by-side comparisons and comprehensive attack analysis.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 dataset_path: str = './celeb-dataset/caucasian/henrygolding',
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 nb_classes: int = 2,
                 device: str = 'cuda',
                 batch_size: int = 16):
        """
        Initialize the generator for Henry Golding dataset.
        
        Args:
            model: PyTorch deepfake detector model
            dataset_path: Path to the perturbation folder containing Henry Golding images
            input_shape: Shape of input images (C, H, W)
            nb_classes: Number of output classes (2 for binary deepfake detection)
            device: Device to use ('cuda' for L4 GPU)
            batch_size: Batch size for processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        
        # Image dimensions
        self.channels, self.height, self.width = input_shape
        
        # Create ART classifier wrapper
        self.art_classifier = PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=nb_classes,
            device_type='gpu' if torch.cuda.is_available() else 'cpu',
            clip_values=(0.0, 1.0),          # inputs are in [0, 1]
            channels_first=True            # because you keep images as HWC
        )
        
        # Initialize attack methods
        self.attacks = {}
        self._initialize_attacks()
        
        # Load Henry Golding images
        self.image_files = self._scan_dataset()
        
        print(f"✓ Initialized generator for Henry Golding dataset")
        print(f"✓ Found {len(self.image_files)} images")
        print(f"✓ Using device: {self.device}")
        print(f"✓ Input shape: {input_shape}")
        
    def _scan_dataset(self) -> List[str]:
        """Scan the dataset folder for Henry Golding images."""
        dataset_path = Path(self.dataset_path)
        image_files = []
        
        # Get all image files
        for file in dataset_path.glob('*'):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.append(str(file))
        
        # Sort for consistent ordering
        image_files.sort()
        
        print(f"\n📁 Dataset Analysis:")
        print(f"  Total Henry Golding images: {len(image_files)}")
        
        # Show first few files
        if image_files:
            print(f"  Sample files:")
            for f in image_files[:3]:
                print(f"    - {os.path.basename(f)}")
            if len(image_files) > 3:
                print(f"    ... and {len(image_files) - 3} more")
        
        return image_files
    
    def _initialize_attacks(self):
        """Initialize attack methods with optimized parameters for deepfake detection."""
        
        # PGD - Standard configuration
        self.attacks['pgd'] = ProjectedGradientDescent(
            estimator=self.art_classifier,
            eps=8/255,
            eps_step=2/255,
            max_iter=40,
            targeted=False,
            num_random_init=1,
            batch_size=self.batch_size
        )
        
        # PGD - Stronger variant for comparison
        self.attacks['pgd_strong'] = ProjectedGradientDescent(
            estimator=self.art_classifier,
            eps=16/255,
            eps_step=2/255,
            max_iter=100,
            targeted=False,
            num_random_init=5,
            batch_size=self.batch_size
        )
        
        # FGSM - Fast baseline
        self.attacks['fgsm'] = FastGradientMethod(
            estimator=self.art_classifier,
            eps=8/255,
            eps_step=8/255,
            targeted=False,
            batch_size=self.batch_size
        )
        
        # MI-FGSM - Momentum variant
        self.attacks['mi_fgsm'] = MomentumIterativeMethod(
            estimator=self.art_classifier,
            eps=8/255,
            eps_step=2/255,
            max_iter=40,
            decay=1.0,
            targeted=False,
            batch_size=self.batch_size
        )
        
        # DeepFool - Minimal perturbation
        self.attacks['deepfool'] = DeepFool(
            classifier=self.art_classifier,
            max_iter=50,
            epsilon=1e-6,
            batch_size=self.batch_size
        )
    
    def load_henry_golding_images(self, max_images: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load Henry Golding images from the dataset.
        
        Args:
            max_images: Maximum number of images to load (None for all)
            
        Returns:
            Tuple of (images array, file paths)
        """
        files_to_load = self.image_files[:max_images] if max_images else self.image_files
        
        images = []
        valid_files = []
        
        print(f"\n🖼️  Loading {len(files_to_load)} Henry Golding images...")
        
        for img_path in tqdm(files_to_load, desc="Loading images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.width, self.height), Image.LANCZOS)
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)
                valid_files.append(img_path)
            except Exception as e:
                print(f"  ⚠️  Failed to load {os.path.basename(img_path)}: {e}")
        
        if images:
            print(f"✓ Successfully loaded {len(images)} images")
            return np.stack(images), valid_files
        else:
            raise ValueError("No images could be loaded!")
    
    def generate_perturbations_with_comparison(self,
                                                attack_type: str = 'pgd',
                                                max_images: Optional[int] = None,
                                                save_dir: str = './henry_golding_results') -> Dict:
        """
        Generate perturbations and create side-by-side comparisons.
        
        Args:
            attack_type: Type of attack to use
            max_images: Maximum number of images to process
            save_dir: Directory to save results
            
        Returns:
            Dictionary with results and metrics
        """
        # Load images
        images, file_paths = self.load_henry_golding_images(max_images)
        
        print(f"\n{'='*60}")
        print(f"Generating {attack_type.upper()} perturbations for Henry Golding images")
        print(f"{'='*60}\n")
        
        # Get attack
        if attack_type not in self.attacks:
            raise ValueError(f"Attack {attack_type} not found. Available: {list(self.attacks.keys())}")
        
        attack = self.attacks[attack_type]
        
        # Generate adversarial examples
        print(f"⚔️  Launching {attack_type} attack on {len(images)} images...")
        perturbed_images = attack.generate(x=images)
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(images, perturbed_images, file_paths)
        
        # Create output directory structure
        os.makedirs(save_dir, exist_ok=True)
        comparison_dir = os.path.join(save_dir, f'{attack_type}_comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Generate side-by-side comparisons for each image
        print(f"\n🎨 Creating side-by-side comparisons...")
        for i in tqdm(range(len(images)), desc="Creating comparisons"):
            self._create_single_comparison(
                original=images[i],
                perturbed=perturbed_images[i],
                file_name=os.path.basename(file_paths[i]),
                save_path=os.path.join(comparison_dir, f'comparison_{i:03d}.png'),
                metrics=metrics['per_image'][i]
            )
        
        # Create overview grid
        self._create_overview_grid(
            images, perturbed_images, file_paths,
            save_path=os.path.join(save_dir, f'{attack_type}_overview.png'),
            attack_type=attack_type
        )
        
        # Save detailed report
        self._save_detailed_report(metrics, attack_type, save_dir)
        
        # Print summary
        self._print_summary(metrics, attack_type)
        
        return {
            'original_images': images,
            'perturbed_images': perturbed_images,
            'file_paths': file_paths,
            'metrics': metrics,
            'attack_type': attack_type,
            'save_dir': save_dir
        }
    
    def _create_single_comparison(self, original: np.ndarray, perturbed: np.ndarray,
                                  file_name: str, save_path: str, metrics: Dict):
        """Create a detailed side-by-side comparison for a single image."""
        
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(original)
        ax1.set_title('Original Image\n' + file_name, fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Perturbed image
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.imshow(perturbed)
        ax2.set_title(f'Perturbed Image\n(Flipped: {metrics["flipped"]})', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Perturbation (amplified)
        ax3 = fig.add_subplot(gs[0, 2])
        perturbation = perturbed - original
        perturbation_vis = np.clip(perturbation * 10 + 0.5, 0, 1)
        ax3.imshow(perturbation_vis)
        ax3.set_title('Perturbation (10x)', fontsize=11)
        ax3.axis('off')
        
        # Heatmap of perturbation magnitude
        ax4 = fig.add_subplot(gs[0, 3])
        magnitude = np.mean(np.abs(perturbation), axis=2)
        im = ax4.imshow(magnitude, cmap='hot', vmin=0, vmax=0.1)
        ax4.set_title('Perturbation Heatmap', fontsize=11)
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046)
        
        # Metrics text box
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        
        metrics_text = f"""
        Attack Metrics:
        ────────────────────────
        • Original Prediction: {metrics['orig_class']} (conf: {metrics['orig_conf']:.2%})
        • Perturbed Prediction: {metrics['pert_class']} (conf: {metrics['pert_conf']:.2%})
        • Prediction Changed: {'✓ YES' if metrics['flipped'] else '✗ NO'}
        • Confidence Drop: {metrics['conf_drop']:.2%}
        
        Perturbation Statistics:
        ────────────────────────
        • L2 Norm: {metrics['l2_norm']:.4f}
        • L∞ Norm: {metrics['linf_norm']:.4f}
        • Mean Absolute: {metrics['mean_abs']:.4f}
        • Max Pixel Change: {metrics['max_change']:.4f}
        """
        
        ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Adversarial Attack Analysis - {file_name}', fontsize=14, fontweight='bold')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_overview_grid(self, originals: np.ndarray, perturbed: np.ndarray,
                              file_paths: List[str], save_path: str, attack_type: str):
        """Create an overview grid showing multiple examples."""
        
        n_examples = min(12, len(originals))  # Show up to 12 examples
        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, n_rows * 8))
        axes = axes.flatten()
        
        for i in range(n_examples):
            # Original
            ax_orig = axes[2 * (i // n_cols) * n_cols + (i % n_cols)]
            ax_orig.imshow(originals[i])
            ax_orig.set_title(f'Original {i+1}', fontsize=10)
            ax_orig.axis('off')
            
            # Perturbed
            ax_pert = axes[(2 * (i // n_cols) + 1) * n_cols + (i % n_cols)]
            ax_pert.imshow(perturbed[i])
            
            # Check if prediction flipped
            orig_pred = self.art_classifier.predict(originals[i:i+1])
            pert_pred = self.art_classifier.predict(perturbed[i:i+1])
            flipped = np.argmax(orig_pred) != np.argmax(pert_pred)
            
            color = 'red' if flipped else 'green'
            ax_pert.set_title(f'Perturbed {i+1} {"(Flipped)" if flipped else "(Same)"}', 
                             fontsize=10, color=color)
            ax_pert.axis('off')
        
        # Hide unused axes
        for i in range(n_examples * 2, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{attack_type.upper()} Attack Overview - Henry Golding Dataset', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved overview grid to {save_path}")
    
    def _calculate_detailed_metrics(self, original: np.ndarray, perturbed: np.ndarray, 
                                   file_paths: List[str]) -> Dict:
        """Calculate detailed metrics for each image and overall."""
        
        # Get predictions
        orig_preds = self.art_classifier.predict(original)
        pert_preds = self.art_classifier.predict(perturbed)
        
        orig_classes = np.argmax(orig_preds, axis=1)
        pert_classes = np.argmax(pert_preds, axis=1)
        
        # Per-image metrics
        per_image_metrics = []
        for i in range(len(original)):
            perturbation = perturbed[i] - original[i]
            
            metrics = {
                'file': os.path.basename(file_paths[i]),
                'orig_class': ['Real', 'Fake'][orig_classes[i]],
                'pert_class': ['Real', 'Fake'][pert_classes[i]],
                'orig_conf': float(np.max(orig_preds[i])),
                'pert_conf': float(np.max(pert_preds[i])),
                'flipped': bool(orig_classes[i] != pert_classes[i]),
                'conf_drop': float(np.max(orig_preds[i]) - np.max(pert_preds[i])),
                'l2_norm': float(np.linalg.norm(perturbation)),
                'linf_norm': float(np.max(np.abs(perturbation))),
                'mean_abs': float(np.mean(np.abs(perturbation))),
                'max_change': float(np.max(np.abs(perturbation)))
            }
            per_image_metrics.append(metrics)
        
        # Overall metrics
        flipped_mask = orig_classes != pert_classes
        
        overall_metrics = {
            'total_images': len(original),
            'attack_success_rate': float(np.mean(flipped_mask)),
            'flipped_count': int(np.sum(flipped_mask)),
            'avg_conf_drop': float(np.mean([m['conf_drop'] for m in per_image_metrics])),
            'avg_l2_norm': float(np.mean([m['l2_norm'] for m in per_image_metrics])),
            'avg_linf_norm': float(np.mean([m['linf_norm'] for m in per_image_metrics])),
            'max_linf_norm': float(np.max([m['linf_norm'] for m in per_image_metrics])),
            'per_image': per_image_metrics
        }
        
        return overall_metrics
    
    def _save_detailed_report(self, metrics: Dict, attack_type: str, save_dir: str):
        """Save a detailed JSON report of the attack results."""
        
        report = {
            'attack_type': attack_type,
            'dataset': 'Henry Golding Images',
            'summary': {
                'total_images': metrics['total_images'],
                'attack_success_rate': metrics['attack_success_rate'],
                'flipped_predictions': metrics['flipped_count'],
                'average_confidence_drop': metrics['avg_conf_drop'],
                'average_l2_norm': metrics['avg_l2_norm'],
                'average_linf_norm': metrics['avg_linf_norm']
            },
            'per_image_results': metrics['per_image']
        }
        
        report_path = os.path.join(save_dir, f'{attack_type}_detailed_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Saved detailed report to {report_path}")
    
    def _print_summary(self, metrics: Dict, attack_type: str):
        """Print a formatted summary of attack results."""
        
        print(f"\n{'='*60}")
        print(f"{attack_type.upper()} ATTACK RESULTS - HENRY GOLDING DATASET")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total Images: {metrics['total_images']}")
        print(f"  • Attack Success Rate: {metrics['attack_success_rate']:.1%}")
        print(f"  • Flipped Predictions: {metrics['flipped_count']}/{metrics['total_images']}")
        print(f"  • Average Confidence Drop: {metrics['avg_conf_drop']:.3f}")
        
        print(f"\n📏 Perturbation Metrics:")
        print(f"  • Average L2 Norm: {metrics['avg_l2_norm']:.4f}")
        print(f"  • Average L∞ Norm: {metrics['avg_linf_norm']:.4f}")
        print(f"  • Maximum L∞ Norm: {metrics['max_linf_norm']:.4f}")
        
        # Show which images were most/least affected
        sorted_by_conf = sorted(metrics['per_image'], key=lambda x: x['conf_drop'], reverse=True)
        
        print(f"\n🎯 Most Affected Images (by confidence drop):")
        for i, img in enumerate(sorted_by_conf[:3]):
            print(f"  {i+1}. {img['file']}: {img['conf_drop']:.3f} drop")
        
        print(f"\n🛡️ Most Robust Images (least affected):")
        for i, img in enumerate(sorted_by_conf[-3:]):
            print(f"  {i+1}. {img['file']}: {img['conf_drop']:.3f} drop")
    
    def run_complete_analysis(self, save_dir: str = './henry_golding_analysis'):
        """
        Run a complete analysis with multiple attack types on Henry Golding images.
        
        Args:
            save_dir: Directory to save all results
        """
        attack_types = ['fgsm', 'pgd', 'pgd_strong', 'mi_fgsm', 'deepfool']
        all_results = {}
        
        print(f"\n{'#'*60}")
        print(f"COMPLETE ADVERSARIAL ANALYSIS - HENRY GOLDING DATASET")
        print(f"Running {len(attack_types)} different attacks")
        print(f"{'#'*60}\n")
        
        for attack_type in attack_types:
            attack_dir = os.path.join(save_dir, attack_type)
            results = self.generate_perturbations_with_comparison(
                attack_type=attack_type,
                save_dir=attack_dir
            )
            all_results[attack_type] = results
        
        # Create comparison chart
        self._create_attack_comparison_chart(all_results, save_dir)
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 Results saved to: {save_dir}")
        
        return all_results
    
    def _create_attack_comparison_chart(self, all_results: Dict, save_dir: str):
        """Create a comparison chart of all attacks."""
        
        attacks = list(all_results.keys())
        success_rates = [all_results[a]['metrics']['attack_success_rate'] for a in attacks]
        avg_l2 = [all_results[a]['metrics']['avg_l2_norm'] for a in attacks]
        avg_linf = [all_results[a]['metrics']['avg_linf_norm'] for a in attacks]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Success rates
        axes[0].bar(attacks, [r * 100 for r in success_rates], color='steelblue')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Attack Success Rates')
        axes[0].set_ylim(0, 100)
        for i, v in enumerate(success_rates):
            axes[0].text(i, v * 100 + 2, f'{v:.1%}', ha='center')
        
        # L2 norms
        axes[1].bar(attacks, avg_l2, color='forestgreen')
        axes[1].set_ylabel('Average L2 Norm')
        axes[1].set_title('Perturbation Magnitude (L2)')
        
        # L∞ norms
        axes[2].bar(attacks, avg_linf, color='coral')
        axes[2].set_ylabel('Average L∞ Norm')
        axes[2].set_title('Perturbation Magnitude (L∞)')
        
        plt.suptitle('Attack Comparison - Henry Golding Dataset', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        chart_path = os.path.join(save_dir, 'attack_comparison_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comparison chart to {chart_path}")


def main():
    """
    Main function to run the Henry Golding dataset analysis.
    """
    
    # Example: Create a dummy deepfake detector model
    # Replace this with your actual model
    class DeepfakeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(256, 2)  # Binary: real/fake
        
        def forward(self, x):
            # If input is (B, H, W, C), reorder to (B, C, H, W)
            if x.ndim == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)   # NHWC -> NCHW
            
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    
    # Initialize your model
    model = DeepfakeDetector()  # Replace with your actual model
    
    # Initialize the generator
    generator = HenryGoldingPerturbationGenerator(
        model=model,
        dataset_path='./celeb-dataset/caucasian/henrygolding',  # Path to your Henry Golding images
        input_shape=(3, 224, 224),
        nb_classes=2,
        device='cuda',
        batch_size=16
    )
    
    # Run single attack with side-by-side comparisons
    print("\n" + "="*60)
    print("Generating PGD attack with side-by-side comparisons")
    print("="*60)
    
    results = generator.generate_perturbations_with_comparison(
        attack_type='pgd',
        save_dir='./henry_golding_pgd_results'
    )
    
    # Or run complete analysis with all attacks
    # print("\n" + "="*60)
    # print("Running complete analysis with all attacks")
    # print("="*60)
    # 
    # all_results = generator.run_complete_analysis(
    #     save_dir='./henry_golding_complete_analysis'
    # )
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    # Install requirements:
    # pip install adversarial-robustness-toolbox[pytorch] torch torchvision pillow numpy matplotlib tqdm
    
    main()