#!/usr/bin/env python3
"""
Knowledge Distillation Training for ProteinMPNN
Train a small "student" model to mimic the pre-trained "teacher"

Architecture:
- Teacher: 3+3 layers, dim=128, k=48 (frozen, pre-trained)
- Student: 1+1 layers, dim=64, k=16 (trainable)

Expected: 10-15x speedup vs baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import sys
from pathlib import Path
import copy
from tqdm import tqdm

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps")

# Hyperparameters
ALPHA = 0.5  # Weight for ground truth loss
BETA = 0.5   # Weight for distillation loss
TEMPERATURE = 3.0  # Temperature for softening distributions
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 4

class DistillationTrainer:
    def __init__(self, teacher_path, student_config):
        """Initialize teacher and student models."""

        # Load frozen teacher (pre-trained)
        print("Loading teacher model (3+3 layers, dim=128, k=48)...")
        checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)

        self.teacher = ProteinMPNN(
            ca_only=False, num_letters=21,
            node_features=128, edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0, k_neighbors=48
        )
        self.teacher.load_state_dict(checkpoint['model_state_dict'])
        self.teacher.to(device)
        self.teacher.eval()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Create student (small model)
        print(f"Creating student model ({student_config['encoder']}+{student_config['decoder']} layers, "
              f"dim={student_config['dim']}, k={student_config['k']})...")

        self.student = ProteinMPNN(
            ca_only=False, num_letters=21,
            node_features=student_config['dim'],
            edge_features=student_config['dim'],
            hidden_dim=student_config['dim'],
            num_encoder_layers=student_config['encoder'],
            num_decoder_layers=student_config['decoder'],
            augment_eps=0.0,
            k_neighbors=student_config['k']
        )
        self.student.to(device)
        self.student.train()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=LEARNING_RATE)

        # Track metrics
        self.history = {
            'train_loss': [],
            'ce_loss': [],
            'kl_loss': [],
            'accuracy': []
        }

    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=3.0):
        """
        Compute distillation loss combining:
        1. Cross-entropy with ground truth (hard labels)
        2. KL divergence with teacher (soft labels)
        """

        # Hard loss: cross-entropy with ground truth
        ce_loss = F.cross_entropy(student_logits, labels, reduction='mean')

        # Soft loss: KL divergence with teacher's soft distributions
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

        # Combined loss
        total_loss = ALPHA * ce_loss + BETA * kl_loss

        return total_loss, ce_loss.item(), kl_loss.item()

    def train_step(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos):
        """Single training step."""

        self.student.train()
        self.optimizer.zero_grad()

        # Generate random decoding order
        randn = torch.randn(chain_M.shape, device=device)

        # Get teacher predictions (frozen, no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher(X, S, mask, chain_M*chain_M_pos,
                                         residue_idx, chain_encoding_all, randn)

        # Get student predictions
        student_logits = self.student(X, S, mask, chain_M*chain_M_pos,
                                     residue_idx, chain_encoding_all, randn)

        # Flatten for loss computation
        B, L, C = student_logits.shape
        student_logits_flat = student_logits.view(-1, C)
        teacher_logits_flat = teacher_logits.view(-1, C)
        labels_flat = S.view(-1)

        # Compute distillation loss
        loss, ce_loss, kl_loss = self.distillation_loss(
            student_logits_flat, teacher_logits_flat, labels_flat, TEMPERATURE
        )

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred = student_logits.argmax(dim=-1)
            accuracy = (pred == S).float().mean().item()

        return loss.item(), ce_loss, kl_loss, accuracy

    def train_epoch(self, train_proteins):
        """Train for one epoch."""

        epoch_losses = []
        epoch_ce_losses = []
        epoch_kl_losses = []
        epoch_accuracies = []

        pbar = tqdm(train_proteins, desc="Training")

        for protein_path in pbar:
            try:
                # Load protein
                pdb_dict_list = parse_PDB(protein_path, ca_only=False)
                protein = pdb_dict_list[0]

                # Create batch
                batch_clones = [copy.deepcopy(protein) for _ in range(BATCH_SIZE)]

                X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
                    batch_clones, device, None, None, None, None, None, None, ca_only=False
                )

                # Training step
                loss, ce_loss, kl_loss, accuracy = self.train_step(
                    X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos
                )

                epoch_losses.append(loss)
                epoch_ce_losses.append(ce_loss)
                epoch_kl_losses.append(kl_loss)
                epoch_accuracies.append(accuracy)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'ce': f'{ce_loss:.4f}',
                    'kl': f'{kl_loss:.4f}',
                    'acc': f'{accuracy:.3f}'
                })

            except Exception as e:
                print(f"\nError processing {protein_path}: {e}")
                continue

        return {
            'loss': np.mean(epoch_losses),
            'ce_loss': np.mean(epoch_ce_losses),
            'kl_loss': np.mean(epoch_kl_losses),
            'accuracy': np.mean(epoch_accuracies)
        }

    def evaluate(self, test_protein):
        """Evaluate student on a single protein."""

        self.student.eval()

        with torch.no_grad():
            pdb_dict_list = parse_PDB(test_protein, ca_only=False)
            protein = pdb_dict_list[0]

            X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
                [protein], device, None, None, None, None, None, None, ca_only=False
            )

            randn = torch.randn(chain_M.shape, device=device)

            # Time student
            torch.mps.synchronize()
            start = time.perf_counter()
            student_logits = self.student(X, S, mask, chain_M*chain_M_pos,
                                         residue_idx, chain_encoding_all, randn)
            torch.mps.synchronize()
            student_time = time.perf_counter() - start

            # Time teacher
            torch.mps.synchronize()
            start = time.perf_counter()
            teacher_logits = self.teacher(X, S, mask, chain_M*chain_M_pos,
                                         residue_idx, chain_encoding_all, randn)
            torch.mps.synchronize()
            teacher_time = time.perf_counter() - start

            # Accuracy
            student_pred = student_logits.argmax(dim=-1)
            teacher_pred = teacher_logits.argmax(dim=-1)

            student_accuracy = (student_pred == S).float().mean().item()
            teacher_accuracy = (teacher_pred == S).float().mean().item()
            agreement = (student_pred == teacher_pred).float().mean().item()

            seq_length = int(mask[0].sum().item())

            return {
                'student_time_ms': student_time * 1000,
                'teacher_time_ms': teacher_time * 1000,
                'speedup': teacher_time / student_time,
                'student_accuracy': student_accuracy,
                'teacher_accuracy': teacher_accuracy,
                'agreement': agreement,
                'seq_length': seq_length
            }

    def save_checkpoint(self, epoch, save_path):
        """Save student model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

def get_training_proteins(num_proteins=20):
    """Get list of training proteins from PDB monomers."""

    pdb_dir = Path('/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs')

    if not pdb_dir.exists():
        print(f"ERROR: PDB directory not found: {pdb_dir}")
        print("Please ensure ProteinMPNN repository is cloned and has example PDBs")
        return []

    pdb_files = list(pdb_dir.glob('*.pdb'))[:num_proteins]

    if len(pdb_files) == 0:
        print("ERROR: No PDB files found")
        return []

    print(f"Found {len(pdb_files)} training proteins")
    return pdb_files

def main():
    print("="*70)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("="*70)

    # Configuration
    teacher_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

    student_configs = {
        'ultra_small': {
            'name': 'Ultra-Small Student (1+1, dim=64, k=16)',
            'encoder': 1,
            'decoder': 1,
            'dim': 64,
            'k': 16
        },
        'small': {
            'name': 'Small Student (1+1, dim=96, k=16)',
            'encoder': 1,
            'decoder': 1,
            'dim': 96,
            'k': 16
        }
    }

    # Choose student config
    student_config = student_configs['ultra_small']

    print(f"\nTeacher: 3+3 layers, dim=128, k=48 (frozen)")
    print(f"Student: {student_config['encoder']}+{student_config['decoder']} layers, "
          f"dim={student_config['dim']}, k={student_config['k']} (training)")
    print(f"\nHyperparameters:")
    print(f"  Alpha (CE weight): {ALPHA}")
    print(f"  Beta (KL weight): {BETA}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # Get training data
    print("\n" + "-"*70)
    print("Loading training proteins...")
    train_proteins = get_training_proteins(num_proteins=20)

    if len(train_proteins) == 0:
        print("ERROR: No training data available")
        return

    test_protein = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'

    # Initialize trainer
    print("\n" + "-"*70)
    trainer = DistillationTrainer(teacher_path, student_config)

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-"*70)

        # Train
        metrics = trainer.train_epoch(train_proteins)

        # Log metrics
        trainer.history['train_loss'].append(metrics['loss'])
        trainer.history['ce_loss'].append(metrics['ce_loss'])
        trainer.history['kl_loss'].append(metrics['kl_loss'])
        trainer.history['accuracy'].append(metrics['accuracy'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  CE Loss: {metrics['ce_loss']:.4f}")
        print(f"  KL Loss: {metrics['kl_loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")

        # Evaluate
        print("\nEvaluating on test protein...")
        eval_metrics = trainer.evaluate(test_protein)

        print(f"  Student time: {eval_metrics['student_time_ms']:.2f} ms")
        print(f"  Teacher time: {eval_metrics['teacher_time_ms']:.2f} ms")
        print(f"  Speedup: {eval_metrics['speedup']:.2f}x")
        print(f"  Student accuracy: {eval_metrics['student_accuracy']:.3f}")
        print(f"  Teacher accuracy: {eval_metrics['teacher_accuracy']:.3f}")
        print(f"  Student-Teacher agreement: {eval_metrics['agreement']:.3f}")

        # Save checkpoint
        save_dir = Path('output/distillation')
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f'student_epoch_{epoch+1}.pt'
        trainer.save_checkpoint(epoch+1, checkpoint_path)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    final_metrics = trainer.evaluate(test_protein)

    print(f"\nStudent Model Performance:")
    print(f"  Architecture: {student_config['encoder']}+{student_config['decoder']} layers, "
          f"dim={student_config['dim']}, k={student_config['k']}")
    print(f"  Inference time: {final_metrics['student_time_ms']:.2f} ms")
    print(f"  Speedup vs teacher: {final_metrics['speedup']:.2f}x")
    print(f"  Accuracy: {final_metrics['student_accuracy']:.3f}")
    print(f"  Agreement with teacher: {final_metrics['agreement']:.3f}")

    # Save final model
    final_model_path = Path('output/distillation/student_final.pt')
    trainer.save_checkpoint(NUM_EPOCHS, final_model_path)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*70}")

    # Save training history
    history_path = Path('output/distillation/training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'config': student_config,
            'hyperparameters': {
                'alpha': ALPHA,
                'beta': BETA,
                'temperature': TEMPERATURE,
                'learning_rate': LEARNING_RATE,
                'epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE
            },
            'history': trainer.history,
            'final_metrics': final_metrics
        }, f, indent=2)

    print(f"\nTraining history saved to: {history_path}")

if __name__ == '__main__':
    main()
