#!/usr/bin/env python3
"""
ProteinMPNN Benchmarking Suite

A comprehensive benchmarking tool for evaluating ProteinMPNN model variants.
Measures timing, recovery rates, burial analysis, and more.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from Bio.PDB import PDBParser, SASA
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Burial analysis will be disabled.")


class ProteinMPNNBenchmark:
    """Benchmarking suite for ProteinMPNN model variants."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        output_dir: str = "./output"
    ):
        """
        Initialize the benchmark suite.

        Args:
            model_path: Path to model weights (downloads if None)
            device: Device to use ('cuda', 'cpu', or 'auto')
            output_dir: Directory for output files
        """
        self.device = self._setup_device(device)
        self.model_path = model_path or self._download_model_weights()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _download_model_weights(self) -> str:
        """Download ProteinMPNN model weights if not present."""
        weights_dir = Path("./models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "proteinmpnn_weights.pt"

        if weights_path.exists():
            print(f"Using existing model weights: {weights_path}")
            return str(weights_path)

        print("Downloading ProteinMPNN model weights...")
        url = "https://files.ipd.uw.edu/pub/ProteinMPNN/model_weights/v_48_020.pt"

        try:
            urllib.request.urlretrieve(url, weights_path)
            print(f"Downloaded model weights to: {weights_path}")
            return str(weights_path)
        except Exception as e:
            print(f"Warning: Could not download weights: {e}")
            print("Please manually download weights and specify with --model_path")
            return ""

    def download_example_pdbs(self, data_dir: str, num_examples: int = 5) -> List[str]:
        """
        Download example PDB files for benchmarking.

        Args:
            data_dir: Directory to save PDB files
            num_examples: Number of example structures to download

        Returns:
            List of paths to downloaded PDB files
        """
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Example PDB IDs (diverse set of proteins)
        pdb_ids = ["1UBQ", "2LYZ", "1CRN", "1VII", "1MB1"][:num_examples]
        downloaded_files = []

        print(f"Downloading {len(pdb_ids)} example PDB files...")
        for pdb_id in tqdm(pdb_ids):
            pdb_file = data_path / f"{pdb_id}.pdb"

            if pdb_file.exists():
                downloaded_files.append(str(pdb_file))
                continue

            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                urllib.request.urlretrieve(url, pdb_file)
                downloaded_files.append(str(pdb_file))
            except Exception as e:
                print(f"Warning: Could not download {pdb_id}: {e}")

        return downloaded_files

    def load_structure(self, pdb_path: str) -> Dict:
        """
        Load a PDB structure.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary with structure information
        """
        # Placeholder: Replace with actual ProteinMPNN structure loading
        # This would normally parse the PDB and extract coordinates, sequence, etc.

        if not BIOPYTHON_AVAILABLE:
            return {
                "path": pdb_path,
                "name": Path(pdb_path).stem,
                "sequence": "",
                "coordinates": None
            }

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        # Extract sequence
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard amino acid
                        sequence += self._three_to_one(residue.resname)

        return {
            "path": pdb_path,
            "name": Path(pdb_path).stem,
            "sequence": sequence,
            "structure": structure,
            "coordinates": None  # Would extract CA coordinates here
        }

    @staticmethod
    def _three_to_one(three_letter: str) -> str:
        """Convert three-letter amino acid code to one-letter."""
        conversion = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return conversion.get(three_letter, 'X')

    def calculate_sasa(self, structure) -> np.ndarray:
        """
        Calculate solvent accessible surface area for residues.

        Args:
            structure: BioPython structure object

        Returns:
            Array of SASA values per residue
        """
        if not BIOPYTHON_AVAILABLE:
            return np.array([])

        # Placeholder: Would use BioPython SASA calculation
        # This is simplified - full implementation would use SASA.ShrakeRupley
        sasa_calculator = SASA.ShrakeRupley()
        sasa_calculator.compute(structure, level="R")

        sasa_values = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        sasa_values.append(residue.sasa if hasattr(residue, 'sasa') else 0.0)

        return np.array(sasa_values)

    def predict_sequences(
        self,
        structure: Dict,
        num_samples: int = 5,
        temperature: float = 0.1
    ) -> List[str]:
        """
        Generate sequences using ProteinMPNN.

        Args:
            structure: Structure dictionary
            num_samples: Number of sequences to generate
            temperature: Sampling temperature

        Returns:
            List of designed sequences
        """
        # Placeholder: Replace with actual ProteinMPNN inference
        # This is where you would call your ProteinMPNN model

        print(f"Generating {num_samples} sequences for {structure['name']}...")

        # Simulate inference time
        time.sleep(0.5)

        # Mock sequences for demonstration
        # In reality, this would call the actual model
        native_seq = structure.get('sequence', 'A' * 50)
        sequences = []

        for i in range(num_samples):
            # Generate a sequence with some random mutations
            seq_list = list(native_seq)
            num_mutations = int(len(seq_list) * 0.3)  # 30% mutations
            positions = np.random.choice(len(seq_list), num_mutations, replace=False)
            amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

            for pos in positions:
                seq_list[pos] = np.random.choice(amino_acids)

            sequences.append(''.join(seq_list))

        return sequences

    def calculate_recovery(
        self,
        native_seq: str,
        designed_seqs: List[str]
    ) -> Dict[str, float]:
        """
        Calculate sequence recovery metrics.

        Args:
            native_seq: Native sequence
            designed_seqs: List of designed sequences

        Returns:
            Dictionary with recovery metrics
        """
        if not native_seq or not designed_seqs:
            return {"mean_recovery": 0.0, "std_recovery": 0.0}

        recoveries = []
        for designed_seq in designed_seqs:
            if len(designed_seq) != len(native_seq):
                continue

            matches = sum(n == d for n, d in zip(native_seq, designed_seq))
            recovery = (matches / len(native_seq)) * 100
            recoveries.append(recovery)

        return {
            "mean_recovery": float(np.mean(recoveries)),
            "std_recovery": float(np.std(recoveries)),
            "min_recovery": float(np.min(recoveries)),
            "max_recovery": float(np.max(recoveries)),
            "num_samples": len(recoveries)
        }

    def burial_analysis(
        self,
        native_seq: str,
        designed_seqs: List[str],
        sasa_values: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze recovery rates for buried vs exposed residues.

        Args:
            native_seq: Native sequence
            designed_seqs: List of designed sequences
            sasa_values: SASA values per residue

        Returns:
            Dictionary with burial-specific recovery rates
        """
        if len(sasa_values) == 0 or len(sasa_values) != len(native_seq):
            return {
                "buried": {"recovery": 0.0, "count": 0},
                "exposed": {"recovery": 0.0, "count": 0},
                "intermediate": {"recovery": 0.0, "count": 0}
            }

        # Normalize SASA (typical max is ~200 Ų for exposed residues)
        max_sasa = 200.0
        normalized_sasa = sasa_values / max_sasa

        # Categorize residues
        buried_mask = normalized_sasa < 0.05
        exposed_mask = normalized_sasa > 0.20
        intermediate_mask = ~(buried_mask | exposed_mask)

        results = {}
        for category, mask in [
            ("buried", buried_mask),
            ("exposed", exposed_mask),
            ("intermediate", intermediate_mask)
        ]:
            if not np.any(mask):
                results[category] = {"recovery": 0.0, "count": 0}
                continue

            recoveries = []
            for designed_seq in designed_seqs:
                if len(designed_seq) != len(native_seq):
                    continue

                matches = sum(
                    n == d and mask[i]
                    for i, (n, d) in enumerate(zip(native_seq, designed_seq))
                )
                total = np.sum(mask)
                if total > 0:
                    recoveries.append((matches / total) * 100)

            results[category] = {
                "recovery": float(np.mean(recoveries)) if recoveries else 0.0,
                "std": float(np.std(recoveries)) if recoveries else 0.0,
                "count": int(np.sum(mask))
            }

        return results

    def benchmark_structure(
        self,
        pdb_path: str,
        num_samples: int = 5
    ) -> Dict:
        """
        Benchmark a single structure.

        Args:
            pdb_path: Path to PDB file
            num_samples: Number of sequences to generate

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {Path(pdb_path).name}")
        print(f"{'='*60}")

        # Load structure
        structure = self.load_structure(pdb_path)
        native_seq = structure.get("sequence", "")

        if not native_seq:
            print("Warning: Could not extract sequence from structure")
            return {"error": "No sequence found"}

        print(f"Sequence length: {len(native_seq)}")

        # Calculate SASA for burial analysis
        sasa_values = np.array([])
        if BIOPYTHON_AVAILABLE and "structure" in structure:
            sasa_values = self.calculate_sasa(structure["structure"])

        # Time the inference
        start_time = time.time()
        designed_seqs = self.predict_sequences(structure, num_samples)
        inference_time = time.time() - start_time

        # Calculate metrics
        recovery_metrics = self.calculate_recovery(native_seq, designed_seqs)
        burial_metrics = self.burial_analysis(native_seq, designed_seqs, sasa_values)

        # Save sequences
        output_file = self.output_dir / f"{structure['name']}_sequences.fasta"
        with open(output_file, 'w') as f:
            f.write(f">native\n{native_seq}\n")
            for i, seq in enumerate(designed_seqs):
                f.write(f">designed_{i+1}\n{seq}\n")

        results = {
            "name": structure["name"],
            "sequence_length": len(native_seq),
            "num_samples": num_samples,
            "inference_time": inference_time,
            "time_per_sample": inference_time / num_samples,
            "recovery": recovery_metrics,
            "burial_analysis": burial_metrics,
            "output_file": str(output_file)
        }

        # Print summary
        print(f"\nResults:")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Time per sample: {inference_time/num_samples:.2f}s")
        print(f"  Mean recovery: {recovery_metrics['mean_recovery']:.2f}%")
        print(f"  Sequences saved to: {output_file}")

        return results

    def run_benchmark(
        self,
        pdb_files: List[str],
        num_samples: int = 5
    ) -> Dict:
        """
        Run benchmark on multiple structures.

        Args:
            pdb_files: List of PDB file paths
            num_samples: Number of sequences per structure

        Returns:
            Dictionary with aggregate results
        """
        print(f"\nStarting benchmark suite")
        print(f"Number of structures: {len(pdb_files)}")
        print(f"Samples per structure: {num_samples}")

        all_results = []
        total_start = time.time()

        for pdb_file in pdb_files:
            result = self.benchmark_structure(pdb_file, num_samples)
            all_results.append(result)

        total_time = time.time() - total_start

        # Aggregate statistics
        valid_results = [r for r in all_results if "error" not in r]

        if not valid_results:
            print("Error: No valid results")
            return {"error": "No valid results"}

        aggregate = {
            "total_structures": len(pdb_files),
            "successful_structures": len(valid_results),
            "total_time": total_time,
            "total_samples": len(valid_results) * num_samples,
            "avg_time_per_structure": total_time / len(valid_results),
            "avg_recovery": np.mean([r["recovery"]["mean_recovery"] for r in valid_results]),
            "std_recovery": np.std([r["recovery"]["mean_recovery"] for r in valid_results]),
            "per_structure_results": all_results
        }

        # Save results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmark Complete")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg recovery: {aggregate['avg_recovery']:.2f}% ± {aggregate['std_recovery']:.2f}%")
        print(f"Results saved to: {results_file}")

        return aggregate


def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="ProteinMPNN Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing PDB files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (downloads if not specified)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sequences to generate per structure"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--download_examples",
        action="store_true",
        help="Download example PDB files"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of example structures to download"
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ProteinMPNNBenchmark(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir
    )

    # Get PDB files
    pdb_files = []
    data_path = Path(args.data_dir)

    if data_path.exists():
        pdb_files = list(data_path.glob("*.pdb"))

    if not pdb_files or args.download_examples:
        print("No PDB files found or download requested.")
        pdb_files = benchmark.download_example_pdbs(
            args.data_dir,
            args.num_examples
        )

    if not pdb_files:
        print("Error: No PDB files to benchmark")
        return

    print(f"Found {len(pdb_files)} PDB files")

    # Run benchmark
    results = benchmark.run_benchmark(
        pdb_files=[str(f) for f in pdb_files],
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
