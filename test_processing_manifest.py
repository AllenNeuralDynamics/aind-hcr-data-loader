#!/usr/bin/env python3
"""
Quick test to verify the processing manifest functionality
"""

from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aind_hcr_data_loader.hcr_dataset import HCRDataset, get_processing_manifests, create_channel_gene_table_from_manifests

def test_processing_manifest_structure():
    """Test that the HCRDataset class has the processing_manifests attribute"""
    
    # Create a mock dataset to test the structure
    spot_files = {}
    zarr_files = {}
    processing_manifests = {"R1": {"gene_dict": {"488": {"gene": "test_gene"}}}}
    
    # This should create without error
    dataset = HCRDataset(
        spot_files=spot_files,
        zarr_files=zarr_files,
        processing_manifests=processing_manifests
    )
    
    print("✓ HCRDataset class successfully created with processing_manifests attribute")
    print(f"✓ Processing manifests: {dataset.processing_manifests}")
    
    # Test the create_channel_gene_table_from_manifests function
    table = create_channel_gene_table_from_manifests(processing_manifests)
    print("✓ create_channel_gene_table_from_manifests function works")
    print(f"✓ Generated table:\n{table}")

if __name__ == "__main__":
    test_processing_manifest_structure()
    print("\n✅ All tests passed! Processing manifest implementation is working correctly.")
