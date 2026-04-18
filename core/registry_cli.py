#!/usr/bin/env python
"""
Registry CLI - Command-line interface for seed registry management

Usage:
    python registry_cli.py register --name "John Doe" --email "john@example.com"
    python registry_cli.py lookup --seed 1001
    python registry_cli.py search --name "John"
    python registry_cli.py export --output backup.json
    python registry_cli.py import --input backup.json
    python registry_cli.py stats
    python registry_cli.py list
    python registry_cli.py update --seed 1001 --notes "Updated notes"
    python registry_cli.py scan --image watermarked.png
    python registry_cli.py audit --seed 1001
"""

import argparse
import sys
import os
from typing import Optional

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.seed_registry import SeedRegistry

# Import inference functions for multi-seed scanning
try:
    from inference.inference import (
        load_image, extract_watermark, load_models, generate_watermark,
        calculate_ber, MODEL_SIZE
    )
    import torch
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


def register_command(args):
    """Register a new owner and assign seed."""
    registry = SeedRegistry(args.db)
    
    try:
        seed = registry.register_seed(
            owner_name=args.name,
            email=args.email,
            organization=args.organization or "",
            license_type=args.license or "personal",
            notes=args.notes or ""
        )
        print(f"\n{'='*50}")
        print(f"✅ REGISTRATION SUCCESSFUL")
        print(f"{'='*50}")
        print(f"Name:      {args.name}")
        print(f"Email:     {args.email}")
        if args.organization:
            print(f"Org:       {args.organization}")
        print(f"License:   {args.license or 'personal'}")
        print(f"{'─'*50}")
        print(f"🔑 ASSIGNED SEED: {seed}")
        print(f"{'='*50}\n")
        
    except ValueError as e:
        print(f"❌ Registration failed: {e}\n")
        return 1
    
    finally:
        registry.close()
    
    return 0


def lookup_command(args):
    """Look up owner by seed."""
    registry = SeedRegistry(args.db)
    
    try:
        owner = registry.lookup_seed(args.seed)
        
        if owner is None:
            print(f"\n❌ Seed {args.seed} not found\n")
            return 1
        
        print(f"\n{'='*50}")
        print(f"✅ SEED {args.seed} FOUND")
        print(f"{'='*50}")
        print(f"Name:        {owner['owner_name']}")
        print(f"Email:       {owner['owner_email']}")
        if owner['organization']:
            print(f"Organization: {owner['organization']}")
        print(f"License:     {owner['license_type']}")
        print(f"Registered:  {owner['created_at']}")
        print(f"Images:      {owner['images_count']} watermarked")
        if owner['notes']:
            print(f"Notes:       {owner['notes']}")
        print(f"{'='*50}\n")
    
    finally:
        registry.close()
    
    return 0


def search_command(args):
    """Search for owners by name or email."""
    registry = SeedRegistry(args.db)
    
    try:
        if args.name:
            seeds = registry.find_by_owner(args.name)
            search_type = f"name: '{args.name}'"
        elif args.email:
            seed = registry.find_by_email(args.email)
            seeds = [seed] if seed else []
            search_type = f"email: '{args.email}'"
        else:
            print("❌ Specify --name or --email\n")
            return 1
        
        if not seeds:
            print(f"\n❌ No seeds found for {search_type}\n")
            return 1
        
        print(f"\n{'='*50}")
        print(f"✅ FOUND {len(seeds)} SEED(S)")
        print(f"{'='*50}")
        
        for seed in seeds:
            owner = registry.lookup_seed(seed)
            print(f"\n  🔑 Seed: {seed}")
            print(f"     Name:   {owner['owner_name']}")
            print(f"     Email:  {owner['owner_email']}")
            print(f"     Images: {owner['images_count']}")
        
        print(f"\n{'='*50}\n")
    
    finally:
        registry.close()
    
    return 0


def export_command(args):
    """Export registry to JSON."""
    registry = SeedRegistry(args.db)
    
    try:
        success = registry.export_to_json(args.output)
        if not success:
            return 1
    
    finally:
        registry.close()
    
    return 0


def import_command(args):
    """Import registry from JSON."""
    registry = SeedRegistry(args.db)
    
    try:
        success = registry.import_from_json(args.input)
        if not success:
            return 1
    
    finally:
        registry.close()
    
    return 0


def stats_command(args):
    """Display registry statistics."""
    registry = SeedRegistry(args.db)
    
    try:
        stats = registry.get_statistics()
        
        print(f"\n{'='*50}")
        print(f"📊 REGISTRY STATISTICS")
        print(f"{'='*50}")
        print(f"Total Seeds:           {stats['total_seeds']}")
        print(f"Total Watermarked:     {stats['total_images_watermarked']} images")
        
        if stats['most_active_seed']:
            seed_info = stats['most_active_seed']
            # Handle both dict and Row object
            seed_num = seed_info.get('seed') if isinstance(seed_info, dict) else seed_info['seed']
            owner_name = seed_info.get('owner_name') if isinstance(seed_info, dict) else seed_info['owner_name']
            images_count = seed_info.get('images_count') if isinstance(seed_info, dict) else seed_info['images_count']
            
            print(f"\nMost Active Seed:      {seed_num}")
            print(f"  Owner:               {owner_name}")
            print(f"  Images:              {images_count}")
        
        print(f"\nDatabase Size:         {stats['db_size_bytes'] / 1024:.2f} KB")
        print(f"{'='*50}\n")
    
    finally:
        registry.close()
    
    return 0


def list_command(args):
    """List all registered seeds."""
    registry = SeedRegistry(args.db)
    
    try:
        seeds = registry.get_all_seeds()
        
        if not seeds:
            print("\n❌ No seeds registered\n")
            return 1
        
        print(f"\n{'='*70}")
        print(f"📋 SEED REGISTRY ({len(seeds)} total)")
        print(f"{'='*70}")
        print(f"{'Seed':<6} {'Owner':<20} {'Email':<25} {'Images':<8} License")
        print(f"{'-'*70}")
        
        for seed in seeds:
            print(f"{seed['seed']:<6} {seed['owner_name']:<20} {seed['owner_email']:<25} {seed['images_count']:<8} {seed['license_type']}")
        
        print(f"{'='*70}\n")
    
    finally:
        registry.close()
    
    return 0


def update_command(args):
    """Update seed record."""
    registry = SeedRegistry(args.db)
    
    try:
        if not registry.lookup_seed(args.seed):
            print(f"\n❌ Seed {args.seed} not found\n")
            return 1
        
        updates = {}
        if args.name:
            updates['owner_name'] = args.name
        if args.notes:
            updates['notes'] = args.notes
        if args.license:
            updates['license_type'] = args.license
        if args.organization:
            updates['organization'] = args.organization
        
        if not updates:
            print("❌ No fields to update\n")
            return 1
        
        registry.update_owner(args.seed, **updates)
        
        print(f"\n✅ Updated seed {args.seed}")
        owner = registry.lookup_seed(args.seed)
        print(f"   Name:    {owner['owner_name']}")
        print(f"   License: {owner['license_type']}")
        print(f"   Notes:   {owner['notes']}\n")
    
    finally:
        registry.close()
    
    return 0


def delete_command(args):
    """Delete a seed (with confirmation)."""
    registry = SeedRegistry(args.db)
    
    try:
        owner = registry.lookup_seed(args.seed)
        
        if not owner:
            print(f"\n❌ Seed {args.seed} not found\n")
            return 1
        
        if not args.force:
            print(f"\n⚠️  About to delete:")
            print(f"   Seed:  {args.seed}")
            print(f"   Owner: {owner['owner_name']}")
            response = input("\nType 'yes' to confirm deletion: ").strip().lower()
            
            if response != 'yes':
                print("❌ Deletion cancelled\n")
                return 1
        
        registry.delete_seed(args.seed)
        print(f"✅ Deleted seed {args.seed}\n")
    
    finally:
        registry.close()
    
    return 0


def scan_command(args):
    """Multi-seed scanning: Try all registered seeds against an image (blind extraction)."""
    if not INFERENCE_AVAILABLE:
        print("❌ Inference modules not available. Install with: pip install torch torchvision")
        return 1
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}\n")
        return 1
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}\n")
        return 1
    
    registry = SeedRegistry(args.db)
    
    try:
        # Load models
        print("⏳ Loading models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder, decoder, semantic_encoder, semantic_decoder, poisoner = \
            load_models(args.checkpoint, device)
        
        # Load image
        print(f"📷 Loading image: {args.image}")
        img_tensor, orig_size = load_image(args.image, resize_to=(MODEL_SIZE, MODEL_SIZE))
        
        # Get all registered seeds
        all_seeds = registry.get_all_seeds()
        if not all_seeds:
            print("❌ No seeds registered in database\n")
            return 1
        
        print(f"\n🔍 Scanning {len(all_seeds)} registered seed(s)...")
        print("=" * 80)
        
        results = []
        
        # Try each seed
        for seed_record in all_seeds:
            seed = seed_record['seed']
            owner_name = seed_record['owner_name']
            
            # Generate watermark for this seed
            watermark = generate_watermark(seed=seed)
            
            # Extract from image
            extracted_bits, logits = extract_watermark(
                img_tensor, decoder, semantic_decoder, device, use_semantic=False
            )
            
            # Calculate BER
            ber = calculate_ber(extracted_bits, watermark.to(device))
            
            # Confidence: 0% BER = 100%, 50% BER = 0%
            confidence = max(0, (0.5 - ber) / 0.5) * 100 if ber <= 0.5 else 0
            
            # Determine match status
            match = "✅ MATCH" if ber < 0.15 else "🟡 PARTIAL" if ber < 0.3 else "❌ NO"
            
            results.append({
                'seed': seed,
                'owner': owner_name,
                'ber': ber,
                'confidence': confidence,
                'match': match
            })
            
            print(f"Seed {seed:6d} | {owner_name:25s} | BER: {ber:.4f} | Conf: {confidence:5.1f}% | {match}")
        
        print("=" * 80)
        
        # Sort by BER and show top matches
        results_sorted = sorted(results, key=lambda x: x['ber'])
        
        print(f"\n🏆 TOP MATCHES:")
        for i, result in enumerate(results_sorted[:3], 1):
            print(f"\n  {i}. Seed {result['seed']} - {result['owner']}")
            print(f"     BER:        {result['ber']:.4f}")
            print(f"     Confidence: {result['confidence']:.1f}%")
            print(f"     Status:     {result['match']}")
        
        print(f"\n✅ Scan complete. Best match: Seed {results_sorted[0]['seed']} ({results_sorted[0]['owner']})\n")
    
    finally:
        registry.close()
    
    return 0


def audit_command(args):
    """View audit log for a seed or all seeds."""
    registry = SeedRegistry(args.db)
    
    try:
        entries = registry.get_audit_log(seed=args.seed, limit=args.limit)
        
        if not entries:
            if args.seed:
                print(f"\n❌ No audit log entries for seed {args.seed}\n")
            else:
                print(f"\n❌ No audit log entries\n")
            return 1
        
        seed_filter = f" for seed {args.seed}" if args.seed else ""
        print(f"\n{'='*80}")
        print(f"📋 AUDIT LOG{seed_filter} ({len(entries)} entries)")
        print(f"{'='*80}")
        print(f"{'Timestamp':<25} {'Action':<12} {'Seed':<8} {'Details':<30}")
        print(f"{'-'*80}")
        
        for entry in entries:
            ts = str(entry['timestamp'])[:19] if entry['timestamp'] else "N/A"
            action = entry['action']
            seed = str(entry['seed']) if entry['seed'] else "-"
            details = (entry['details'][:25] + "...") if len(entry['details'] or "") > 25 else (entry['details'] or "-")
            print(f"{ts:<25} {action:<12} {seed:<8} {details:<30}")
        
        print(f"{'='*80}\n")
    
    finally:
        registry.close()
    
    return 0


def batch_command(args):
    """Batch process images using owner from registry."""
    import subprocess
    import os
    
    registry = SeedRegistry(args.db)
    
    try:
        # Find owner's seed
        seeds = registry.find_by_owner(args.owner)
        if not seeds:
            print(f"\n❌ Owner '{args.owner}' not found in registry\n")
            return 1
        
        seed = seeds[0]
        owner_info = registry.lookup_seed(seed)
        print(f"\n✅ Found owner: {owner_info['owner_name']}")
        print(f"   Seed: {seed}")
        print(f"   Email: {owner_info['owner_email']}")
        
        # Build inference.py command from the repo's current layout
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inference_script = os.path.join(project_root, 'inference', 'inference.py')
        cmd = [
            'python', inference_script,
            '--input_dir', args.input_dir,
            '--output_dir', args.output_dir,
            '--mode', args.mode,
            '--owner', args.owner,
            '--alpha', str(args.alpha)
        ]
        
        if args.csv_output:
            cmd.extend(['--output_csv', args.csv_output])
        
        print(f"\n🚀 Starting batch processing...")
        result = subprocess.run(cmd, cwd=project_root)
        
        return result.returncode
    
    finally:
        registry.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🔑 Seed Registry Manager - Manage watermark ownership",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Register a new owner:
    python registry_cli.py register --name "Alice" --email "alice@photo.com"
  
  Lookup by seed:
    python registry_cli.py lookup --seed 1001
  
  Search by name:
    python registry_cli.py search --name "Alice"
  
  Export database:
    python registry_cli.py export --output backup.json
  
  Show statistics:
    python registry_cli.py stats
  
  List all:
    python registry_cli.py list
        """
    )
    
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_db = os.path.join(_project_root, 'core', 'seed_registry.db')
    parser.add_argument('--db', default=_default_db, 
                       help='Database path (default: seed_registry.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register new owner')
    register_parser.add_argument('--name', required=True, help='Owner name')
    register_parser.add_argument('--email', required=True, help='Email address')
    register_parser.add_argument('--organization', help='Organization/company')
    register_parser.add_argument('--license', choices=['personal', 'non-exclusive', 'exclusive'],
                               help='License type (default: personal)')
    register_parser.add_argument('--notes', help='Additional notes')
    register_parser.set_defaults(func=register_command)
    
    # Lookup command
    lookup_parser = subparsers.add_parser('lookup', help='Look up seed')
    lookup_parser.add_argument('--seed', type=int, required=True, help='Seed number')
    lookup_parser.set_defaults(func=lookup_command)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search by name or email')
    search_parser.add_argument('--name', help='Owner name (partial match)')
    search_parser.add_argument('--email', help='Email address (exact match)')
    search_parser.set_defaults(func=search_command)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export to JSON')
    export_parser.add_argument('--output', required=True, help='Output JSON file')
    export_parser.set_defaults(func=export_command)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import from JSON')
    import_parser.add_argument('--input', required=True, help='Input JSON file')
    import_parser.set_defaults(func=import_command)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=stats_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all seeds')
    list_parser.set_defaults(func=list_command)
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update seed record')
    update_parser.add_argument('--seed', type=int, required=True, help='Seed to update')
    update_parser.add_argument('--name', help='New owner name')
    update_parser.add_argument('--organization', help='New organization')
    update_parser.add_argument('--license', help='New license type')
    update_parser.add_argument('--notes', help='New notes')
    update_parser.set_defaults(func=update_command)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete seed (caution!)')
    delete_parser.add_argument('--seed', type=int, required=True, help='Seed to delete')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    delete_parser.set_defaults(func=delete_command)
    
    # Scan command (multi-seed)
    scan_parser = subparsers.add_parser('scan', help='Scan image against all seeds (blind extraction)')
    scan_parser.add_argument('--image', type=str, required=True, help='Image file to scan')
    _default_ckpt = os.path.join(_project_root, 'checkpoints', 'checkpoint_hvs_best.pth')
    scan_parser.add_argument('--checkpoint', type=str, default=_default_ckpt, help='Model checkpoint')
    scan_parser.set_defaults(func=scan_command)
    
    # Audit command
    audit_parser = subparsers.add_parser('audit', help='View audit log entries')
    audit_parser.add_argument('--seed', type=int, help='Filter by seed (optional)')
    audit_parser.add_argument('--limit', type=int, default=50, help='Maximum entries (default 50)')
    audit_parser.set_defaults(func=audit_command)
    
    # Batch command - list owners for seed selection
    batch_parser = subparsers.add_parser('batch', help='Batch process images with owner/seed')
    batch_parser.add_argument('--owner', required=True, help='Owner name')
    batch_parser.add_argument('--input-dir', required=True, help='Input directory')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--csv-output', help='CSV report path')
    batch_parser.add_argument('--mode', choices=['embed', 'decode'], default='embed', help='Processing mode')
    batch_parser.add_argument('--alpha', type=float, default=0.035, help='Watermark strength')
    batch_parser.set_defaults(func=batch_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
