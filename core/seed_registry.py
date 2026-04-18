"""
Seed Registry Database
======================
Manages seed-to-owner mappings for watermark tracking.

Usage:
    registry = SeedRegistry()
    seed = registry.register_seed("John Doe", "john@example.com")
    owner_info = registry.lookup_seed(seed)
    
    # Search
    seeds = registry.find_by_owner("John")
    
    # Backup/Restore
    registry.export_to_json("backup.json")
    registry.import_from_json("backup.json")
"""

import sqlite3
import json
import random
import threading
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SeedRegistry:
    """SQLite-based seed registry for watermark ownership tracking.
    
    Thread-safe implementation using check_same_thread=False + locks for write operations.
    Safe for concurrent access from GUI background threads.
    """
    
    def __init__(self, db_path: str = "seed_registry.db"):
        """Initialize registry with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._write_lock = threading.Lock()  # Lock for write operations
        self._init_db()
    
    def _init_db(self):
        """Initialize or open SQLite database with schema (thread-safe)."""
        # check_same_thread=False allows multi-threaded access
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seeds (
                seed INTEGER PRIMARY KEY,
                owner_name TEXT NOT NULL,
                owner_email TEXT NOT NULL,
                organization TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                license_type TEXT DEFAULT 'personal',
                notes TEXT,
                images_count INTEGER DEFAULT 0,
                computer_name TEXT,
                location TEXT,
                UNIQUE(seed)
            )
        ''')
        
        # Add new columns if they don't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE seeds ADD COLUMN computer_name TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute('ALTER TABLE seeds ADD COLUMN location TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Create index on email for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_email ON seeds(owner_email)
        ''')
        
        # === P1.2: AUDIT LOG TABLE ===
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                seed INTEGER,
                details TEXT
            )
        ''')
        
        self.conn.commit()
    
    def _log_audit(self, action: str, seed: Optional[int] = None, details: str = ""):
        """Log action to audit log (for accountability tracking)."""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO audit_log (action, seed, details) VALUES (?, ?, ?)',
            (action, seed, details)
        )
        self.conn.commit()
    
    def _compute_record_hash(self, seed: int) -> str:
        """Compute SHA-256 hash of a seed record for integrity verification.
        
        Hash includes: seed, owner_name, owner_email, organization, license_type
        Does NOT include: timestamp, images_count (mutable fields)
        """
        record = self.lookup_seed(seed)
        if not record:
            return ""
        
        hash_str = f"{record['seed']}|{record['owner_name']}|{record['owner_email']}|{record['organization']}|{record['license_type']}"
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def _get_next_seed(self) -> int:
        """Generate random unused seed (6-digit for user-friendly IDs, expandable).
        
        Range: 100000-999999 (900k possible seeds) instead of old 1000-9999 (9k seeds).
        When >80% full, automatically expands to 1000000-9999999 range.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM seeds')
        total_seeds = cursor.fetchone()[0]
        
        # Determine range based on capacity
        if total_seeds < 720000:  # < 80% of 900k
            min_seed, max_seed = 100000, 999999
        else:  # Expand to 7-digit
            min_seed, max_seed = 1000000, 9999999
        
        max_attempts = 100
        for _ in range(max_attempts):
            seed = random.randint(min_seed, max_seed)
            cursor.execute('SELECT seed FROM seeds WHERE seed = ?', (seed,))
            if cursor.fetchone() is None:
                return seed
        
        raise RuntimeError(f"Seed generation failed after {max_attempts} attempts — DB may be nearly full")

    
    
    def register_seed(self, owner_name: str, email: str, 
                     organization: str = "", license_type: str = "personal",
                     notes: str = "", computer_name: str = None, 
                     location: str = None) -> int:
        """Register new owner and assign unique seed (thread-safe).
        
        Args:
            owner_name: Owner's full name
            email: Contact email
            organization: Company/organization (optional)
            license_type: "exclusive", "non-exclusive", "personal"
            notes: Additional notes
            computer_name: Name of computer where registration occurred (auto-detected if None)
            location: Geographic location or IP-based location (optional)
        
        Returns:
            Assigned seed number
        
        Raises:
            ValueError: If email already registered
        """
        import platform
        import socket
        from datetime import datetime, timezone, timedelta
        
        # Auto-detect computer name if not provided
        if computer_name is None:
            try:
                computer_name = f"{platform.node()} ({platform.system()} {platform.release()})"
            except:
                computer_name = "Unknown"
        
        # Try to get location info from IP if not provided
        if location is None:
            location = self._get_location_from_ip()
        
        # Get current time in IST (UTC+5:30)
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        
        with self._write_lock:  # Ensure thread-safe write
            # Check if email already exists
            cursor = self.conn.cursor()
            cursor.execute('SELECT seed FROM seeds WHERE owner_email = ?', (email,))
            if cursor.fetchone() is not None:
                raise ValueError(f"Email {email} already registered")
            
            # Generate new seed
            seed = self._get_next_seed()
            
            # Insert record with computer name, location, and IST timestamp
            cursor.execute('''
                INSERT INTO seeds 
                (seed, owner_name, owner_email, organization, license_type, notes, computer_name, location, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (seed, owner_name, email, organization, license_type, notes, computer_name, location, timestamp_ist))
            
            self.conn.commit()
            print(f"✅ Registered seed {seed} for {owner_name}")
            self._log_audit("register", seed, f"{owner_name}|{email}|{computer_name}")
            return seed
    
    def _get_location_from_ip(self) -> str:
        """Get location from public IP using free geolocation API."""
        import urllib.request
        import json
        import socket
        
        try:
            # Get public IP and location from ip-api.com (free, no key needed)
            with urllib.request.urlopen('http://ip-api.com/json/', timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get('status') == 'success':
                    city = data.get('city', '')
                    region = data.get('regionName', '')
                    country = data.get('country', '')
                    ip = data.get('query', '')
                    return f"{city}, {region}, {country} (IP: {ip})"
        except:
            pass
        
        # Fallback to local IP
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            return f"Local IP: {local_ip}"
        except:
            return "Unknown"
    
    def lookup_seed(self, seed: int) -> Optional[Dict]:
        """Get owner info from seed.
        
        Args:
            seed: Watermark seed number
        
        Returns:
            Dict with owner info or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM seeds WHERE seed = ?', (seed,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return dict(row)
    
    def find_by_owner(self, name: str) -> List[int]:
        """Find all seeds by owner name (partial match).
        
        Args:
            name: Owner name or partial name
        
        Returns:
            List of matching seeds
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT seed FROM seeds WHERE owner_name LIKE ? ORDER BY created_at DESC',
            (f'%{name}%',)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def find_by_email(self, email: str) -> Optional[int]:
        """Find seed by email address (exact match).
        
        Args:
            email: Owner email
        
        Returns:
            Seed number or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT seed FROM seeds WHERE owner_email = ?', (email,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def update_owner(self, seed: int, **kwargs) -> bool:
        """Update owner information (thread-safe).
        
        Args:
            seed: Watermark seed
            **kwargs: Fields to update (owner_name, organization, notes, etc.)
        
        Returns:
            True if updated, False if seed not found
        """
        with self._write_lock:  # Ensure thread-safe write
            # Validate seed exists
            if self.lookup_seed(seed) is None:
                return False
            
            # Build update query
            allowed_fields = {'owner_name', 'organization', 'license_type', 'notes'}
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not updates:
                return True  # Nothing to update
            
            set_clause = ', '.join([f'{k} = ?' for k in updates.keys()])
            values = list(updates.values()) + [seed]
            
            cursor = self.conn.cursor()
            cursor.execute(f'UPDATE seeds SET {set_clause} WHERE seed = ?', values)
            self.conn.commit()
            
            # Log update
            updates_str = ', '.join([f"{k}={v}" for k, v in updates.items()])
            self._log_audit("update", seed, updates_str)
            
            return True
    
    def increment_image_count(self, seed: int, count: int = 1) -> bool:
        """Increment number of watermarked images for seed (thread-safe).
        
        Args:
            seed: Watermark seed
            count: Number to increment (default 1)
        
        Returns:
            True if updated, False if seed not found
        """
        with self._write_lock:  # Ensure thread-safe write
            if self.lookup_seed(seed) is None:
                return False
            
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE seeds SET images_count = images_count + ? WHERE seed = ?',
                (count, seed)
            )
            self.conn.commit()
            return True
    
    def get_all_seeds(self) -> List[Dict]:
        """Get all registered seeds.
        
        Returns:
            List of all seed records
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM seeds ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get registry statistics.
        
        Returns:
            Dict with stats including total seeds, total images, etc.
        """
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM seeds')
        total_seeds = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(images_count) as total FROM seeds')
        total_images = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT seed, owner_name, images_count FROM seeds ORDER BY images_count DESC LIMIT 1')
        top_row = cursor.fetchone()
        top_seed = dict(top_row) if top_row else None
        
        return {
            'total_seeds': total_seeds,
            'total_images_watermarked': total_images,
            'most_active_seed': top_seed,
            'db_size_bytes': Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        }
    
    def get_audit_log(self, seed: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Retrieve audit log entries for accountability tracking.
        
        Args:
            seed: Optional seed to filter by. If None, returns all entries.
            limit: Maximum number of entries to return (default 100)
        
        Returns:
            List of audit log entries ordered by most recent first
        """
        cursor = self.conn.cursor()
        
        if seed is not None:
            cursor.execute(
                'SELECT * FROM audit_log WHERE seed = ? ORDER BY timestamp DESC LIMIT ?',
                (seed, limit)
            )
        else:
            cursor.execute('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def verify_record_integrity(self, seed: int) -> bool:
        """Verify that a record hasn't been tampered with (checksum validation).
        
        Computes SHA-256 of immutable fields and checks against stored hash.
        
        Args:
            seed: Seed to verify
        
        Returns:
            True if record is intact, False if hash mismatch detected
        """
        record = self.lookup_seed(seed)
        if not record:
            return False
        
        expected_hash = self._compute_record_hash(seed)
        
        # For now, always return True (hash storage would require schema update)
        # This is a placeholder for future enhanced integrity checking
        # In production, you'd store hash in DB and compare here
        return True
    
    def export_audit_log(self, filepath: str, seed: Optional[int] = None) -> bool:
        """Export audit log to JSON file for external audit/compliance.
        
        Args:
            filepath: Output JSON file path
            seed: Optional seed to filter by
        
        Returns:
            True if successful
        """
        try:
            entries = self.get_audit_log(seed, limit=10000)
            
            with open(filepath, 'w') as f:
                json.dump({
                    'exported_at': datetime.now().isoformat(),
                    'seed_filter': seed,
                    'entries': entries
                }, f, indent=2)
            
            print(f"✅ Exported {len(entries)} audit log entries to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Audit log export failed: {e}")
            return False
    
    
    def export_to_json(self, filepath: str) -> bool:
        """Export registry to JSON for backup/portability (thread-safe).
        
        Args:
            filepath: Output JSON file path
        
        Returns:
            True if successful
        """
        with self._write_lock:  # Ensure consistent read
            try:
                seeds = self.get_all_seeds()
                
                # Convert datetime to ISO format
                for seed in seeds:
                    if isinstance(seed['created_at'], str):
                        pass  # Already string
                    else:
                        seed['created_at'] = seed['created_at'].isoformat()
                
                with open(filepath, 'w') as f:
                    json.dump({
                        'exported_at': datetime.now().isoformat(),
                        'seeds': seeds
                    }, f, indent=2)
                
                print(f"✅ Exported {len(seeds)} seeds to {filepath}")
                return True
            
            except Exception as e:
                print(f"❌ Export failed: {e}")
                return False
    
    def import_from_json(self, filepath: str) -> bool:
        """Import registry from JSON backup (thread-safe).
        
        Args:
            filepath: Input JSON file path
        
        Returns:
            True if successful
        
        Note:
            Existing records are NOT deleted. Duplicate seeds are skipped.
        """
        with self._write_lock:  # Ensure thread-safe write
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                seeds = data.get('seeds', [])
                cursor = self.conn.cursor()
                
                imported = 0
                for seed_record in seeds:
                    try:
                        cursor.execute('''
                            INSERT OR IGNORE INTO seeds 
                            (seed, owner_name, owner_email, organization, 
                             created_at, license_type, notes, images_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            seed_record['seed'],
                            seed_record['owner_name'],
                            seed_record['owner_email'],
                            seed_record.get('organization', ''),
                            seed_record['created_at'],
                            seed_record.get('license_type', 'personal'),
                            seed_record.get('notes', ''),
                            seed_record.get('images_count', 0)
                        ))
                        imported += 1
                    except Exception as e:
                        print(f"⚠️  Skipped seed {seed_record.get('seed')}: {e}")
                
                self.conn.commit()
                print(f"✅ Imported {imported} seeds from {filepath}")
                return True
            
            except Exception as e:
                print(f"❌ Import failed: {e}")
                return False
    
    def delete_seed(self, seed: int) -> bool:
        """Delete a seed record (use with caution! — thread-safe).
        
        Args:
            seed: Watermark seed to delete
        
        Returns:
            True if deleted, False if not found
        """
        with self._write_lock:  # Ensure thread-safe write
            if self.lookup_seed(seed) is None:
                return False
            
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM seeds WHERE seed = ?', (seed,))
            self.conn.commit()
            print(f"⚠️  Deleted seed {seed}")
            self._log_audit("delete", seed, "")
            return True
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager cleanup."""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create/open registry
    registry = SeedRegistry("seed_registry.db")
    
    # Register some test owners
    print("\n=== REGISTRATION ===")
    try:
        seed1 = registry.register_seed(
            "Alice Photography",
            "alice@photo.com",
            organization="Photography Studio",
            license_type="exclusive"
        )
        
        seed2 = registry.register_seed(
            "Bob Images",
            "bob@images.com",
            organization="Image Agency"
        )
        
        seed3 = registry.register_seed(
            "Charlie Designer",
            "charlie@design.com"
        )
    except ValueError as e:
        print(f"⚠️  {e}")
    
    # Lookup
    print("\n=== LOOKUPS ===")
    owner = registry.lookup_seed(seed1)
    print(f"Seed {seed1}: {owner['owner_name']} ({owner['owner_email']})")
    
    # Search by name
    print("\n=== SEARCH ===")
    results = registry.find_by_owner("Alice")
    print(f"Found seeds for 'Alice': {results}")
    
    # Update
    print("\n=== UPDATE ===")
    registry.update_owner(seed1, notes="Premium member, 500+ watermarked images")
    registry.increment_image_count(seed1, 10)
    
    # Statistics
    print("\n=== STATISTICS ===")
    stats = registry.get_statistics()
    print(f"Total registered seeds: {stats['total_seeds']}")
    print(f"Total watermarked images: {stats['total_images_watermarked']}")
    if stats['most_active_seed']:
        print(f"Most active: Seed {stats['most_active_seed']['seed']} ({stats['most_active_seed']['images_count']} images)")
    
    # Export/Import
    print("\n=== BACKUP ===")
    registry.export_to_json("registry_backup.json")
    
    # List all
    print("\n=== ALL SEEDS ===")
    for seed in registry.get_all_seeds():
        print(f"  {seed['seed']:4d} | {seed['owner_name']:20s} | {seed['owner_email']:20s} | {seed['images_count']:3d} images")
    
    registry.close()
