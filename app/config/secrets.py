"""
Secrets management for Contract Analysis System.
"""

import os
import json
import base64
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

class SecretsManager:
    """Secure secrets management."""

    def __init__(self, master_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.secrets_file = Path("app/var/secrets.encrypted")
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)

        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._get_or_create_master_key()

        self.cipher = self._create_cipher()
        self._secrets_cache = {}

    def _get_or_create_master_key(self) -> bytes:
        """Get or create master key for encryption."""
        key_file = Path("app/var/master.key")

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(master_key)

            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            self.logger.info("Generated new master key")
            return master_key

    def _create_cipher(self) -> Fernet:
        """Create encryption cipher."""
        return Fernet(self.master_key)

    def store_secret(self, key: str, value: str, description: str = "") -> bool:
        """Store a secret securely."""
        try:
            # Load existing secrets
            secrets_data = self._load_secrets()

            # Add new secret
            secrets_data[key] = {
                'value': value,
                'description': description,
                'created_at': self._get_timestamp(),
                'encrypted': True
            }

            # Save encrypted secrets
            self._save_secrets(secrets_data)
            self._secrets_cache[key] = value

            self.logger.info(f"Secret '{key}' stored successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        try:
            # Check cache first
            if key in self._secrets_cache:
                return self._secrets_cache[key]

            # Load from file
            secrets_data = self._load_secrets()

            if key not in secrets_data:
                self.logger.warning(f"Secret '{key}' not found")
                return None

            secret_info = secrets_data[key]
            value = secret_info['value']

            # Cache the value
            self._secrets_cache[key] = value

            return value

        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        try:
            secrets_data = self._load_secrets()

            if key in secrets_data:
                del secrets_data[key]
                self._save_secrets(secrets_data)

                # Remove from cache
                if key in self._secrets_cache:
                    del self._secrets_cache[key]

                self.logger.info(f"Secret '{key}' deleted successfully")
                return True
            else:
                self.logger.warning(f"Secret '{key}' not found")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False

    def list_secrets(self) -> Dict[str, Dict[str, Any]]:
        """List all secrets (without values)."""
        try:
            secrets_data = self._load_secrets()

            # Return metadata without actual values
            return {
                key: {
                    'description': info.get('description', ''),
                    'created_at': info.get('created_at', ''),
                    'encrypted': info.get('encrypted', False)
                }
                for key, info in secrets_data.items()
            }

        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return {}

    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from encrypted file."""
        if not self.secrets_file.exists():
            return {}

        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())

        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
            return {}

    def _save_secrets(self, secrets_data: Dict[str, Any]) -> None:
        """Save secrets to encrypted file."""
        try:
            json_data = json.dumps(secrets_data).encode()
            encrypted_data = self.cipher.encrypt(json_data)

            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)

        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
            raise

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def rotate_secrets(self) -> bool:
        """Rotate all secrets (generate new values)."""
        try:
            secrets_data = self._load_secrets()

            for key, info in secrets_data.items():
                if 'rotate' in info.get('description', '').lower():
                    # Generate new secret value
                    new_value = secrets.token_urlsafe(32)
                    info['value'] = new_value
                    info['rotated_at'] = self._get_timestamp()

                    # Update cache
                    self._secrets_cache[key] = new_value

            self._save_secrets(secrets_data)
            self.logger.info("Secrets rotated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to rotate secrets: {e}")
            return False

class EnvironmentSecrets:
    """Environment-based secrets management."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_secrets = [
            'API_TOKEN',
            'JWT_SECRET_KEY',
            'DATABASE_URL',
            'REDIS_URL'
        ]

    def validate_environment_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are present in environment."""
        validation_results = {}

        for secret in self.required_secrets:
            value = os.getenv(secret)
            validation_results[secret] = value is not None and len(value) > 0

        return validation_results

    def get_missing_secrets(self) -> List[str]:
        """Get list of missing required secrets."""
        validation_results = self.validate_environment_secrets()
        return [secret for secret, present in validation_results.items() if not present]

    def generate_secret_template(self) -> str:
        """Generate template for required secrets."""
        template = "# Required Environment Secrets\n"
        template += "# Copy this to .env file and fill in the values\n\n"

        for secret in self.required_secrets:
            template += f"{secret}=your-{secret.lower().replace('_', '-')}-here\n"

        return template

# Global instances
secrets_manager = SecretsManager()
env_secrets = EnvironmentSecrets()
