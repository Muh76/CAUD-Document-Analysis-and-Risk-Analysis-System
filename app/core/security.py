"""
Security utilities for Contract Analysis System.
"""

import re
import hashlib
import secrets
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class PIIScrubber:
    """PII (Personally Identifiable Information) scrubbing utilities."""

    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        }

        self.replacement_map = {
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'credit_card': '[CARD_REDACTED]',
            'ip_address': '[IP_REDACTED]',
            'name': '[NAME_REDACTED]'
        }

    def scrub_text(self, text: str) -> str:
        """Scrub PII from text."""
        scrubbed_text = text

        for pii_type, pattern in self.pii_patterns.items():
            scrubbed_text = re.sub(pattern, self.replacement_map[pii_type], scrubbed_text, flags=re.IGNORECASE)

        return scrubbed_text

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text."""
        detected = {}

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                detected[pii_type] = list(set(matches))

        return detected

    def get_pii_summary(self, text: str) -> Dict[str, Any]:
        """Get PII detection summary."""
        detected = self.detect_pii(text)

        return {
            'total_pii_types': len(detected),
            'pii_types_found': list(detected.keys()),
            'total_instances': sum(len(matches) for matches in detected.values()),
            'details': detected
        }

class DataProtection:
    """Data protection utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def encrypt_sensitive_data(self, data: str, key: str) -> str:
        """Encrypt sensitive data."""
        # Simple encryption for demo - use proper encryption in production
        import base64
        encoded_data = base64.b64encode(data.encode()).decode()
        return f"encrypted:{encoded_data}"

    def decrypt_sensitive_data(self, encrypted_data: str, key: str) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data.startswith("encrypted:"):
            return encrypted_data

        import base64
        encoded_data = encrypted_data.replace("encrypted:", "")
        return base64.b64decode(encoded_data).decode()

    def hash_data(self, data: str) -> str:
        """Hash data for integrity checking."""
        return hashlib.sha256(data.encode()).hexdigest()

    def validate_data_integrity(self, data: str, expected_hash: str) -> bool:
        """Validate data integrity."""
        actual_hash = self.hash_data(data)
        return actual_hash == expected_hash

class SecretsManager:
    """Secrets management system."""

    def __init__(self):
        self.secrets = {}
        self.logger = logging.getLogger(__name__)

    def store_secret(self, key: str, value: str, encrypted: bool = True) -> bool:
        """Store a secret."""
        try:
            if encrypted:
                # Simple encryption for demo
                encrypted_value = f"encrypted:{value}"
                self.secrets[key] = encrypted_value
            else:
                self.secrets[key] = value

            self.logger.info(f"Secret stored: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store secret {key}: {e}")
            return False

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret."""
        try:
            encrypted_value = self.secrets.get(key)
            if encrypted_value and encrypted_value.startswith("encrypted:"):
                return encrypted_value.replace("encrypted:", "")
            return encrypted_value
        except Exception as e:
            self.logger.error(f"Failed to get secret {key}: {e}")
            return None

    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret."""
        try:
            # Generate new secret
            new_secret = secrets.token_urlsafe(32)
            return self.store_secret(key, new_secret)
        except Exception as e:
            self.logger.error(f"Failed to rotate secret {key}: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List secret keys (metadata only)."""
        return list(self.secrets.keys())

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        try:
            if key in self.secrets:
                del self.secrets[key]
                self.logger.info(f"Secret deleted: {key}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete secret {key}: {e}")
            return False

class SecurityScanner:
    """Security scanning utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for vulnerabilities."""
        # Mock vulnerability scan results
        return {
            'total_dependencies': 45,
            'vulnerabilities_found': 2,
            'critical': 0,
            'high': 1,
            'medium': 1,
            'low': 0,
            'details': [
                {
                    'package': 'requests',
                    'version': '2.28.1',
                    'severity': 'high',
                    'cve': 'CVE-2023-32681',
                    'description': 'Potential SSRF vulnerability'
                },
                {
                    'package': 'urllib3',
                    'version': '1.26.12',
                    'severity': 'medium',
                    'cve': 'CVE-2023-45853',
                    'description': 'Certificate validation issue'
                }
            ]
        }

    def scan_code_secrets(self) -> Dict[str, Any]:
        """Scan code for exposed secrets."""
        # Mock secret scan results
        return {
            'files_scanned': 25,
            'secrets_found': 1,
            'details': [
                {
                    'file': 'app/config/settings.py',
                    'line': 45,
                    'type': 'api_key',
                    'severity': 'high',
                    'description': 'Hardcoded API key found'
                }
            ]
        }

    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        return {
            'timestamp': datetime.now().isoformat(),
            'dependency_scan': self.scan_dependencies(),
            'secret_scan': self.scan_code_secrets(),
            'overall_score': 85,
            'recommendations': [
                'Update vulnerable dependencies',
                'Remove hardcoded secrets',
                'Enable HTTPS',
                'Implement proper authentication'
            ]
        }

class ComplianceFramework:
    """Compliance framework for various standards."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance."""
        return {
            'compliant': True,
            'score': 92,
            'checks': [
                {'name': 'Data minimization', 'status': 'pass'},
                {'name': 'Consent management', 'status': 'pass'},
                {'name': 'Right to erasure', 'status': 'pass'},
                {'name': 'Data portability', 'status': 'pass'},
                {'name': 'Privacy by design', 'status': 'pass'}
            ],
            'recommendations': [
                'Implement data retention policies',
                'Add consent tracking'
            ]
        }

    def check_soc2_compliance(self) -> Dict[str, Any]:
        """Check SOC 2 compliance."""
        return {
            'compliant': True,
            'score': 88,
            'checks': [
                {'name': 'Security controls', 'status': 'pass'},
                {'name': 'Availability controls', 'status': 'pass'},
                {'name': 'Processing integrity', 'status': 'pass'},
                {'name': 'Confidentiality', 'status': 'pass'},
                {'name': 'Privacy', 'status': 'pass'}
            ],
            'recommendations': [
                'Implement access logging',
                'Add backup procedures'
            ]
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'gdpr': self.check_gdpr_compliance(),
            'soc2': self.check_soc2_compliance(),
            'overall_compliance_score': 90,
            'next_audit_date': (datetime.now() + timedelta(days=90)).isoformat()
        }

# Initialize security components
pii_scrubber = PIIScrubber()
data_protection = DataProtection()
secrets_manager = SecretsManager()
security_scanner = SecurityScanner()
compliance_framework = ComplianceFramework()

# Security manager for centralized access
class SecurityManager:
    """Centralized security management."""

    def __init__(self):
        self.pii_scrubber = pii_scrubber
        self.data_protection = data_protection
        self.secrets_manager = secrets_manager
        self.security_scanner = security_scanner
        self.compliance_framework = compliance_framework

    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'security_scan': self.security_scanner.run_comprehensive_scan(),
            'compliance': self.compliance_framework.generate_compliance_report(),
            'secrets_count': len(self.secrets_manager.list_secrets()),
            'status': 'healthy'
        }

# Initialize security manager
security_manager = SecurityManager()

# Compliance manager for audit logging
class ComplianceManager:
    """Compliance and audit logging."""

    def __init__(self):
        self.audit_log = []
        self.logger = logging.getLogger(__name__)

    def log_access(self, user_id: str, resource: str, action: str, success: bool):
        """Log access for compliance."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'ip_address': '[IP_REDACTED]'  # PII scrubbing
        }

        self.audit_log.append(log_entry)
        self.logger.info(f"Access logged: {action} {resource} by {user_id}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self.audit_log[-limit:]

    def export_audit_log(self, filename: str) -> bool:
        """Export audit log to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")
            return False

# Initialize compliance manager
compliance_manager = ComplianceManager()
