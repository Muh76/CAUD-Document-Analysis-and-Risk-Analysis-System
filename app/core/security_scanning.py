"""
Security scanning and compliance framework.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import re

class SecurityScanner:
    """Security scanning utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scan_results = {}

    def run_dependency_scan(self) -> Dict[str, Any]:
        """Run dependency vulnerability scan."""
        try:
            # Run safety check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )

            if result.returncode == 0:
                vulnerabilities = []
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                except json.JSONDecodeError:
                    vulnerabilities = [{'error': 'Failed to parse safety output'}]

            return {
                'scan_type': 'dependency_vulnerability',
                'timestamp': datetime.utcnow().isoformat(),
                'vulnerabilities_found': len(vulnerabilities),
                'vulnerabilities': vulnerabilities,
                'status': 'PASS' if len(vulnerabilities) == 0 else 'FAIL'
            }

        except FileNotFoundError:
            return {
                'scan_type': 'dependency_vulnerability',
                'timestamp': datetime.utcnow().isoformat(),
                'error': 'Safety tool not installed',
                'status': 'ERROR'
            }
        except Exception as e:
            return {
                'scan_type': 'dependency_vulnerability',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'ERROR'
            }

    def run_code_security_scan(self) -> Dict[str, Any]:
        """Run code security scan with Bandit."""
        try:
            # Run bandit scan
            result = subprocess.run(
                ['bandit', '-r', 'app/', '-f', 'json'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )

            if result.returncode == 0:
                issues = []
            else:
                try:
                    bandit_output = json.loads(result.stdout)
                    issues = bandit_output.get('results', [])
                except json.JSONDecodeError:
                    issues = [{'error': 'Failed to parse bandit output'}]

            return {
                'scan_type': 'code_security',
                'timestamp': datetime.utcnow().isoformat(),
                'issues_found': len(issues),
                'issues': issues,
                'status': 'PASS' if len(issues) == 0 else 'FAIL'
            }

        except FileNotFoundError:
            return {
                'scan_type': 'code_security',
                'timestamp': datetime.utcnow().isoformat(),
                'error': 'Bandit tool not installed',
                'status': 'ERROR'
            }
        except Exception as e:
            return {
                'scan_type': 'code_security',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'ERROR'
            }

    def scan_for_hardcoded_secrets(self) -> Dict[str, Any]:
        """Scan for hardcoded secrets in code."""
        issues = []
        secret_patterns = [
            (r'password\s*=\s*["'][^"']+["']', 'Hardcoded password'),
            (r'api_key\s*=\s*["'][^"']+["']', 'Hardcoded API key'),
            (r'secret\s*=\s*["'][^"']+["']', 'Hardcoded secret'),
            (r'token\s*=\s*["'][^"']+["']', 'Hardcoded token'),
            (r'sk-[a-zA-Z0-9]{48}', 'OpenAI API key'),
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'[0-9a-fA-F]{32}', 'MD5 hash (potential secret)')
        ]

        for file_path in Path('app').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issues.append({
                            'file': str(file_path),
                            'line': content[:match.start()].count('\n') + 1,
                            'description': description,
                            'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group()
                        })
            except Exception as e:
                issues.append({
                    'file': str(file_path),
                    'error': f'Failed to scan file: {e}'
                })

        return {
            'scan_type': 'hardcoded_secrets',
            'timestamp': datetime.utcnow().isoformat(),
            'issues_found': len(issues),
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'FAIL'
        }

    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        self.logger.info("Starting comprehensive security scan...")

        scan_results = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'scans': {}
        }

        # Run all scans
        scan_results['scans']['dependency'] = self.run_dependency_scan()
        scan_results['scans']['code_security'] = self.run_code_security_scan()
        scan_results['scans']['hardcoded_secrets'] = self.scan_for_hardcoded_secrets()

        # Calculate overall status
        all_statuses = [scan['status'] for scan in scan_results['scans'].values()]
        if 'FAIL' in all_statuses:
            scan_results['overall_status'] = 'FAIL'
        elif 'ERROR' in all_statuses:
            scan_results['overall_status'] = 'ERROR'
        else:
            scan_results['overall_status'] = 'PASS'

        self.scan_results = scan_results
        return scan_results

class ComplianceFramework:
    """Compliance framework for various standards."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_checks = {
            'GDPR': self._check_gdpr_compliance,
            'SOC2': self._check_soc2_compliance,
            'HIPAA': self._check_hipaa_compliance,
            'PCI_DSS': self._check_pci_dss_compliance
        }

    def check_compliance(self, standard: str) -> Dict[str, Any]:
        """Check compliance for a specific standard."""
        if standard not in self.compliance_checks:
            return {
                'standard': standard,
                'status': 'ERROR',
                'error': f'Unknown compliance standard: {standard}'
            }

        try:
            return self.compliance_checks[standard]()
        except Exception as e:
            return {
                'standard': standard,
                'status': 'ERROR',
                'error': str(e)
            }

    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance."""
        checks = []

        # Check for data encryption
        checks.append({
            'check': 'Data encryption in transit',
            'status': 'PASS' if os.getenv('SSL_ENABLED') else 'FAIL',
            'description': 'HTTPS/TLS encryption for data transmission'
        })

        # Check for data encryption at rest
        checks.append({
            'check': 'Data encryption at rest',
            'status': 'PASS' if os.getenv('ENCRYPTION_ENABLED') else 'FAIL',
            'description': 'Database and file encryption'
        })

        # Check for data retention policy
        checks.append({
            'check': 'Data retention policy',
            'status': 'PASS' if os.getenv('DATA_RETENTION_DAYS') else 'FAIL',
            'description': 'Automated data deletion after retention period'
        })

        # Check for audit logging
        checks.append({
            'check': 'Audit logging',
            'status': 'PASS' if os.getenv('AUDIT_LOGGING_ENABLED') else 'FAIL',
            'description': 'Comprehensive audit trail'
        })

        passed_checks = len([c for c in checks if c['status'] == 'PASS'])
        overall_status = 'PASS' if passed_checks == len(checks) else 'FAIL'

        return {
            'standard': 'GDPR',
            'timestamp': datetime.utcnow().isoformat(),
            'status': overall_status,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} checks passed'
        }

    def _check_soc2_compliance(self) -> Dict[str, Any]:
        """Check SOC2 compliance."""
        checks = []

        # Check for access controls
        checks.append({
            'check': 'Access controls',
            'status': 'PASS' if os.getenv('AUTHENTICATION_ENABLED') else 'FAIL',
            'description': 'User authentication and authorization'
        })

        # Check for monitoring
        checks.append({
            'check': 'System monitoring',
            'status': 'PASS' if os.getenv('MONITORING_ENABLED') else 'FAIL',
            'description': 'System and application monitoring'
        })

        # Check for incident response
        checks.append({
            'check': 'Incident response',
            'status': 'PASS' if os.getenv('INCIDENT_RESPONSE_ENABLED') else 'FAIL',
            'description': 'Automated incident detection and response'
        })

        passed_checks = len([c for c in checks if c['status'] == 'PASS'])
        overall_status = 'PASS' if passed_checks == len(checks) else 'FAIL'

        return {
            'standard': 'SOC2',
            'timestamp': datetime.utcnow().isoformat(),
            'status': overall_status,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} checks passed'
        }

    def _check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        checks = []

        # Check for PHI protection
        checks.append({
            'check': 'PHI protection',
            'status': 'PASS' if os.getenv('PHI_PROTECTION_ENABLED') else 'FAIL',
            'description': 'Protected Health Information safeguards'
        })

        # Check for access logging
        checks.append({
            'check': 'Access logging',
            'status': 'PASS' if os.getenv('ACCESS_LOGGING_ENABLED') else 'FAIL',
            'description': 'Detailed access and modification logs'
        })

        passed_checks = len([c for c in checks if c['status'] == 'PASS'])
        overall_status = 'PASS' if passed_checks == len(checks) else 'FAIL'

        return {
            'standard': 'HIPAA',
            'timestamp': datetime.utcnow().isoformat(),
            'status': overall_status,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} checks passed'
        }

    def _check_pci_dss_compliance(self) -> Dict[str, Any]:
        """Check PCI DSS compliance."""
        checks = []

        # Check for card data protection
        checks.append({
            'check': 'Card data protection',
            'status': 'PASS' if os.getenv('CARD_DATA_PROTECTION') else 'FAIL',
            'description': 'Credit card data encryption and protection'
        })

        # Check for network security
        checks.append({
            'check': 'Network security',
            'status': 'PASS' if os.getenv('NETWORK_SECURITY_ENABLED') else 'FAIL',
            'description': 'Network segmentation and firewall protection'
        })

        passed_checks = len([c for c in checks if c['status'] == 'PASS'])
        overall_status = 'PASS' if passed_checks == len(checks) else 'FAIL'

        return {
            'standard': 'PCI_DSS',
            'timestamp': datetime.utcnow().isoformat(),
            'status': overall_status,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} checks passed'
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'standards': {}
        }

        for standard in self.compliance_checks.keys():
            report['standards'][standard] = self.check_compliance(standard)

        return report

# Global instances
security_scanner = SecurityScanner()
compliance_framework = ComplianceFramework()
