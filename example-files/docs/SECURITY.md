# 🔒 Security Guide

This guide outlines security best practices for the GPU-accelerated trading system.

## 📋 Table of Contents
1. [Overview](#overview)
2. [API Security](#api-security)
3. [Data Protection](#data-protection)
4. [Authentication](#authentication)
5. [Secure Configuration](#secure-configuration)
6. [Monitoring](#monitoring)
7. [Incident Response](#incident-response)

## 🛡️ Overview

Our security strategy ensures:
- API key protection
- Data encryption
- Secure authentication
- Safe configuration
- Continuous monitoring
- Incident handling

## 🔑 API Security

### 1. API Key Management
```python
# Bad - Hardcoded API keys
api_key = "1234567890abcdef"
api_secret = "abcdef1234567890"

# Good - Environment variables
import os
from dotenv import load_dotenv

load_dotenv()

class APIKeyManager:
    def __init__(self):
        self.api_key = os.getenv('TRADING_API_KEY')
        self.api_secret = os.getenv('TRADING_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not found in environment")
            
    def get_credentials(self) -> dict:
        """Get API credentials safely."""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret
        }
```

### 2. Request Signing
```python
import hmac
import hashlib
import time

class RequestSigner:
    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        
    def sign_request(
        self,
        method: str,
        endpoint: str,
        params: dict
    ) -> str:
        """Sign API request."""
        # Add timestamp
        timestamp = int(time.time() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature string
        query_string = '&'.join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        
        # Calculate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
```

### 3. Rate Limiting
```python
from time import sleep
from datetime import datetime, timedelta
from collections import deque

class RateLimiter:
    def __init__(
        self,
        max_requests: int = 1200,
        time_window: int = 60
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        
    def wait_if_needed(self):
        """Implement rate limiting."""
        now = datetime.now()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
            self.requests.popleft()
            
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            sleep_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if sleep_time > 0:
                sleep(sleep_time)
                
        # Add current request
        self.requests.append(now)
```

## 🔐 Data Protection

### 1. Data Encryption
```python
from cryptography.fernet import Fernet
import base64
import os

class DataEncryption:
    def __init__(self):
        self.key = self.load_or_generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def load_or_generate_key(self) -> bytes:
        """Load or generate encryption key."""
        key_file = '.encryption_key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(
            data.encode()
        ).decode()
        
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(
            encrypted_data.encode()
        ).decode()
```

### 2. Secure Storage
```python
import json
from pathlib import Path

class SecureStorage:
    def __init__(self):
        self.encryption = DataEncryption()
        self.storage_path = Path('.secure_storage')
        self.storage_path.mkdir(exist_ok=True)
        
    def save_sensitive_data(
        self,
        key: str,
        data: dict
    ) -> None:
        """Save encrypted sensitive data."""
        # Encrypt data
        encrypted_data = self.encryption.encrypt_data(
            json.dumps(data)
        )
        
        # Save to file
        file_path = self.storage_path / f"{key}.enc"
        with open(file_path, 'w') as f:
            f.write(encrypted_data)
            
    def load_sensitive_data(self, key: str) -> dict:
        """Load and decrypt sensitive data."""
        file_path = self.storage_path / f"{key}.enc"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for key: {key}")
            
        with open(file_path, 'r') as f:
            encrypted_data = f.read()
            
        return json.loads(
            self.encryption.decrypt_data(encrypted_data)
        )
```

## 🔐 Authentication

### 1. User Authentication
```python
import bcrypt
from typing import Optional

class UserAuth:
    def __init__(self):
        self.storage = SecureStorage()
        
    def create_user(
        self,
        username: str,
        password: str
    ) -> None:
        """Create new user with hashed password."""
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(
            password.encode(),
            salt
        )
        
        # Store user data
        user_data = {
            'username': username,
            'password_hash': hashed.decode(),
            'salt': salt.decode()
        }
        
        self.storage.save_sensitive_data(
            f"user_{username}",
            user_data
        )
        
    def verify_user(
        self,
        username: str,
        password: str
    ) -> bool:
        """Verify user credentials."""
        try:
            user_data = self.storage.load_sensitive_data(
                f"user_{username}"
            )
        except FileNotFoundError:
            return False
            
        stored_hash = user_data['password_hash'].encode()
        return bcrypt.checkpw(
            password.encode(),
            stored_hash
        )
```

### 2. Session Management
```python
import secrets
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = timedelta(hours=1)
        
    def create_session(self, username: str) -> str:
        """Create new session."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'username': username,
            'created_at': datetime.now()
        }
        return session_id
        
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username."""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
            
        if datetime.now() - session['created_at'] > self.session_timeout:
            del self.sessions[session_id]
            return None
            
        return session['username']
        
    def end_session(self, session_id: str) -> None:
        """End user session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
```

## ⚙️ Secure Configuration

### 1. Configuration Management
```python
from typing import Any, Dict
import yaml

class SecureConfig:
    def __init__(self):
        self.config = self.load_config()
        self.encryption = DataEncryption()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration safely."""
        config_path = Path('config.yml')
        
        if not config_path.exists():
            raise FileNotFoundError("Configuration file not found")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_sensitive_value(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get decrypted sensitive value."""
        encrypted_value = self.config.get(key)
        
        if not encrypted_value:
            return default
            
        try:
            return self.encryption.decrypt_data(encrypted_value)
        except Exception:
            return default
            
    def set_sensitive_value(
        self,
        key: str,
        value: str
    ) -> None:
        """Set encrypted sensitive value."""
        self.config[key] = self.encryption.encrypt_data(value)
        
        with open('config.yml', 'w') as f:
            yaml.safe_dump(self.config, f)
```

## 📊 Monitoring

### 1. Security Monitoring
```python
import logging
from datetime import datetime

class SecurityMonitor:
    def __init__(self):
        self.logger = self.setup_logger()
        self.suspicious_patterns = {
            'login_attempts': {},
            'api_calls': {},
            'data_access': {}
        }
        
    def setup_logger(self) -> logging.Logger:
        """Setup security logger."""
        logger = logging.getLogger('security')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('security.log')
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        
        logger.addHandler(handler)
        return logger
        
    def log_event(
        self,
        event_type: str,
        details: dict
    ) -> None:
        """Log security event."""
        self.logger.info(
            f"Security event: {event_type}",
            extra=details
        )
        
        # Check for suspicious patterns
        self.check_patterns(event_type, details)
        
    def check_patterns(
        self,
        event_type: str,
        details: dict
    ) -> None:
        """Check for suspicious patterns."""
        now = datetime.now()
        
        # Clean old entries
        for pattern in self.suspicious_patterns.values():
            pattern = {
                k: v for k, v in pattern.items()
                if now - v['timestamp'] < timedelta(hours=1)
            }
            
        # Track event
        if event_type == 'login_attempt':
            self.track_login_attempts(details)
        elif event_type == 'api_call':
            self.track_api_calls(details)
        elif event_type == 'data_access':
            self.track_data_access(details)
            
    def track_login_attempts(self, details: dict) -> None:
        """Track login attempts."""
        ip = details.get('ip')
        if not ip:
            return
            
        attempts = self.suspicious_patterns['login_attempts']
        
        if ip in attempts:
            attempts[ip]['count'] += 1
            if attempts[ip]['count'] >= 5:
                self.alert('Excessive login attempts', {
                    'ip': ip,
                    'count': attempts[ip]['count']
                })
        else:
            attempts[ip] = {
                'count': 1,
                'timestamp': datetime.now()
            }
```

### 2. Audit Logging
```python
from typing import Any, Dict
import json

class AuditLogger:
    def __init__(self):
        self.log_file = 'audit.log'
        
    def log_action(
        self,
        user: str,
        action: str,
        details: Dict[str, Any]
    ) -> None:
        """Log user action for audit."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'details': details
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def get_user_actions(
        self,
        user: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """Get user actions within timeframe."""
        actions = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(
                    entry['timestamp']
                )
                
                if (entry['user'] == user and
                    start_time <= timestamp <= end_time):
                    actions.append(entry)
                    
        return actions
```

## 🚨 Incident Response

### 1. Incident Handler
```python
from enum import Enum
from typing import Optional

class IncidentSeverity(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class IncidentHandler:
    def __init__(self):
        self.logger = logging.getLogger('incidents')
        self.active_incidents = {}
        
    def report_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        details: dict
    ) -> str:
        """Report security incident."""
        incident_id = secrets.token_urlsafe(16)
        
        incident = {
            'id': incident_id,
            'title': title,
            'severity': severity,
            'details': details,
            'status': 'open',
            'reported_at': datetime.now(),
            'resolved_at': None
        }
        
        self.active_incidents[incident_id] = incident
        
        # Log incident
        self.logger.warning(
            f"Security incident reported: {title}",
            extra={
                'incident_id': incident_id,
                'severity': severity.value,
                **details
            }
        )
        
        # Take immediate action based on severity
        self.handle_incident(incident)
        
        return incident_id
        
    def handle_incident(self, incident: dict) -> None:
        """Handle security incident."""
        severity = IncidentSeverity(incident['severity'])
        
        if severity == IncidentSeverity.CRITICAL:
            # Immediate actions for critical incidents
            self.emergency_shutdown()
            self.notify_team(incident)
        elif severity == IncidentSeverity.HIGH:
            # Actions for high severity
            self.restrict_access()
            self.notify_team(incident)
        else:
            # Monitor and log
            self.monitor_incident(incident)
            
    def resolve_incident(
        self,
        incident_id: str,
        resolution: str
    ) -> None:
        """Resolve security incident."""
        if incident_id not in self.active_incidents:
            raise ValueError(f"Unknown incident: {incident_id}")
            
        incident = self.active_incidents[incident_id]
        incident['status'] = 'resolved'
        incident['resolved_at'] = datetime.now()
        incident['resolution'] = resolution
        
        # Log resolution
        self.logger.info(
            f"Security incident resolved: {incident['title']}",
            extra={
                'incident_id': incident_id,
                'resolution': resolution
            }
        )
        
    def emergency_shutdown(self) -> None:
        """Implement emergency shutdown."""
        # Stop all trading activities
        trading_system.stop()
        
        # Revoke all active sessions
        session_manager.clear_sessions()
        
        # Disable API access
        api_manager.disable_access()
        
    def notify_team(self, incident: dict) -> None:
        """Notify security team."""
        message = f"""
        Security Incident Report
        -----------------------
        ID: {incident['id']}
        Title: {incident['title']}
        Severity: {incident['severity'].value}
        Time: {incident['reported_at']}
        
        Details:
        {json.dumps(incident['details'], indent=2)}
        """
        
        # Send notifications
        email_service.send_emergency(message)
        sms_service.send_alert(message)
```

### 2. Incident Recovery
```python
class IncidentRecovery:
    def __init__(self):
        self.backup_manager = BackupManager()
        self.system_state = SystemState()
        
    def create_recovery_point(self) -> str:
        """Create system recovery point."""
        # Backup configuration
        config_backup = self.backup_manager.backup_config()
        
        # Backup data
        data_backup = self.backup_manager.backup_data()
        
        # Save system state
        state_backup = self.system_state.save()
        
        recovery_point = {
            'timestamp': datetime.now(),
            'config': config_backup,
            'data': data_backup,
            'state': state_backup
        }
        
        return self.save_recovery_point(recovery_point)
        
    def restore_from_point(self, point_id: str) -> bool:
        """Restore system from recovery point."""
        try:
            # Load recovery point
            recovery_point = self.load_recovery_point(point_id)
            
            # Stop all activities
            trading_system.stop()
            
            # Restore configuration
            self.backup_manager.restore_config(
                recovery_point['config']
            )
            
            # Restore data
            self.backup_manager.restore_data(
                recovery_point['data']
            )
            
            # Restore system state
            self.system_state.restore(
                recovery_point['state']
            )
            
            return True
            
        except Exception as e:
            logging.error(
                f"Recovery failed: {str(e)}"
            )
            return False
``` 