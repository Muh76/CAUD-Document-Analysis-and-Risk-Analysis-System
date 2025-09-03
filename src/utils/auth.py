"""
Authentication and Security Utilities
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import hashlib

from src.config.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES


class AuthManager:
    """Authentication and security manager"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.secret_key = SECRET_KEY or "your-secret-key-here"
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

        # Mock user database (in production, use real database)
        self.users_db = {
            "admin": {
                "username": "admin",
                "email": "admin@legalai.com",
                "hashed_password": self.pwd_context.hash("admin123"),
                "full_name": "System Administrator",
                "role": "admin",
                "active": True,
            },
            "user": {
                "username": "user",
                "email": "user@legalai.com",
                "hashed_password": self.pwd_context.hash("user123"),
                "full_name": "Regular User",
                "role": "user",
                "active": True,
            },
        }

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """
        Authenticate user with username and password

        Args:
            username: Username
            password: Plain text password

        Returns:
            User data if authenticated, None otherwise
        """
        try:
            user = self.users_db.get(username)
            if not user:
                return None

            if not self.verify_password(password, user["hashed_password"]):
                return None

            if not user["active"]:
                return None

            return user

        except Exception as e:
            self.logger.error(f"Error authenticating user: {e}")
            return None

    def create_access_token(
        self, data: Dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            data: Data to encode in token
            expires_delta: Optional expiration time

        Returns:
            JWT token string
        """
        try:
            to_encode = data.copy()

            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    minutes=self.access_token_expire_minutes
                )

            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(
                to_encode, self.secret_key, algorithm=self.algorithm
            )

            return encoded_jwt

        except Exception as e:
            self.logger.error(f"Error creating access token: {e}")
            raise

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            Decoded token data if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")

            if username is None:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            self.logger.error(f"JWT error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error verifying token: {e}")
            return None

    def get_current_user(self, token: str) -> Optional[Dict]:
        """
        Get current user from token

        Args:
            token: JWT token string

        Returns:
            User data if valid token, None otherwise
        """
        try:
            payload = self.verify_token(token)
            if payload is None:
                return None

            username: str = payload.get("sub")
            if username is None:
                return None

            user = self.users_db.get(username)
            if user is None:
                return None

            return user

        except Exception as e:
            self.logger.error(f"Error getting current user: {e}")
            return None

    def encrypt_data(self, data: str) -> bytes:
        """
        Encrypt sensitive data

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        try:
            return self.cipher.encrypt(data.encode())
        except Exception as e:
            self.logger.error(f"Error encrypting data: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt sensitive data

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data
        """
        try:
            return self.cipher.decrypt(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Error decrypting data: {e}")
            raise

    def hash_data(self, data: str) -> str:
        """
        Hash data for integrity checking

        Args:
            data: Data to hash

        Returns:
            SHA256 hash
        """
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing data: {e}")
            raise

    def validate_access(self, user: Dict, resource: str, action: str) -> bool:
        """
        Validate user access to resource

        Args:
            user: User data
            resource: Resource to access
            action: Action to perform

        Returns:
            True if access allowed, False otherwise
        """
        try:
            # Role-based access control
            role = user.get("role", "user")

            # Admin has full access
            if role == "admin":
                return True

            # User permissions
            if role == "user":
                # Users can read and analyze contracts
                if action in ["read", "analyze"] and resource in [
                    "contracts",
                    "analysis",
                ]:
                    return True

                # Users cannot access admin functions
                if resource in ["users", "system", "admin"]:
                    return False

            return False

        except Exception as e:
            self.logger.error(f"Error validating access: {e}")
            return False

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        role: str = "user",
    ) -> bool:
        """
        Create new user

        Args:
            username: Username
            email: Email address
            password: Plain text password
            full_name: Full name
            role: User role

        Returns:
            True if user created successfully, False otherwise
        """
        try:
            # Check if user already exists
            if username in self.users_db:
                return False

            # Create new user
            hashed_password = self.get_password_hash(password)

            self.users_db[username] = {
                "username": username,
                "email": email,
                "hashed_password": hashed_password,
                "full_name": full_name,
                "role": role,
                "active": True,
                "created_at": datetime.utcnow().isoformat(),
            }

            self.logger.info(f"User created: {username}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return False

    def update_user(self, username: str, **kwargs) -> bool:
        """
        Update user information

        Args:
            username: Username
            **kwargs: Fields to update

        Returns:
            True if user updated successfully, False otherwise
        """
        try:
            if username not in self.users_db:
                return False

            user = self.users_db[username]

            # Update allowed fields
            allowed_fields = ["email", "full_name", "role", "active"]
            for field, value in kwargs.items():
                if field in allowed_fields:
                    user[field] = value

            user["updated_at"] = datetime.utcnow().isoformat()

            self.logger.info(f"User updated: {username}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating user: {e}")
            return False

    def delete_user(self, username: str) -> bool:
        """
        Delete user

        Args:
            username: Username to delete

        Returns:
            True if user deleted successfully, False otherwise
        """
        try:
            if username not in self.users_db:
                return False

            # Soft delete - mark as inactive
            self.users_db[username]["active"] = False
            self.users_db[username]["deleted_at"] = datetime.utcnow().isoformat()

            self.logger.info(f"User deleted: {username}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting user: {e}")
            return False

    def get_user_stats(self) -> Dict:
        """
        Get user statistics

        Returns:
            Dictionary with user statistics
        """
        try:
            total_users = len(self.users_db)
            active_users = sum(
                1 for user in self.users_db.values() if user.get("active", False)
            )
            admin_users = sum(
                1 for user in self.users_db.values() if user.get("role") == "admin"
            )

            return {
                "total_users": total_users,
                "active_users": active_users,
                "admin_users": admin_users,
                "regular_users": active_users - admin_users,
            }

        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {}
