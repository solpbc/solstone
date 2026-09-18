# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import unittest

from solstone_platform.redact import redact_sensitive_text


class TestRedact(unittest.TestCase):
    def test_redact_auth_header(self):
        sample = "Error sending request: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host, Signature=abcdef123456"
        redacted = redact_sensitive_text(sample)
        self.assertNotIn("AKIDEXAMPLE", redacted)
        self.assertNotIn("abcdef123456", redacted)
        self.assertIn("AWS4-HMAC-SHA256 Credential=[REDACTED]", redacted)

    def test_redact_minisign_secret(self):
        sample = "Key dump:\nuntrusted comment: minisign secret key\nRWRkZXNrdG9wLWtleQo=\nDone."
        redacted = redact_sensitive_text(sample)
        self.assertNotIn("RWRkZXNrdG9wLWtleQo=", redacted)
        self.assertIn("[REDACTED_MINISIGN_SECRET_KEY]", redacted)

    def test_redact_session_token(self):
        sample = "Headers: x-amz-security-token: AQoDYXdzEJr111222333444555666777888999"
        redacted = redact_sensitive_text(sample)
        self.assertNotIn("AQoDYXdzEJr111222333444555666777888999", redacted)
        self.assertIn("[REDACTED_SESSION_TOKEN]", redacted)

    def test_redact_named_secrets(self):
        sample = "Config error: secret_key = mySecretPassphrase123, passphrase: superSecretPassword"
        redacted = redact_sensitive_text(sample)
        self.assertNotIn("mySecretPassphrase123", redacted)
        self.assertNotIn("superSecretPassword", redacted)


if __name__ == "__main__":
    unittest.main()
