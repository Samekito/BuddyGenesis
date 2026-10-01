-- One-use links sent by email (buddy/accounts.py): sign-up confirmation and password reset.
-- Only a SHA-256 of each token is stored, so a leaked database cannot be used to open a link.
-- A sign-up token carries the pending name and password hash: the account is created only when
-- the owner of the inbox clicks the link, so nobody can claim someone else's address.
CREATE TABLE IF NOT EXISTS email_tokens (
    "tokenHash" TEXT PRIMARY KEY,
    "purpose" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "name" TEXT,
    "passwordHash" TEXT,
    "createdAt" TEXT NOT NULL,
    "expiresAt" TEXT NOT NULL,
    "usedAt" TEXT
);

CREATE INDEX IF NOT EXISTS email_tokens_email_purpose ON email_tokens ("email", "purpose")
