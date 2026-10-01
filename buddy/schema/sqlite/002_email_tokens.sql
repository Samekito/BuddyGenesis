-- SQLite port of postgres/002_email_tokens.sql (identical: every column is TEXT).
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
