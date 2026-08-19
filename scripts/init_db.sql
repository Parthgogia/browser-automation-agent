-- Runs once, on first creation of the Postgres data volume.
-- Table creation itself is handled by SQLAlchemy at application startup;
-- all this needs to do is make the vector type available beforehand.
CREATE EXTENSION IF NOT EXISTS vector;
