-- ============================================================================
-- COMPLETE SUPABASE DATABASE SETUP FOR PLANT RECOGNITION SYSTEM
-- ============================================================================
-- This file consolidates all database setup, functions, and maintenance scripts
-- Run this in Supabase SQL Editor to set up the complete database
-- 
-- Sections:
-- 1. Initial Setup (Extensions, Tables, Indexes)
-- 2. Core Functions (Vector Search)
-- 3. Advanced Functions (Plant Filters, Case-Insensitive Search)
-- 4. Optimization (Index Tuning)
-- 5. Maintenance Operations (Cleanup, Clear)
-- ============================================================================

-- ============================================================================
-- SECTION 1: INITIAL DATABASE SETUP
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing objects if recreating (uncomment if needed for clean setup)
-- DROP FUNCTION IF EXISTS match_hypernodes_combined(vector, float, int, float);
-- DROP FUNCTION IF EXISTS match_hypernodes_by_value(vector, float, int, text);
-- DROP FUNCTION IF EXISTS match_hypernodes_by_key(vector, float, int, text);
-- DROP TABLE IF EXISTS hypernodes CASCADE;

-- Create hypernodes table with proper schema
CREATE TABLE IF NOT EXISTS hypernodes (
    id BIGSERIAL PRIMARY KEY,
    
    -- HyperNode data
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    
    -- Embeddings (1024 dimensions for Vietnamese_Embedding)
    key_embedding vector(1024),
    value_embedding vector(1024),
    
    -- Metadata
    plant_name TEXT NOT NULL,
    section TEXT,
    chunk_id INTEGER DEFAULT 0,
    is_chunked BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);


-- Regular indexes for filtering
CREATE INDEX IF NOT EXISTS hypernodes_plant_name_idx ON hypernodes(plant_name);
CREATE INDEX IF NOT EXISTS hypernodes_section_idx ON hypernodes(section);

-- Case-insensitive plant name index for filtering
CREATE INDEX IF NOT EXISTS hypernodes_plant_name_lower_idx ON hypernodes(LOWER(plant_name));

-- ============================================================================
-- SECTION 3: CORE VECTOR SEARCH FUNCTIONS
-- ============================================================================

-- Function: Search by Key Embedding
-- Supports optional case-insensitive plant name filtering
CREATE OR REPLACE FUNCTION match_hypernodes_by_key(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 20,
    filter_plant_name text DEFAULT NULL
)
RETURNS TABLE (
    id int,
    key text,
    value text,
    plant_name text,
    section text,
    key_embedding vector(1024),
    value_embedding vector(1024),
    similarity float
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        hypernodes.id,
        hypernodes.key,
        hypernodes.value,
        hypernodes.plant_name,
        hypernodes.section,
        hypernodes.key_embedding,
        hypernodes.value_embedding,
        1 - (hypernodes.key_embedding <=> query_embedding) as similarity
    FROM hypernodes
    WHERE 1 - (hypernodes.key_embedding <=> query_embedding) > match_threshold
        AND (filter_plant_name IS NULL OR LOWER(hypernodes.plant_name) = LOWER(filter_plant_name))
    ORDER BY hypernodes.key_embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Function: Search by Value Embedding
-- Supports optional case-insensitive plant name filtering
CREATE OR REPLACE FUNCTION match_hypernodes_by_value(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 20,
    filter_plant_name text DEFAULT NULL
)
RETURNS TABLE (
    id int,
    key text,
    value text,
    plant_name text,
    section text,
    key_embedding vector(1024),
    value_embedding vector(1024),
    similarity float
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        hypernodes.id,
        hypernodes.key,
        hypernodes.value,
        hypernodes.plant_name,
        hypernodes.section,
        hypernodes.key_embedding,
        hypernodes.value_embedding,
        1 - (hypernodes.value_embedding <=> query_embedding) as similarity
    FROM hypernodes
    WHERE 1 - (hypernodes.value_embedding <=> query_embedding) > match_threshold
        AND (filter_plant_name IS NULL OR LOWER(hypernodes.plant_name) = LOWER(filter_plant_name))
    ORDER BY hypernodes.value_embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Function: Combined Key+Value Search
-- Weighted combination of key and value similarity
CREATE OR REPLACE FUNCTION match_hypernodes_combined(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 20,
    key_weight float DEFAULT 0.5
)
RETURNS TABLE (
    id bigint,
    key text,
    value text,
    plant_name text,
    section text,
    similarity float
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        id,
        key,
        value,
        plant_name,
        section,
        (
            key_weight * (1 - (key_embedding <=> query_embedding)) +
            (1 - key_weight) * (1 - (value_embedding <=> query_embedding))
        ) as similarity
    FROM hypernodes
    WHERE (
        key_weight * (1 - (key_embedding <=> query_embedding)) +
        (1 - key_weight) * (1 - (value_embedding <=> query_embedding))
    ) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

-- ============================================================================
-- SECTION 4: PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Set statement timeout for search functions to prevent long-running queries
ALTER FUNCTION match_hypernodes_by_key SET statement_timeout = '30s';
ALTER FUNCTION match_hypernodes_by_value SET statement_timeout = '30s';
ALTER FUNCTION match_hypernodes_combined SET statement_timeout = '30s';

-- Update table statistics for query optimizer
ANALYZE hypernodes;

-- ============================================================================
-- SECTION 5: MAINTENANCE OPERATIONS
-- ============================================================================

-- These are optional maintenance commands. Uncomment to use when needed.

-- -----------------------------------------------------------------------------
-- 5.1 Check Database Status
-- -----------------------------------------------------------------------------

-- Check total number of nodes
-- SELECT COUNT(*) as total_nodes FROM hypernodes;

-- Check data distribution
-- SELECT plant_name, COUNT(*) as node_count 
-- FROM hypernodes 
-- GROUP BY plant_name 
-- ORDER BY node_count DESC 
-- LIMIT 20;

-- -----------------------------------------------------------------------------
-- 5.2 Clean Duplicate Nodes (Run only if needed)
-- -----------------------------------------------------------------------------

-- Check for duplicates first
-- SELECT COUNT(*) as duplicate_count
-- FROM (
--     SELECT key, value, plant_name, COUNT(*) as cnt
--     FROM hypernodes
--     GROUP BY key, value, plant_name
--     HAVING COUNT(*) > 1
-- ) AS duplicates;

-- Delete duplicates (keeps lowest ID)
-- DELETE FROM hypernodes a
-- USING hypernodes b
-- WHERE a.id > b.id 
--   AND a.key = b.key 
--   AND a.value = b.value 
--   AND a.plant_name = b.plant_name;

-- -----------------------------------------------------------------------------
-- 5.3 Clear All Data (USE WITH CAUTION!)
-- -----------------------------------------------------------------------------

-- Truncate all data and reset auto-increment
-- WARNING: This deletes ALL data!
-- TRUNCATE TABLE hypernodes;

-- Verify deletion
-- SELECT COUNT(*) as remaining_nodes FROM hypernodes;

-- -----------------------------------------------------------------------------
-- 5.4 Rebuild Indexes (Run after large data changes)
-- -----------------------------------------------------------------------------

-- Rebuild vector indexes
-- REINDEX INDEX hypernodes_key_embedding_idx;
-- REINDEX INDEX hypernodes_value_embedding_idx;

-- Rebuild regular indexes
-- REINDEX INDEX hypernodes_plant_name_idx;
-- REINDEX INDEX hypernodes_section_idx;
-- REINDEX INDEX hypernodes_plant_name_lower_idx;

-- Update statistics after rebuild
-- ANALYZE hypernodes;

-- ============================================================================
-- SETUP COMPLETE!
-- ============================================================================
-- Next steps:
-- 1. Import your plant hypernode data using the import scripts
-- 2. Verify data with: SELECT COUNT(*) FROM hypernodes;
-- 3. Test vector search functions with sample queries
-- ============================================================================