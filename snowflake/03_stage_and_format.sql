USE ROLE ACCOUNTADMIN;
USE DATABASE epc;
USE SCHEMA raw;

-- PARSE_HEADER is what allows MATCH_BY_COLUMN_NAME on CSV, so columns bind by
-- name rather than position. It is mutually exclusive with SKIP_HEADER.
-- Positional binding would be actively dangerous here: every column is STRING,
-- so a reordered export would land postcodes in the rating column silently.
--
-- NULL_IF is deliberately empty. EPC sentinels 'NO DATA!' and 'INVALID!' are
-- preserved verbatim: "surveyor recorded nothing" and "field absent from the
-- form" are different facts, and collapsing both to NULL here would destroy
-- that distinction irreversibly. Staging decides what each sentinel means.
CREATE OR REPLACE FILE FORMAT epc.raw.csv_format
  TYPE = CSV
  COMPRESSION = AUTO
  FIELD_DELIMITER = ','
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  PARSE_HEADER = TRUE
  EMPTY_FIELD_AS_NULL = TRUE
  NULL_IF = ()
  TRIM_SPACE = FALSE;

CREATE OR REPLACE STAGE epc.raw.epc_raw_stage
  STORAGE_INTEGRATION = epc_s3_int
  URL = 's3://epc-sw-pipeline-dev/raw/'
  FILE_FORMAT = epc.raw.csv_format;

-- Preflight. Must list files before any COPY INTO is attempted.
LIST @epc.raw.epc_raw_stage;
