-- Single quotes only: Snowflake rejects double-quoted literals here.
--
-- ${AWS_ACCOUNT_ID} is a placeholder, not valid Snowflake syntax. Render this
-- file with `make snowflake-sql` and paste the output into the console. The
-- account id is kept out of version control because this repository is public.
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE STORAGE INTEGRATION epc_s3_int
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::${AWS_ACCOUNT_ID}:role/epc-snowflake-s3-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://epc-sw-pipeline-dev/raw/');

GRANT USAGE ON INTEGRATION epc_s3_int TO ROLE epc_dbt_role;

-- Copy STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID into the IAM
-- role's trust policy.
DESCRIBE INTEGRATION epc_s3_int;
