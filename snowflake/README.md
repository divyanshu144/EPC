# Snowflake setup

Run in numeric order. Raw table DDL and `COPY INTO` are generated at runtime by
`ingestion/load.py` from the CSV header — certificates have 93 columns and the
list is not worth hand-maintaining.

## Order of operations

1. **`01_setup.sql`** — replace `<your_snowflake_user>` first.
2. **AWS: create the bucket, policy and role.**
   ```bash
   aws s3 mb s3://epc-sw-pipeline-dev --region eu-west-2
   ```
   Policy `epc-s3-read` — least privilege, this bucket only. Note there is no
   `GetObjectVersion`: we never read a specific version, and leaving versioning
   off keeps stable keys from accumulating noncurrent copies.
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject"],
         "Resource": "arn:aws:s3:::epc-sw-pipeline-dev/raw/*"
       },
       {
         "Effect": "Allow",
         "Action": ["s3:ListBucket"],
         "Resource": "arn:aws:s3:::epc-sw-pipeline-dev",
         "Condition": {"StringLike": {"s3:prefix": ["raw/*"]}}
       }
     ]
   }
   ```
   Role `epc-snowflake-s3-role` with that policy attached, any placeholder trust
   policy for now.
3. **`02_storage_integration.sql`** — this file does NOT contain the AWS account id.

   It carries a `${AWS_ACCOUNT_ID}` placeholder, which is not valid Snowflake syntax.
   This repository is public, so the account id is supplied from the environment
   rather than committed. Render the runnable SQL and paste that into the console:

   ```bash
   export AWS_ACCOUNT_ID=<your 12-digit account id>
   make snowflake-sql
   ```

   Then run the rendered statements, then read the `DESCRIBE` output.
4. **AWS: replace the role's trust policy** with the two values from `DESCRIBE`:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {"AWS": "<STORAGE_AWS_IAM_USER_ARN>"},
         "Action": "sts:AssumeRole",
         "Condition": {
           "StringEquals": {"sts:ExternalId": "<STORAGE_AWS_EXTERNAL_ID>"}
         }
       }
     ]
   }
   ```
5. **`03_stage_and_format.sql`** — the final `LIST` must return files.

## Why the trust is bidirectional

Snowflake's IAM user assumes roles for many customers. Without the external ID,
anyone knowing that user's ARN could write a trust policy naming it and have
Snowflake — a trusted third party — read a bucket they are not entitled to.
That is the confused-deputy problem; the external ID is the shared secret that
scopes the assume-role to one specific integration.

## Region

Snowflake must be on **AWS / eu-west-2** to match the bucket, or every load pays
cross-region egress.
