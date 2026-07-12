#!/usr/bin/env bash
# One-shot answer to "is anything billing right now?" for deepsculpt.
# GPU cost exists ONLY while a Cloud Run execution is running; schedulers
# themselves are free but keep creating executions until paused.
set -euo pipefail
PROJECT=garassino-ml

echo "=== RUNNING executions (each one = a GPU billing right now) ==="
for region in europe-west1 europe-west4; do
  for job in $(gcloud run jobs list --project $PROJECT --region $region --format "value(metadata.name)" 2>/dev/null); do
    gcloud run jobs executions list --job "$job" --project $PROJECT --region $region \
      --format "csv[no-heading](name, metadata.creationTimestamp, status.completionTime)" 2>/dev/null \
      | awk -F, -v r=$region '$3 == "" { print "  " r "  " $1 "  started " $2 }'
  done
done
echo "(nothing above = zero GPU spend right now)"

echo
echo "=== ENABLED schedulers (each keeps launching future slices) ==="
gcloud scheduler jobs list --project $PROJECT --location europe-west1 \
  --filter "state=ENABLED AND name~deepsculpt" --format "table[no-heading](name.basename(), schedule)" | sed 's/^/  /'
echo "(pause everything: for j in \$(gcloud scheduler jobs list --project $PROJECT --location europe-west1 --filter 'state=ENABLED AND name~deepsculpt' --format 'value(name.basename())'); do gcloud scheduler jobs pause \$j --project $PROJECT --location europe-west1; done)"

echo
echo "=== Always-on Cloud Run services in $PROJECT (should be none for deepsculpt) ==="
gcloud run services list --project $PROJECT --format "table[no-heading](metadata.name, status.url)" 2>/dev/null | sed 's/^/  /' || true
