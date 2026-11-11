#!/bin/bash
# ================================================================
# Truncation Acquirer — LiME Memory Acquisition with Time Control
# ================================================================
# Acquires memory using LiME and terminates acquisition after a
# controlled fraction of the total expected acquisition time,
# simulating cloud-ephemeral truncation.
#
# Usage:
#   ./truncation_acquirer.sh --pct 50 --output /tmp/dump_50pct.lime
#   ./truncation_acquirer.sh --pct 100 --output /tmp/dump_full.lime
#   ./truncation_acquirer.sh --pct 25 --output /tmp/dump_25pct.lime --test
#
# Environment Variables:
#   LIME_MODULE_PATH  — Path to compiled lime.ko (default: /opt/lime/lime.ko)
#   FULL_ACQ_TIME     — Expected full acquisition time in seconds (default: auto-detect)
#   MEM_SIZE_MB       — System memory size in MB (default: auto-detect)
# ================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────

LIME_MODULE=${LIME_MODULE_PATH:-"/opt/lime/lime.ko"}
OUTPUT_DIR="/tmp/acquisition"
OUTPUT_FILE=""
TRUNCATION_PCT=100
TEST_MODE=false
LOG_FILE="/tmp/acquisition/acquisition.log"
METADATA_FILE=""

# LiME throughput estimate: ~2-4 GB/min on modern hardware
# Conservative: 2 GB/min = ~33 MB/s
LIME_THROUGHPUT_MBPS=${LIME_THROUGHPUT_MBPS:-33}

# ── Argument Parsing ──────────────────────────────────────────────

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --pct PCT          Truncation percentage (1-100, default: 100)
    --output FILE      Output .lime file path
    --output-dir DIR   Output directory (default: /tmp/acquisition)
    --lime-module PATH Path to lime.ko kernel module
    --test             Test mode (simulate without actual acquisition)
    --help             Show this help message

Examples:
    $0 --pct 50 --output /tmp/dump_50pct.lime
    $0 --pct 100 --output-dir /data/dumps/
    $0 --pct 25 --test
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --pct)         TRUNCATION_PCT="$2"; shift 2 ;;
        --output)      OUTPUT_FILE="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --lime-module) LIME_MODULE="$2"; shift 2 ;;
        --test)        TEST_MODE=true; shift ;;
        --help)        usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${OUTPUT_DIR}/memdump_${TRUNCATION_PCT}pct_${TIMESTAMP}.lime"
fi

METADATA_FILE="${OUTPUT_FILE%.lime}_metadata.json"
LOG_FILE="${OUTPUT_DIR}/acquisition_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# ── Detect System Memory ─────────────────────────────────────────

if [[ -z "${MEM_SIZE_MB:-}" ]]; then
    MEM_SIZE_MB=$(free -m | awk '/^Mem:/{print $2}')
fi

# Calculate expected full acquisition time
if [[ -z "${FULL_ACQ_TIME:-}" ]]; then
    FULL_ACQ_TIME=$(echo "scale=0; $MEM_SIZE_MB / $LIME_THROUGHPUT_MBPS" | bc)
    # Minimum 5 seconds
    if [[ "$FULL_ACQ_TIME" -lt 5 ]]; then
        FULL_ACQ_TIME=5
    fi
fi

# Calculate truncation time
TRUNC_TIME=$(echo "scale=2; $FULL_ACQ_TIME * $TRUNCATION_PCT / 100" | bc)
TRUNC_TIME_INT=$(echo "scale=0; $TRUNC_TIME / 1" | bc)

# Ensure minimum 1 second
if [[ "$TRUNC_TIME_INT" -lt 1 ]]; then
    TRUNC_TIME_INT=1
fi

# ── Log Configuration ────────────────────────────────────────────

log "================================================================"
log "TRUNCATION ACQUIRER — Memory Acquisition"
log "================================================================"
log "System memory:      ${MEM_SIZE_MB} MB"
log "LiME module:        ${LIME_MODULE}"
log "Output file:        ${OUTPUT_FILE}"
log "Truncation:         ${TRUNCATION_PCT}%"
log "Full acq time est:  ${FULL_ACQ_TIME}s"
log "Truncation time:    ${TRUNC_TIME_INT}s"
log "Test mode:          ${TEST_MODE}"
log "Timestamp:          ${TIMESTAMP}"
log "================================================================"

# ── Acquisition ──────────────────────────────────────────────────

ACQ_START=$(date +%s%N)

if [[ "$TEST_MODE" == "true" ]]; then
    log "[TEST] Simulating acquisition..."
    log "[TEST] Would load lime.ko with path=${OUTPUT_FILE} format=lime"
    log "[TEST] Would wait ${TRUNC_TIME_INT}s then kill acquisition"
    
    # Create a simulated partial dump
    EXPECTED_SIZE_MB=$(echo "scale=0; $MEM_SIZE_MB * $TRUNCATION_PCT / 100" | bc)
    # Create a small test file instead of actual memory dump
    dd if=/dev/urandom of="$OUTPUT_FILE" bs=1M count=1 2>/dev/null
    
    log "[TEST] Created test file: $(du -h "$OUTPUT_FILE" | cut -f1)"
    sleep 1
else
    log "Loading LiME kernel module..."
    
    # Verify module exists
    if [[ ! -f "$LIME_MODULE" ]]; then
        log "ERROR: LiME module not found at ${LIME_MODULE}"
        exit 1
    fi
    
    # Load LiME — starts acquisition immediately
    insmod "$LIME_MODULE" "path=${OUTPUT_FILE} format=lime" &
    LIME_PID=$!
    
    log "LiME loaded (PID: ${LIME_PID}), acquiring for ${TRUNC_TIME_INT}s..."
    
    # Wait for truncation window then kill
    sleep "$TRUNC_TIME_INT"
    
    # Force stop acquisition
    log "Truncation time reached — stopping acquisition..."
    rmmod lime 2>/dev/null || true
    kill "$LIME_PID" 2>/dev/null || true
    wait "$LIME_PID" 2>/dev/null || true
    
    log "Acquisition stopped."
fi

ACQ_END=$(date +%s%N)
ACQ_ELAPSED_MS=$(( (ACQ_END - ACQ_START) / 1000000 ))

# ── Verify Output ─────────────────────────────────────────────────

if [[ -f "$OUTPUT_FILE" ]]; then
    FILE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo 0)
    FILE_SIZE_MB=$(echo "scale=2; $FILE_SIZE / 1048576" | bc)
    EXPECTED_FULL_SIZE_MB=$MEM_SIZE_MB
    ACTUAL_PCT=$(echo "scale=1; $FILE_SIZE_MB * 100 / $EXPECTED_FULL_SIZE_MB" | bc 2>/dev/null || echo "N/A")
    
    log "Output file:     ${OUTPUT_FILE}"
    log "File size:       ${FILE_SIZE_MB} MB"
    log "Expected full:   ${EXPECTED_FULL_SIZE_MB} MB"
    log "Actual capture:  ${ACTUAL_PCT}%"
    log "Elapsed time:    ${ACQ_ELAPSED_MS} ms"
else
    log "WARNING: Output file not created"
    FILE_SIZE=0
    FILE_SIZE_MB=0
    ACTUAL_PCT=0
fi

# ── Write Metadata ────────────────────────────────────────────────

cat > "$METADATA_FILE" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "hostname": "$(hostname)",
    "system_memory_mb": ${MEM_SIZE_MB},
    "truncation_pct_nominal": ${TRUNCATION_PCT},
    "truncation_time_seconds": ${TRUNC_TIME_INT},
    "full_acquisition_time_estimate_seconds": ${FULL_ACQ_TIME},
    "actual_elapsed_ms": ${ACQ_ELAPSED_MS},
    "output_file": "${OUTPUT_FILE}",
    "output_size_bytes": ${FILE_SIZE:-0},
    "output_size_mb": ${FILE_SIZE_MB:-0},
    "actual_capture_pct": "${ACTUAL_PCT:-0}",
    "lime_module": "${LIME_MODULE}",
    "lime_throughput_mbps": ${LIME_THROUGHPUT_MBPS},
    "test_mode": ${TEST_MODE},
    "kernel_version": "$(uname -r)",
    "status": "success"
}
EOF

log "Metadata written to: ${METADATA_FILE}"
log "Acquisition complete."
