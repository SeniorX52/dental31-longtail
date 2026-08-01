#!/usr/bin/env bash
# Log RAM/swap pressure so an unattended OOM is diagnosable after the fact,
# and record when swap grows past a threshold that predicts trouble.
cd "$HOME/Documents/ML_SOTA" || exit 0
read -r _ tot used free shared buff avail <<< "$(free -m | awk 'NR==2')"
swap=$(free -m | awk 'NR==3{print $3}')
arm=$(ps -eo cmd | grep -oE 'abl_[A-Za-z0-9]+' | head -1)
printf '%s avail=%sMi swap=%sMi arm=%s\n' "$(date '+%F %T')" "$avail" "$swap" "${arm:-none}" >> logs/memguard.log
# warn loudly in the log if we are close to the edge
if [ "$avail" -lt 1500 ] || [ "$swap" -gt 1500 ]; then
  echo "$(date '+%F %T') WARNING low memory: avail=${avail}Mi swap=${swap}Mi" >> logs/memguard.log
fi
