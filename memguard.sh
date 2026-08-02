#!/usr/bin/env bash
# Log RAM/swap pressure so an unattended OOM is diagnosable after the fact,
# and record when swap grows past a threshold that predicts trouble.
cd "$HOME/Documents/ML_SOTA" || exit 0
read -r _ tot used free shared buff avail <<< "$(free -m | awk 'NR==2')"
swap=$(free -m | awk 'NR==3{print $3}')
arm=$(ps -eo cmd | grep -oE 'abl_[A-Za-z0-9]+' | head -1)
printf '%s avail=%sMi swap=%sMi arm=%s\n' "$(date '+%F %T')" "$avail" "$swap" "${arm:-none}" >> logs/memguard.log
# warn loudly in the log if we are close to the edge
# Warn only when RAM is genuinely scarce. Swap alone is a poor signal:
# once pages are swapped out they stay there, so a high figure can sit
# stale while gigabytes of RAM are free.
if [ "$avail" -lt 2000 ]; then
  echo "$(date '+%F %T') WARNING low memory: avail=${avail}Mi swap=${swap}Mi" >> logs/memguard.log
fi
