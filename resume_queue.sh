#!/usr/bin/env bash
# Relaunch any queue driver that is not already running. Safe to run at any
# time and safe to run twice: every driver skips work whose checkpoint says it
# finished, and each one waits for its predecessors by argv[1], so restarting
# all eight cannot put two trainings on the GPU at once.
#
# Installed as an @reboot cron entry so a power cut resumes the queue without
# anyone logging in. To stop the queue deliberately, create the file STOP_QUEUE
# in the project root. (The older PAUSE file belongs to a watchdog that no
# longer exists and is deliberately left untouched.)
cd /mnt/ssd/ML_SOTA || exit 1
mkdir -p logs
exec >> logs/resume_queue.log 2>&1
echo "[$(date '+%F %T')] resume_queue invoked"
if [ -f STOP_QUEUE ]; then echo "  STOP_QUEUE present, doing nothing"; exit 0; fi

running() {   # is a driver with this argv[1] already alive?
  local p a1
  for p in $(pgrep -x bash 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
    case "$a1" in ./$1|*/$1) return 0 ;; esac
  done
  return 1
}

# run_hr1600.sh and run_compound.sh are deliberately absent: their cells are
# finished, and run_compound.sh waits only on the drivers that preceded it, so
# relaunching it would let it start a training alongside run_phase2.sh. Its one
# unfinished cell, abl_HR1280hp, is owned by run_protohp.sh instead.
for s in run_final.sh run_extras.sh run_protohp.sh run_labelnoise.sh \
         run_selftrain.sh run_bestof.sh; do
  [ -x "$s" ] || { echo "  $s missing or not executable, skipped"; continue; }
  if running "$s"; then
    echo "  $s already running"
  else
    setsid nohup "./$s" >> "logs/${s%.sh}.log" 2>&1 < /dev/null &
    echo "  $s launched (pid $!)"
    sleep 2
  fi
done
echo "[$(date '+%F %T')] resume_queue done"
