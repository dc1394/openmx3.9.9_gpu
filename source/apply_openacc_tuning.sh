#!/usr/bin/env bash
# ==============================================================
#  apply_openacc_tuning.sh   (2025-06-07)
#  *Only* OpenACC pragmas are replaced.  All other lines stay put.
#  Usage:  bash apply_openacc_tuning.sh  Divide_Conquer.c
# ==============================================================

set -euo pipefail
F=${1:-Divide_Conquer.c}
[[ -f "$F" ]] || { echo "File $F not found"; exit 1; }

cp "$F" "${F}.bak_acc_tuning"

# 1) kernels → parallel loop gang ...
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+kernels[^\n]*}{
    #pragma acc parallel loop gang vector_length(128) tile(32,32) async(1) /* tuned */
  }gmi
'       -i "$F"

# 2) loop independent collapse(2) → gang async(1) collapse(2) vector_length(128)
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+loop[ \t]+independent[^\n]*collapse\(2\)[^\n]*}{
    #pragma acc parallel loop gang async(1) collapse(2) vector_length(128) /* tuned */
  }gmi
'       -i "$F"

# 3) loop independent → loop vector independent
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+loop[ \t]+independent\b}{
    #pragma acc loop vector independent
  }gmi
'       -i "$F"

# 4) reduction(...) の語法を vector に揃える
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+loop[^\n]*reduction\(\+\s*:\s*}{
    #pragma acc loop vector reduction(+: 
  }gmi
'       -i "$F"

# 5) data copyin(...) → present_or_copyin(...  async(1) を追加
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+data[ \t]+copyin\(([^)]*)\)}{
    #pragma acc data present_or_copyin($1) async(1)
  }gmi
'       -i "$F"

# 6) data copy(...) → present_or_copy(...)
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+data[ \t]+copy\(([^)]*)\)}{
    #pragma acc data present_or_copy($1) async(1)
  }gmi
'       -i "$F"

# 7) data create(...) → present_or_create(...)
perl -0777 -pe '
  s{^[ \t]*#pragma[ \t]+acc[ \t]+data[ \t]+create\(([^)]*)\)}{
    #pragma acc data present_or_create($1) async(1)
  }gmi
'       -i "$F"

# 8) update self(ko..., C...) → 分割＋wait
perl -0777 -pe '
  s{#pragma[ \t]+acc[ \t]+update[ \t]+self\(\s*ko\[0[^\)]*\],\s*C\[0[^\)]*\]\)}{
#pragma acc update self(ko[0:NUM+1]) async(1)\n#pragma acc update self(C[0:NUM+1][0:NUM+1]) async(1)\n#pragma acc wait(1)
  }g
'       -i "$F"

echo "OpenACC tuning applied.  Backup saved as ${F}.bak_acc_tuning"
