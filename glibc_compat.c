// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
//
// glibc portability shim — opt-in via -DGGML_ROCKET_PORTABLE_GLIBC=ON.
//
// glibc 2.38 redirected the integer-parse and scanf families to versioned __isoc23_*
// symbols, so libggml-rocket.so compiled on a glibc >= 2.38 host (e.g. Debian
// trixie/forky) imports them at GLIBC_2.38 and then fails to load on an older runtime —
// e.g. a Debian 12 bookworm container (glibc 2.36) running llama.cpp / whisper.cpp.
//
// The .so pulls in exactly five such symbols: __isoc23_strtol / __isoc23_strtoul /
// __isoc23_strtoll / __isoc23_sscanf / __isoc23_fscanf — strtol/strtoul/fscanf from the
// statically-linked rocket-userspace driver (its ROCKET_* env + sysfs parsing) and
// strtoll/sscanf from the backend itself. They are the only > 2.36 symbols it imports.
// To re-derive after a toolchain or dependency bump:
//     nm -D --undefined-only libggml-rocket.so | grep __isoc23
// and add a forwarder below for any new entry (a missing one is a load failure; an
// unused one is harmless).
//
// This unit is compiled as C11 (-std=gnu11, so its OWN strtol/scanf are NOT redirected
// and stay the classic GLIBC_2.17 symbols) and defines those five __isoc23_* entry
// points as thin forwarders. It is linked INTO libggml-rocket.so with hidden
// visibility, so:
//   * the backend's (and the static driver lib's) references to __isoc23_* resolve to
//     these definitions at link time — the .so no longer imports them from glibc,
//   * the symbols are not exported, so they cannot interpose on a newer host's libc,
//   * the .so's glibc floor drops to 2.34, which loads on bookworm.
// The forwarders are behaviourally identical for the integer/sysfs parsing the backend
// and driver do; the C23 additions (e.g. 0b binary literals) are unused.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((visibility("hidden")))
long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

__attribute__((visibility("hidden")))
unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

__attribute__((visibility("hidden")))
long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

__attribute__((visibility("hidden")))
int __isoc23_sscanf(const char *str, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    int r = vsscanf(str, format, ap);
    va_end(ap);
    return r;
}

__attribute__((visibility("hidden")))
int __isoc23_fscanf(FILE *stream, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    int r = vfscanf(stream, format, ap);
    va_end(ap);
    return r;
}
