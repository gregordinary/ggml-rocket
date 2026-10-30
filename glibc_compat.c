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
// glibc 2.43 then gave sqrtf, among 53 libm entries, a new GLIBC_2.43 version, so a build
// host on glibc >= 2.43 (e.g. Debian forky) also binds every sqrtf call to it.
//
// The forwarders below cover seven such symbols: __isoc23_strtol / __isoc23_strtoul /
// __isoc23_strtoll / __isoc23_strtoull / __isoc23_sscanf / __isoc23_fscanf / sqrtf. The
// statically-linked rocket-userspace driver uses strtol/strtoul/fscanf (its ROCKET_* env +
// sysfs parsing) and sqrtf (its norms and encoders), and the backend itself
// strtoll/strtoull/sscanf/sqrtf; which of them a given build imports depends on the
// toolchain, the glibc and the driver's source. The build checks the linked .so for any
// symbol versioned above GLIBC_2.34 after every link (the POST_BUILD check in
// CMakeLists.txt), and by hand it is:
//     objdump -T libggml-rocket.so | grep -E 'GLIBC_2\.(3[5-9]|[4-9][0-9])'
// Add a forwarder below for any new entry (a missing one is a load failure; an unused one
// is harmless).
//
// This unit is compiled as C11 (-std=gnu11, so its OWN strtol/scanf are NOT redirected
// and stay the classic GLIBC_2.17 symbols) and defines those seven entry points as thin
// forwarders. It is linked INTO libggml-rocket.so with hidden visibility, so:
//   * the backend's (and the static driver lib's) references to them resolve to these
//     definitions at link time — the .so no longer imports them from glibc,
//   * the symbols are not exported, so they cannot interpose on a newer host's libc,
//   * the .so's glibc floor drops to 2.34, which loads on bookworm.
// The forwarders are behaviourally identical for the integer/sysfs parsing the backend
// and driver do; the C23 additions (e.g. 0b binary literals) are unused.
//
// The shim lowers the glibc floor only. The .so also needs the libstdc++ of the GCC that
// built it, and bookworm's (GCC 12) stops at GLIBCXX_3.4.30 / CXXABI_1.3.13. GCC 13 and
// later give std::string::_M_replace an out-of-line cold path, _M_replace_cold, versioned
// GLIBCXX_3.4.31, and every string operation that goes through _M_replace instantiates it:
// assigning a C string (operator= or assign), insert, replace, and an operator+ whose right
// operand is a temporary string (it may insert the left one in front). So the backend builds
// its strings by append, and the POST_BUILD check also fails a .so that imports a libstdc++
// symbol above GCC 12's. Built with GCC 13.3 (x86-64) and GCC 16.2 (aarch64), that symbol was
// the only one; a newer GCC can add another, and the check names it.

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
unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}

// This unit is compiled with -fno-math-errno, so __builtin_sqrtf is the hardware square-root
// instruction and never calls back into sqrtf. The result is identical; only errno on a
// negative argument differs, and nothing in the backend or the driver reads it. Without the
// flag a negative argument WOULD call sqrtf, which is this function, so the build refuses
// it. GCC reads the builtin as a self-call before it expands it and warns of infinite
// recursion; the generated code is `fsqrt s0, s0; ret` on aarch64 (sqrtss on x86-64).
#if defined(__GNUC__) && !defined(__clang__) && !defined(__NO_MATH_ERRNO__)
#error "glibc_compat.c needs -fno-math-errno: its sqrtf must not call sqrtf"
#endif
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
#endif
__attribute__((visibility("hidden")))
float sqrtf(float x) {
    return __builtin_sqrtf(x);
}
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

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
