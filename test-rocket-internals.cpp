// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-internals.cpp — host-only gate for the backend's internal contracts that no
 * device run can see: the FA dump's file handling, the MB-budget parser, the weight cache's
 * identity, the MoE residency ledger across two live backends, the resident-weight budget
 * each residency variable sets at init, the H60 table's one-time
 * build under concurrent first use, and the device-route circuit breaker.
 *
 * It includes ggml-rocket.cpp directly, so every `static` helper is in reach without a test
 * seam in the shipped backend, and links ggml, ggml-cpu and rocketnpu in place of the
 * ggml-rocket library. Nothing here opens the NPU, so it runs on any host, CI included.
 */
#include "ggml-rocket.cpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <sys/wait.h>

static int g_fails = 0;
#define CHECK(cond, what) do {                                              \
        const bool ok_ = (cond);                                            \
        printf("  [%s] %s\n", ok_ ? "PASS" : "FAIL", what);                 \
        if (!ok_) g_fails++;                                                \
    } while (0)

// ---------------------------------------------------------------------------
// FA dump: a planted symlink at the next name is refused, and its target is untouched.
static void test_fadump(void) {
    printf("-- FA dump file handling\n");
    char tmpl[] = "/tmp/rk_fadump_test_XXXXXX";
    const char * dir = mkdtemp(tmpl);
    if (!dir) { CHECK(false, "mkdtemp"); return; }
    const std::string victim = std::string(dir) + "/victim";
    { FILE * f = fopen(victim.c_str(), "w"); fputs("keep", f); fclose(f); }

    const long op = 7;
    char name[128];
    snprintf(name, sizeof name, "%s/fa_op%04ld_pid%d_q.f16", dir, op, (int)getpid());
    CHECK(symlink(victim.c_str(), name) == 0, "plant a symlink at the dump's next name");

    const int dfd = rocket_fadump_open_dir(dir);
    CHECK(dfd >= 0, "open the dump directory");
    const char payload[16] = "model-data-here";
    CHECK(!rocket_fadump_write(dfd, dir, op, "q", payload, sizeof payload),
          "a write to the planted name is refused");
    struct stat st;
    CHECK(stat(victim.c_str(), &st) == 0 && st.st_size == 4, "the symlink's target is untouched");
    CHECK(rocket_fadump_write(dfd, dir, op, "k", payload, sizeof payload), "a fresh name is written");
    snprintf(name, sizeof name, "%s/fa_op%04ld_pid%d_k.f16", dir, op, (int)getpid());
    CHECK(stat(name, &st) == 0 && (st.st_mode & 0777) == 0600 && st.st_size == (off_t)sizeof payload,
          "the dump is a mode-0600 regular file of the full size");
    CHECK(!rocket_fadump_write(dfd, dir, op, "k", payload, sizeof payload),
          "an existing dump is never replaced");
    close(dfd);

    // The directory itself may not be a planted link.
    const std::string dlink = std::string(dir) + "/dirlink";
    CHECK(symlink(dir, dlink.c_str()) == 0, "plant a symlink to a directory");
    const int bad = rocket_fadump_open_dir(dlink.c_str());
    CHECK(bad < 0, "a symlinked dump directory is refused");
    if (bad >= 0) close(bad);

    unlink(dlink.c_str());
    snprintf(name, sizeof name, "%s/fa_op%04ld_pid%d_q.f16", dir, op, (int)getpid()); unlink(name);
    snprintf(name, sizeof name, "%s/fa_op%04ld_pid%d_k.f16", dir, op, (int)getpid()); unlink(name);
    unlink(victim.c_str());
    rmdir(dir);
}

// ---------------------------------------------------------------------------
// The MB parser: whole numbers only, auto is the caller's, the cap holds.
static void test_parser(void) {
    printf("-- MB budget parser\n");
    size_t b = 12345;
    struct { const char * in; bool ok; size_t want; } cases[] = {
        { "0",      true,  0 },
        { "1",      true,  (size_t)1 << 20 },
        { " 128 ",  true,  (size_t)128 << 20 },
        { "",       false, 0 },
        { "   ",    false, 0 },
        { "-1",     false, 0 },
        { "128junk",false, 0 },
        { "1e3",    false, 0 },
        { "auto",   false, 0 },
        { " auto ", false, 0 },
        { "99999999999999999999999", false, 0 },   // past strtoll: ERANGE
    };
    for (const auto & c : cases) {
        b = 12345;
        const bool ok = rocket_parse_mb_budget("TEST_KNOB", c.in, &b);
        char what[96];
        snprintf(what, sizeof what, "\"%s\" -> %s", c.in, c.ok ? "accepted" : "refused, default kept");
        CHECK(ok == c.ok && (ok ? b == c.want : b == 12345), what);
    }
    b = 0;
    CHECK(rocket_parse_mb_budget("TEST_KNOB", "9223372036854775807", &b) && b == ((size_t)(1ULL << 40) << 20),
          "the largest strtoll value is capped, not wrapped");
}

// ---------------------------------------------------------------------------
// Weight identity: the scheduler-copy grammar, duplicate names, changed contents.
static void test_weight_key(void) {
    printf("-- weight cache identity\n");
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + 8 * 4096 * 4, NULL, false };
    ggml_context * ctx = ggml_init(ip);
    auto mk = [&](const char * name, float fill) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 16);
        ggml_set_name(t, name);
        float * d = (float *)t->data;
        for (int i = 0; i < 64 * 16; i++) d[i] = fill + 0.001f * (float)i;
        return t;
    };
    ggml_tensor * a = mk("a#shared#0", 1.0f);
    ggml_tensor * b = mk("b#shared#1", 1.0f);
    CHECK(rocket_weight_key(a) != rocket_weight_key(b),
          "a model's own '#' names are not stripped: a#shared#0 and b#shared#1 stay two weights");
    CHECK(rocket_weight_key(a).rfind("a#shared#0@", 0) == 0, "a model name keeps every '#'");

    ggml_tensor * orig = mk("blk.0.attn_q.weight", 2.0f);
    ggml_tensor * copy = mk("ROCKET#blk.0.attn_q.weight#0", 2.0f);
    CHECK(rocket_weight_key(orig) == rocket_weight_key(copy),
          "a scheduler copy of identical bytes maps to its original's key");
    ggml_tensor * trunc = mk("ROCKET#blk.0.attn_q.weight", 2.0f);   // suffix lost to GGML_MAX_NAME
    CHECK(rocket_weight_key(trunc) == rocket_weight_key(orig), "a copy name truncated before its index");

    ggml_tensor * same = mk("blk.0.attn_q.weight", 3.0f);
    CHECK(rocket_weight_key(same) != rocket_weight_key(orig),
          "same name, same shape, different bytes -> two keys (a reload, a second model)");
    const std::string before = rocket_weight_key(orig);
    CHECK(rocket_weight_key(orig) == before, "an unchanged weight keeps its key across calls");
    ((float *)orig->data)[0] += 1.0f;
    CHECK(rocket_weight_key(orig) != before, "a leaf whose contents change gets a new key");

    ggml_tensor * leaf = mk("leaf_12", 1.0f);
    CHECK(rocket_weight_key(leaf).empty(), "a ggml auto-name stays uncacheable");
    ggml_free(ctx);

    ggml_init_params np = { 4 * ggml_tensor_overhead(), NULL, true };
    ggml_context * nctx = ggml_init(np);
    ggml_tensor * nd = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, 64, 16);
    ggml_set_name(nd, "blk.3.ffn_up.weight");
    CHECK(rocket_weight_key(nd) == "blk.3.ffn_up.weight", "a tensor with no data keys on its name");
    ggml_free(nctx);
}

// ---------------------------------------------------------------------------
// The MoE ledger across two live backends. Runs in a child: the ledger's budget is frozen
// for the life of the process at its first admission.
static int test_moe_ledger_child(void) {
    printf("-- MoE residency ledger, two live backends\n");
    // One big stack's copy (codes+scales ~34.6 MB, source ~35.7 MB) fits 80 MB; two do not.
    setenv("ROCKET_MOE_CACHE_MB", "80", 1);
    const int64_t BK = 2048, BN = 2048, BE = 8, G = 128;
    const size_t  bsrc = (size_t)BN * BK * 34 / 32;              // one Q8_0 expert's bytes
    const size_t  bcodes = (size_t)BE * ((size_t)BN * BK + (size_t)BN * (BK / G) * sizeof(float));
    // A small stack, ~8.8 MB a copy, for the different-key case.
    const int64_t SK = 1024, SN = 1024, SE = 4;
    const size_t  ssrc = (size_t)SN * SK * 34 / 32;
    const size_t  scodes = (size_t)SE * ((size_t)SN * SK + (size_t)SN * (SK / G) * sizeof(float));

    rk_moe_preflight_open(5);                    // backend A
    rk_moe_preflight_open(5);                    // backend B
    std::unordered_map<std::string, int> claims_a, claims_b;

    CHECK(rk_moe_preflight_admit("big", BK, BN, BE, (int)G, bsrc, nullptr, 0),
          "the pre-flight admits one copy of the big stack");
    const size_t ram0 = g_moe_pf.ram_used, src0 = g_moe_pf.src_used;
    CHECK(ram0 == bcodes && src0 == (size_t)BE * bsrc, "the first copy is charged codes and source");

    claims_a["big"] = rk_moe_preflight_claim("big");
    CHECK(claims_a["big"] == RK_MOE_CLAIMED && g_moe_pf.ram_used == ram0,
          "backend A takes the pre-flight's copy without a second charge");
    claims_b["big"] = rk_moe_preflight_claim("big");
    CHECK(claims_b["big"] == RK_MOE_REFUSED && g_moe_pf.ram_used == ram0,
          "backend B's second copy does not fit and is refused, not admitted past the budget");

    // Different keys: B places the small stack while A is live.
    CHECK(rk_moe_preflight_admit("small", SK, SN, SE, (int)G, ssrc, nullptr, 0),
          "the pre-flight admits the small stack");
    claims_b["small"] = rk_moe_preflight_claim("small");
    CHECK(claims_b["small"] == RK_MOE_CLAIMED, "backend B claims the small stack");
    const size_t ram1 = g_moe_pf.ram_used, src1 = g_moe_pf.src_used;
    CHECK(ram1 == bcodes + scodes && src1 == (size_t)BE * bsrc + (size_t)SE * ssrc,
          "both stacks are charged once");

    // A is freed while B lives: A's copy of the big stack goes back to the ledger.
    rk_moe_preflight_close(claims_a);
    CHECK(g_moe_pf.ram_used == scodes && g_moe_pf.src_used == (size_t)SE * ssrc,
          "freeing backend A returns its copy while backend B is still live");
    CHECK(g_moe_pf.live == 1, "one backend remains live");

    // A third backend on the same model now finds the big stack's copy free and re-charges it.
    rk_moe_preflight_open(5);
    std::unordered_map<std::string, int> claims_c;
    claims_c["big"] = rk_moe_preflight_claim("big");
    CHECK(claims_c["big"] == RK_MOE_CLAIMED && g_moe_pf.ram_used == scodes + bcodes
              && g_moe_pf.src_used == (size_t)SE * ssrc + (size_t)BE * bsrc,
          "a later backend's copy is charged again, codes and source");
    bool admitted = false;
    CHECK(rk_moe_preflight_decided("big", &admitted) && admitted,
          "the placement decision itself is unchanged by claims and releases");

    rk_moe_preflight_close(claims_b);
    rk_moe_preflight_close(claims_c);
    CHECK(g_moe_pf.ram_used == 0 && g_moe_pf.src_used == 0 && g_moe_pf.iova_used == 0
              && g_moe_pf.live == 0,
          "every claimed copy is returned when the last backend is freed");
    return g_fails;
}

static void test_moe_ledger(void) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) { const int f = test_moe_ledger_child(); fflush(stdout); _exit(f ? 1 : 0); }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        printf("  [FAIL] the MoE ledger child failed\n");
        g_fails++;
    }
}

// ---------------------------------------------------------------------------
// The resident-weight budget at init: the variable that turned residency on sizes it, and only
// an accepted ROCKET_CACHE_MB overrides it. One child per case, because the on/off readers
// cache the environment for the life of the process.
static const size_t BUDGET_AUTO = SIZE_MAX;   // want_mb marker: the auto branch ran
struct budget_case { const char * cache_mb, * f16, * quant; size_t want_mb; const char * what; };

static int test_init_budget_child(const budget_case & c) {
    const char * names[] = { "ROCKET_CACHE_MB", "ROCKET_F16_RESIDENT", "ROCKET_QUANT_RESIDENT",
                             "ROCKET_QUANT_RESIDENT_RESERVE_MB" };
    for (const char * n : names) unsetenv(n);
    if (c.cache_mb) setenv("ROCKET_CACHE_MB", c.cache_mb, 1);
    if (c.f16)      setenv("ROCKET_F16_RESIDENT", c.f16, 1);
    if (c.quant)    setenv("ROCKET_QUANT_RESIDENT", c.quant, 1);
    ggml_backend_t be = ggml_backend_rocket_init();
    const auto * ctx = (const ggml_backend_rocket_context *)be->context;
    if (c.want_mb == BUDGET_AUTO)
        CHECK(ctx->resident_floor_bytes == rocket_auto_budget_reserve(), c.what);
    else
        CHECK(ctx->cache_budget == c.want_mb << 20, c.what);
    ggml_backend_free(be);
    return g_fails;
}

static void test_init_budget(void) {
    printf("-- resident-weight budget at init\n");
    const budget_case cases[] = {
        { nullptr, nullptr, nullptr, 2048, "with nothing set the budget is the 2 GB default" },
        { nullptr, "1",     nullptr, 2048, "ROCKET_F16_RESIDENT=1 keeps the default budget" },
        { nullptr, "",      "3000",  3000, "an empty ROCKET_F16_RESIDENT leaves ROCKET_QUANT_RESIDENT's budget" },
        { nullptr, "0",     "3000",  3000, "ROCKET_F16_RESIDENT=0 leaves ROCKET_QUANT_RESIDENT's budget" },
        { nullptr, "",      "auto",  BUDGET_AUTO,
          "an empty ROCKET_F16_RESIDENT leaves ROCKET_QUANT_RESIDENT=auto's budget and floor" },
        { nullptr, "3000",  "5000",  3000, "ROCKET_F16_RESIDENT's budget wins when both are on" },
        { "",      "3000",  nullptr, 3000, "an empty ROCKET_CACHE_MB does not override the residency budget" },
        { "abc",   "3000",  nullptr, 3000, "a refused ROCKET_CACHE_MB does not override the residency budget" },
        { "4096",  "3000",  nullptr, 4096, "an accepted ROCKET_CACHE_MB overrides the residency budget" },
    };
    for (const auto & c : cases) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) { const int f = test_init_budget_child(c); fflush(stdout); _exit(f ? 1 : 0); }
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) g_fails++;
    }
}

// ---------------------------------------------------------------------------
// H60: concurrent first callers all see the finished table.
static void test_h60(void) {
    printf("-- H60 one-time build under concurrent first use\n");
    std::vector<std::thread> th;
    for (int i = 0; i < 8; i++) th.emplace_back([] { rk_build_H60(); });
    for (auto & t : th) t.join();
    bool hadamard = true;
    for (int i = 0; i < 60 && hadamard; i++)
        for (int j = 0; j < 60; j++) {
            double d = 0.0;
            for (int k = 0; k < 60; k++) d += (double)g_H60[i][k] * (double)g_H60[j][k];
            if (fabs(d - (i == j ? 60.0 : 0.0)) > 1e-9) { hadamard = false; break; }
        }
    CHECK(hadamard, "H60 * H60^T == 60 I after eight concurrent first calls");
}

// ---------------------------------------------------------------------------
// The circuit breaker: transient failures cost a retry, persistent ones are bypassed.
static void test_breaker(void) {
    printf("-- device-route circuit breaker\n");
    rk_breaker b;
    rk_breaker_note(b, ROCKET_E_DEVICE, "test");
    rk_breaker_note(b, ROCKET_E_DEVICE, "test");
    CHECK(rk_breaker_allow(b), "two failures leave the route in use");
    rk_breaker_note(b, 0, "test");
    rk_breaker_note(b, ROCKET_E_DEVICE, "test");
    rk_breaker_note(b, ROCKET_E_DEVICE, "test");
    CHECK(rk_breaker_allow(b), "a success in between resets the count");
    rk_breaker_note(b, ROCKET_E_TILING, "test");
    rk_breaker_note(b, ROCKET_E_UNSUPPORTED, "test");
    rk_breaker_note(b, ROCKET_E_RANGE, "test");
    CHECK(rk_breaker_allow(b), "structural declines and range verdicts do not count");
    rk_breaker_note(b, ROCKET_E_DEVICE, "test");       // third consecutive device failure
    int skipped = 0;
    while (!rk_breaker_allow(b)) skipped++;
    CHECK(skipped == 64, "three consecutive failures bypass the route for 64 calls, then probe");
    rk_breaker_note(b, -1, "test");                    // the probe fails
    skipped = 0;
    while (!rk_breaker_allow(b)) skipped++;
    CHECK(skipped == 128, "a failed probe reopens for twice as long");
    rk_breaker_note(b, 0, "test");                     // the probe succeeds
    CHECK(rk_breaker_allow(b) && rk_breaker_allow(b), "a successful probe closes the breaker");
}

int main(void) {
    test_fadump();
    test_parser();
    test_weight_key();
    test_moe_ledger();
    test_init_budget();
    test_h60();
    test_breaker();
    printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
