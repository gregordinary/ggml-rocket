// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
//
// rocket_raii.h -- small RAII helpers for the C++ backend to hold C-style
// librocketnpu handles (packed-weight BOs, contexts, fds) leak-free across the
// exception/error paths of this layer.
//
// The backend's normal teardown (ggml_backend_rocket_free) frees every handle in
// a fixed order that is load-bearing: resident weights hold BOs on their context's
// fds, so they must be freed BEFORE the context. That ordered teardown stays
// explicit. What it cannot cover is a handle that is acquired but not yet owned by
// its cache when a std::bad_alloc unwinds through the acquiring scope -- e.g. a
// rocket_*_weights_pack() that succeeds, followed by an unordered_map insert
// (rehash / key copy) that throws before the map takes ownership. scope_guard
// closes exactly that window: free-on-scope-exit unless dismiss()ed once ownership
// has transferred.
#pragma once

#include <utility>

namespace rocketraii {

// Runs a cleanup functor on scope exit unless dismiss() is called first. Non-movable
// and non-copyable: it guards one specific handle over one specific scope. Construct
// via C++17 CTAD, e.g. rocketraii::scope_guard g([&]{ rocket_weights_free(dev, w); });
template <class F>
class scope_guard {
public:
    explicit scope_guard(F f) : f_(std::move(f)) {}
    ~scope_guard() { if (live_) f_(); }
    void dismiss() noexcept { live_ = false; }

    scope_guard(const scope_guard &) = delete;
    scope_guard(scope_guard &&) = delete;
    scope_guard &operator=(const scope_guard &) = delete;
    scope_guard &operator=(scope_guard &&) = delete;

private:
    F    f_;
    bool live_ = true;
};

}  // namespace rocketraii
