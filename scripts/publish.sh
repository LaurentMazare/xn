#!/usr/bin/env bash
# Publishes xn to crates.io from a fresh worktree of origin/main, with KleidiAI in it.
#
# The `kai` feature compiles KleidiAI's kernels from the `xn-core/third_party/kleidiai`
# submodule. A package made where the submodule is not checked out has none of them; its build
# script then turns the feature off with a warning that cargo does not show to dependents, so
# `kai` quietly does nothing. This makes the package in a clean worktree, where untracked files
# cannot slip in either, and refuses to publish one without the kernels.
#
#   scripts/publish.sh            check and dry run only
#   scripts/publish.sh --publish  publish
#
# XN_PUBLISH_REF packages another commit than origin/main, to try the script on a branch.
set -euo pipefail

publish=false
[[ "${1:-}" == "--publish" ]] && publish=true

repo=$(git rev-parse --show-toplevel)
git -C "$repo" fetch -q origin
dir=$(mktemp -d)
cleanup() { git -C "$repo" worktree remove --force "$dir"; }
trap cleanup EXIT
git -C "$repo" worktree add -q --detach "$dir" "${XN_PUBLISH_REF:-origin/main}"
cd "$dir"
git submodule update -q --init --depth 1 xn-core/third_party/kleidiai

version=$(cargo pkgid -p xn | sed 's/.*[@#]//')
if curl -sf -o /dev/null -A xn-publish "https://crates.io/api/v1/crates/xn/$version"; then
  echo "xn $version is already on crates.io: bump the version first" >&2
  exit 1
fi
if ! cargo package -p xn --list | grep -qx 'third_party/kleidiai/kai/kai_common.h'; then
  echo "the package has no KleidiAI kernels: refusing to publish" >&2
  exit 1
fi
# Verifies with `kai` on, so the kernels must build from the package's own files.
cargo publish -p xn --features kai --dry-run
if ! $publish; then
  echo
  echo "Dry run only: nothing was published. To publish xn $version, run"
  echo "  scripts/publish.sh --publish"
  echo "and not cargo publish, which packages your own checkout, with or without KleidiAI."
  exit 0
fi
cargo publish -p xn --features kai

# Read the crate back from crates.io: what was uploaded is what users get.
crate="https://static.crates.io/crates/xn/xn-$version.crate"
for _ in $(seq 30); do
  curl -sfL "$crate" -o "$dir/published.crate" && break
  sleep 10
done
if ! tar -tzf "$dir/published.crate" | grep -q "/third_party/kleidiai/kai/kai_common.h$"; then
  echo "xn $version on crates.io has no KleidiAI kernels: yank it (cargo yank --version $version xn)" >&2
  exit 1
fi
echo "Published xn $version with the KleidiAI kernels."
