# Dependencies of the prplOS (musl) build

Every external source the musl build of ort-server fetches, pinned, with how to supply it from a local
directory for an offline build (OpenWrt: everything fetched in the download step, then no network), and its
licence. CPU only: no GPU/NPU SDK is fetched or linked. **No dependency blocks shipping its source in a public
feed** (all are MIT, BSD, Apache-2.0, MPL-2.0, BSL-1.0, Unicode-3.0 or public domain); see the flags at the end.

The engine itself: https://github.com/ianbmacdonald/ort-server branch `feed/ort-0.4.0`, on top of
https://github.com/lemonade-sdk/ort-server tag `0.4.0` (`c9f193f`). Apache-2.0.

## 1. ort-server (CMake FetchContent, `CMakeLists.txt`)

| Name | URL | Pin | Licence | Offline switch |
|---|---|---|---|---|
| cpp-httplib | https://github.com/yhirose/cpp-httplib.git | `a7bc00e3307fecdb4d67545e93be7b88cfb1e186` (v0.18.3) | MIT | `-DFETCHCONTENT_SOURCE_DIR_HTTPLIB=<dir>` |
| nlohmann/json | https://github.com/nlohmann/json.git | `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` (v3.11.3) | MIT | `-DFETCHCONTENT_SOURCE_DIR_JSON=<dir>` |
| ONNX Runtime (prebuilt, glibc/Windows/macOS only) | GitHub release archive | `ONNXRUNTIME_VERSION` 1.27.0, sha256 per archive in `CMakeLists.txt` | MIT | not fetched for musl: `-DORT_SERVER_ORT_ROOT=<root>` (section 3) |

Vendored in the repository (nothing to fetch): `third_party/stb/stb_image.h` v2.30 (nothings/stb
`013ac3beddff3dbffafd5177e7972067cd2b5083`, sha256 in `third_party/stb/README`; MIT or public domain) and
`third_party/tok_ffi` (Apache-2.0, this project).

## 2. Rust crates of `third_party/tok_ffi` (cargo, `Cargo.lock`, built `--locked`)

The build runs `cargo build --release --locked --target <triple>` (`-DORT_SERVER_TOK_FFI_TARGET=x86_64-unknown-linux-musl`
or `aarch64-unknown-linux-musl`); it needs a Rust toolchain with that target's standard library
(`rustup target add <triple>`; built here with rustc 1.98.1) and the prplOS cross compiler in
`CC_<triple>` / `AR_<triple>` for `onig_sys`, which compiles the vendored Oniguruma C sources.

Offline: vendor the crates once, then build with the network off:

    cargo vendor --locked --manifest-path third_party/tok_ffi/Cargo.toml vendor/
    # $CARGO_HOME/config.toml (or .cargo/config.toml next to the build):
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "<abs path>/vendor"
    # and export CARGO_NET_OFFLINE=true

Each crate is a crates.io download, `https://static.crates.io/crates/<name>/<name>-<version>.crate`, whose
sha256 is the `checksum` in `Cargo.lock`. The crates linked into the musl build (normal and build
dependencies reachable from `tok_ffi` for the musl target; `Cargo.lock` lists 80, some only for other targets):

| Crate | Version | Licence | sha256 (`Cargo.lock`) |
|---|---|---|---|
| `ahash` | 0.8.12 | MIT OR Apache-2.0 | `5a15f179cd60c4584b8a8c596927aadc462e27f2ca70c04e0071964a73ba7a75` |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | `c982642fa9e8606056828ee9a8505737230110bb1099153c79efe865c59d12ba` |
| `base64` | 0.13.1 | MIT/Apache-2.0 | `9e1b586273c5702936fe7b7d6896644d8be71e6314cfe09d3167c95f712589e8` |
| `bitflags` | 2.13.2 | MIT OR Apache-2.0 | `3ded4057c258ba199e2d26386d3af3780957ecaee6c4ef4041c6b4b8b97c0b06` |
| `castaway` | 0.2.4 | MIT | `dec551ab6e7578819132c713a93c022a05d60159dc86e7a7050223577484c55a` |
| `cc` | 1.5.1 | MIT OR Apache-2.0 | `f360145194ee8e21db5ee7f3fcd4fe52210864c75c985dae33218202c8bbe040` |
| `cfg-if` | 1.0.5 | MIT OR Apache-2.0 | `4e7648175b45a9a48536d676f68d918270699102aa8dab5496df06904c914600` |
| `compact_str` | 0.9.1 | MIT | `9dfdd1c2274d9aa354115b09dc9a901d6c5576818cdf70d14cae2bdb47df00ab` |
| `crossbeam-deque` | 0.8.8 | MIT OR Apache-2.0 | `622f3fc73690be383c7214310406f28a90e6edeadc3cea882f9d71e495b9711a` |
| `crossbeam-epoch` | 0.9.21 | MIT OR Apache-2.0 | `dc74980687109a3b14c72fd458107bf0baa1da1a1a805e178d15501ba9b86d9d` |
| `crossbeam-utils` | 0.8.23 | MIT OR Apache-2.0 | `a31eee39dddec8330830986fcd7625edb5a24ec90ea038215273bbc3adb08ac6` |
| `darling` | 0.20.11 | MIT | `fc7f46116c46ff9ab3eb1597a45688b6715c6e628b5c133e288e709a29bcb4ee` |
| `darling_core` | 0.20.11 | MIT | `0d00b9596d185e565c2207a0b01f8bd1a135483d02d9b7b0a54b11da8d53412e` |
| `darling_macro` | 0.20.11 | MIT | `fc34b93ccb385b40dc71c6fceac4b2ad23662c7eeb248cf10d529b7e055b6ead` |
| `dary_heap` | 0.3.9 | MIT OR Apache-2.0 | `8b1e3a325bc115f096c8b77bbf027a7c2592230e70be2d985be950d3d5e60ebe` |
| `derive_builder` | 0.20.2 | MIT OR Apache-2.0 | `507dfb09ea8b7fa618fcf76e953f4f5e192547945816d5358edffe39f6f94947` |
| `derive_builder_core` | 0.20.2 | MIT OR Apache-2.0 | `2d5bcf7b024d6835cfb3d473887cd966994907effbe9227e8c8219824d06c4e8` |
| `derive_builder_macro` | 0.20.2 | MIT OR Apache-2.0 | `ab63b0e2bf4d5928aff72e83a7dace85d7bba5fe12dcc3c5a572d78caffd3f3c` |
| `either` | 1.18.0 | MIT OR Apache-2.0 | `252afb9ae5eaa683babdc6a068b3f5726eb19e05070c731f9b2a23a7c3e8ed34` |
| `esaxx-rs` | 0.1.10 | Apache-2.0 | `d817e038c30374a4bcb22f94d0a8a0e216958d4c3dcde369b1439fec4bdda6e6` |
| `find-msvc-tools` | 0.1.14 | MIT OR Apache-2.0 | `aedcfb3409746eddb02b9e19ebda1c3394f759a152e48ee875a0844d1b955484` |
| `fnv` | 1.0.7 | Apache-2.0 / MIT | `3f9eec918d3f24069decb9af1554cad7c880e2da24a9afd88aca000531ab82c1` |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | `899def5c37c4fd7b2664648c28120ecec138e4d395b459e5ca34f9cce2dd77fd` |
| `ident_case` | 1.0.1 | MIT/Apache-2.0 | `b9e0384b61958566e926dc50660321d12159025e767c18e043daf26b70104c39` |
| `itertools` | 0.14.0 | MIT OR Apache-2.0 | `2b192c782037fadd9cfa75548310488aabdbf3d2da73885b31bd0abd03351285` |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 | `8f42a60cbdf9a97f5d2305f08a87dc4e09308d1276d28c869c684d7777685682` |
| `libc` | 0.2.189 | MIT OR Apache-2.0 | `3eaf3ede3fee6db1a4c2ee091bf8a8b4dccdc6d17f656fb07896ee72867612f2` |
| `log` | 0.4.34 | MIT OR Apache-2.0 | `f9f8bd3e56ce4dfc153cf470fffbfa98c7620958b312ca5c3a4b8d5181fd13c6` |
| `macro_rules_attribute` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `b3ae8f6d608c795738406608304d30a2dfbdc8e58e44f7ba43236da5208ded3c` |
| `macro_rules_attribute-proc_macro` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `fc04a4c58212d57930a24bf47d3fa87485264a3a054e9c10e042eb373573ad3c` |
| `memchr` | 2.8.3 | Unlicense OR MIT | `cf8baf1c55e62ffcace7a9f06f4bd9cd3f0c4beb022d3b367256b91b87513d98` |
| `minimal-lexical` | 0.2.1 | MIT/Apache-2.0 | `68354c5c6bd36d73ff3feceb05efa59b6acb7626617f4962be322a825e61f79a` |
| `monostate` | 0.1.18 | MIT OR Apache-2.0 | `3341a273f6c9d5bef1908f17b7267bbab0e95c9bf69a0d4dcf8e9e1b2c76ef67` |
| `monostate-impl` | 0.1.18 | MIT OR Apache-2.0 | `e4db6d5580af57bf992f59068d4ea26fd518574ff48d7639b255a36f9de6e7e9` |
| `nom` | 7.1.3 | MIT | `d273983c5a657a70a3e8f2a01329822f3b8c8172b73826411a55751e404a0a4a` |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 | `9f7c3e4beb33f85d45ae3e3a1792185706c8e16d043238c593331cc7cd313b50` |
| `onig` | 6.5.3 | MIT | `0cc3cbf698f9438986c11a880c90a6d04b9de27575afd28bbf45b154b6c709e2` |
| `onig_sys` | 69.9.3 | MIT | `1e68317604e77e53b85896388e1a803c1d21b74c899ec9e5e1112db90735edd7` |
| `paste` | 1.0.15 | MIT OR Apache-2.0 | `57c0d7b74b563b49d38dae00a0c37d4d6de9b432382b2892f0574ddcae73fd0a` |
| `pastey` | 0.2.3 | MIT OR Apache-2.0 | `2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4` |
| `pkg-config` | 0.3.34 | MIT OR Apache-2.0 | `f6b464fbc74e149a392436b17d523f769e057cb6877f6a5c4618bc6f11800548` |
| `ppv-lite86` | 0.2.21 | MIT OR Apache-2.0 | `85eae3c4ed2f50dcfe72643da4befc30deadb458a9b590d720cde2f2b1e97da9` |
| `proc-macro2` | 1.0.107 | MIT OR Apache-2.0 | `985e7ec9bb745e6ce6535b544d84d6cd6f7ad8bd711c398938ae983b91a766d9` |
| `quote` | 1.0.47 | MIT OR Apache-2.0 | `1fbf4db142a473a8d80c26bbf18454ed458bf8d26c8219c331daecfdbd079001` |
| `rand` | 0.9.5 | MIT OR Apache-2.0 | `b9ef1d0d795eb7d84685bca4f72f3649f064e6641543d3a8c415898726a57b41` |
| `rand_chacha` | 0.9.0 | MIT OR Apache-2.0 | `d3022b5f1df60f26e1ffddd6c66e8aa15de382ae63b3a0c1bfc0e4d3e3f325cb` |
| `rand_core` | 0.9.5 | MIT OR Apache-2.0 | `76afc826de14238e6e8c374ddcc1fa19e374fd8dd986b0d2af0d02377261d83c` |
| `rayon` | 1.12.0 | MIT OR Apache-2.0 | `fb39b166781f92d482534ef4b4b1b2568f42613b53e5b6c160e24cfbfa30926d` |
| `rayon-cond` | 0.4.0 | Apache-2.0/MIT | `2964d0cf57a3e7a06e8183d14a8b527195c706b7983549cd5462d5aa3747438f` |
| `rayon-core` | 1.13.0 | MIT OR Apache-2.0 | `22e18b0f0062d30d4230b2e85ff77fdfe4326feb054b9783a3460d8435c8ab91` |
| `regex` | 1.13.1 | MIT OR Apache-2.0 | `f020237b6c8eed93db2e2cb53c00c60a8e1bc73da7d073199a1180401450218d` |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | `ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2` |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | `d6f6ff9a378485b298a5286656da665ba74413d36db0979633275d2e708145d4` |
| `rustversion` | 1.0.23 | MIT OR Apache-2.0 | `cf54715a573b99ac80df0bc206da022bcd442c974952c7b9720069370852e21f` |
| `ryu` | 1.0.23 | Apache-2.0 OR BSL-1.0 | `9774ba4a74de5f7b1c1451ed6cd5285a32eddb5cccb8cc655a4e50009e06477f` |
| `serde` | 1.0.229 | MIT OR Apache-2.0 | `4148590afebada386688f18773da617792bf2ef03ffc1e4cbd2b1d45b023e0ba` |
| `serde_core` | 1.0.229 | MIT OR Apache-2.0 | `67dca2c9c51e58a4791a4b1ed58308b39c64224d349a935ab5039aa360942a48` |
| `serde_derive` | 1.0.229 | MIT OR Apache-2.0 | `e7a5d71263a5a7d47b41f6b3f06ba276f10cc18b0931f1799f710578e2309348` |
| `serde_json` | 1.0.151 | MIT OR Apache-2.0 | `c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14` |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 | `f8fadd59c855ef2080decdef8ff161eb6661b86933c9d82e5ba29dc602a55aba` |
| `smallvec` | 1.16.2 | MIT OR Apache-2.0 | `f9395f0f0eee849a9b707b2f06bb92a6a422090e2123bb2ef8e87a0e61892a8e` |
| `spm_precompiled` | 0.1.4 | Apache-2.0 | `5851699c4033c63636f7ea4cf7b7c1f1bf06d0cc03cfb42e711de5a5c46cf326` |
| `static_assertions` | 1.1.0 | MIT OR Apache-2.0 | `a2eb9349b6444b326872e140eb1cf5e7c522154d69e7a0ffb0fb81c06b37543f` |
| `strsim` | 0.11.1 | MIT | `7da8b5736845d9f2fcb837ea5d9e2628564b3b043a70948a3f0b778838c5fb4f` |
| `syn` | 2.0.119 | MIT OR Apache-2.0 | `872831b642d1a07999a962a351ed35b955ea2cfc8f3862091e2a240a84f17297` |
| `syn` | 3.0.6 | MIT OR Apache-2.0 | `8593e8e72159ed2257d083c7a454a85cbf854f37a0966d8d483aff8c8a3ebcee` |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 | `09e52cb86a36cede5cb101bf8908837b3e4c6e5e59fe7fd85c23fb56200d189e` |
| `thiserror-impl` | 2.0.21 | MIT OR Apache-2.0 | `fe5197923287db20a58125f0bc85c062f7f2c892de97b18c356f9efb14b28524` |
| `tokenizers` | 0.22.2 | Apache-2.0 | `b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223` |
| `unicode-ident` | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 | `d245f478577f809a851594d02313b640fb437e0bb33866753cff937863096954` |
| `unicode-normalization-alignments` | 0.1.12 | MIT/Apache-2.0 | `43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de` |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | `c6f5d3c3b1bf09027a88a6bc961fc00497d651009560b5463668dc81b0fa87a8` |
| `unicode_categories` | 0.1.1 | MIT OR Apache-2.0 | `39ec24b3121d976906ece63c9daad25b85969647682eee313cb5779fdd69e14e` |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | `0b928f33d975fc6ad9f86c8f283853ad26bdd5b10b7f1542aa2fa15e2289105a` |
| `zerocopy` | 0.8.59 | BSD-2-Clause OR Apache-2.0 OR MIT | `6df92bf3d9227be3d53173901ddbffac2babc27ae50f397776ffd6dc33f800cb` |
| `zmij` | 1.0.23 | MIT | `29666d0abbfad1e3dc4dcf6144730dd3a3ab225bbbdac83319345b1b44ccfc1b` |

`onig_sys` vendors Oniguruma (BSD-2-Clause) and `esaxx-rs` vendors esaxx (MIT); both are C/C++ compiled by
their build scripts. The Rust standard library linked in is MIT OR Apache-2.0 (libunwind: Apache-2.0 WITH
LLVM-exception).

## 3. ONNX Runtime 1.31.1 (built from source by `tools/ort-musl-build.sh`)

| Name | URL | Pin | Licence |
|---|---|---|---|
| ONNX Runtime | https://github.com/microsoft/onnxruntime | tag `v1.31.1` = `6af1a932e2286d5c58166085d24c68783033edee`, plus `patches/musl/onnxruntime/*.patch` | MIT |

Its build fetches these with CMake FetchContent from `cmake/deps.txt` (`name;url;sha1`), checked by SHA1. This
is the set a CPU build for prplOS fetches (measured from `_deps/` of the default and full builds here):

| deps.txt name | URL | SHA1 | Licence |
|---|---|---|---|
| `abseil_cpp` | https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.zip | `a9eb1d648cbca4d4d788737e971a6a7a63726b07` | Apache-2.0 |
| `date` | https://github.com/HowardHinnant/date/archive/refs/tags/v3.0.1.zip | `2dac0c81dc54ebdd8f8d073a75c053b04b56e159` | MIT |
| `eigen` | https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip | `05b19b49e6fbb91246be711d801160528c135e34` | MPL-2.0 |
| `flatbuffers` | https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.zip | `59422c3b5e573dd192fead2834d25951f1c1670c` | Apache-2.0 |
| `json` | https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip | `5e88795165cc8590138d1f47ce94ee567b85b4d6` | MIT |
| `microsoft_gsl` | https://github.com/microsoft/GSL/archive/refs/tags/v4.2.1.zip | `1094e3bb7a8af763dcb136ccd676e6e75e614eec` | MIT |
| `mp11` | https://github.com/boostorg/mp11/archive/refs/tags/boost-1.82.0.zip | `9bc9e01dffb64d9e0773b2e44d2f22c51aace063` | BSL-1.0 |
| `onnx` | https://github.com/onnx/onnx/archive/refs/tags/v1.22.0.zip | `2b2cd58ac7a26df5371266149e0c76776330cdf1` | Apache-2.0 |
| `protobuf` | https://github.com/protocolbuffers/protobuf/releases/download/v33.6/protobuf-33.6.tar.gz | `051fee744fcca04dab37eb009d49aa3e9546acda` | BSD-3-Clause |
| `protoc_linux_x64 (build host x86_64)` | https://github.com/protocolbuffers/protobuf/releases/download/v33.6/protoc-33.6-linux-x86_64.zip | `44c6aeb3f494e4f8e46d283b05e087c47cf6369d` | BSD-3-Clause (prebuilt binary) |
| `protoc_linux_aarch64 (build host aarch64)` | https://github.com/protocolbuffers/protobuf/releases/download/v33.6/protoc-33.6-linux-aarch_64.zip | `b7ec3edf45c31fed5318f65693ce5f14c850a72f` | BSD-3-Clause (prebuilt binary) |
| `pytorch_cpuinfo` | https://github.com/pytorch/cpuinfo/archive/66ee79c038d70dad9f08705b2c9b3e58f6d8f512.zip | `6e2be17c110e4f4184b059303c722b3c5bd080a6` | BSD-2-Clause |
| `re2` | https://github.com/google/re2/archive/refs/tags/2024-07-02.zip | `646e1728269cde7fcef990bf4a8e87b047882e88` | BSD-3-Clause |
| `safeint` | https://github.com/dcleblanc/SafeInt/archive/refs/tags/3.0.28.zip | `23f252040ff6cb9f1fd18575b32fa8fb5928daac` | MIT |
| `kleidiai (aarch64 target only)` | https://github.com/ARM-software/kleidiai/archive/refs/tags/v1.30.0.tar.gz | `36aebdd5bd28c40b1943eb93498ae42e703510e5` | Apache-2.0 |

Offline, either:

- **mirror directory** (no source change): put each archive at `<onnxruntime>/mirror/<url without https://>`
  (or `build.py --cmake_deps_mirror_dir <dir>` / `onnxruntime_CMAKE_DEPS_MIRROR_DIR=<dir>` with the same layout); ORT uses the
  local file when it exists and still checks the SHA1; or
- **edit `cmake/deps.txt`**: its URL column takes local file paths (`/dl/abseil-cpp-20250814.0.zip`).

`protoc_linux_x64` / `protoc_linux_aarch64` are **prebuilt protoc executables** that ORT runs on the build
host to generate the ONNX protobuf sources (the host's architecture is chosen, not the target's). A source-only
feed can build protoc for the host from the same protobuf v33.6 tarball and pass it with
`build.py --path_to_protoc_exe <host protoc>` (`ONNX_CUSTOM_PROTOC_EXECUTABLE`); that switch was not exercised
in this run.

The prplOS sysroot ships protobuf 3.17.3; ORT's `find_package` would prefer it and fail against its own
protoc, so it must be moved out of the staging sysroot (or the package must not depend on protobuf) before
building. ORT then builds protobuf from the tarball above and links it statically.

## Flags

- No dependency is GPL/LGPL or proprietary; nothing blocks shipping the sources in a public feed. Eigen is
  MPL-2.0 (file-level copyleft: shipping its source satisfies it); ORT compiles it with `EIGEN_MPL2_ONLY`.
- The one non-source input is ORT's prebuilt `protoc` (BSD-3-Clause, the same project's binary release):
  replaceable by a host protoc built from source, see section 3.
- CPU only. The build fetches no CUDA, TensorRT, QNN/QAIRT, DirectML or WebGPU component (those deps.txt
  entries are only fetched when their execution provider is enabled).
- Models are not part of the feed; the ones used to derive `ops/default.required_operators.config` are listed
  in that file (all Apache-2.0 except the LFM2 router, LFM Open License).
