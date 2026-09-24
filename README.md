# Phase Unwrapping Visualizer

An interactive viewer for InSAR phase fields, built with [egui](https://github.com/emilk/egui) and [eframe](https://github.com/emilk/egui/tree/master/crates/eframe). It runs natively and in the browser.

Load an original or wrapped phase field (`.phase`), and optionally a candidate unwrapping and the integration path (`.path`) that produced it. If no candidate is given, the viewer makes one itself with naive path integration or [snaphu-rs](https://github.com/vini-fda/snaphu-rs). You can then pan and zoom over the field and see where the unwrapping disagrees with the wrapped phase. It also ships with synthetic demo data, so there is something to look at without any files.

## Running natively

```sh
cargo run --release
```

On Linux, install the windowing dependencies first:

```sh
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

## Running on the web

The web build uses [Trunk](https://trunk-rs.github.io/trunk):

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
trunk serve
```

Open <http://127.0.0.1:8080/index.html#dev>. The `#dev` suffix skips the service worker's offline cache, so you always get the latest build.

`trunk build --release` writes a static site to `dist/`. The GitHub Pages workflow deploys it on every push to `main`.

## Checks

`./check.sh` runs the same checks as CI: check (native and wasm32), rustfmt, clippy, tests and a Trunk build.
