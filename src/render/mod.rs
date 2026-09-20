//! GPU rendering of the phase field, as an `egui_wgpu` paint callback.
//!
//! The image is drawn by a single oversized triangle whose fragment shader
//! resolves each fragment to a cell, so nothing here scales with `m × n` except
//! the one-off texture upload.
//!
//! # Known limitation: minification
//!
//! Below roughly one pixel per cell each fragment samples a single arbitrary
//! cell rather than reducing over all the cells it covers, so zooming far out
//! aliases. Fixing that properly needs a mip pyramid or a min/max reduction
//! pass; it is deliberately out of scope for this first slice.

use std::sync::Arc;

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use eframe::wgpu;
use egui::Rect;

use crate::colormap::Colormap;
use crate::graph::Unwrapping;
use crate::phase::PhaseField;

/// Number of entries in the colormap lookup texture.
pub const LUT_LEN: usize = 256;

/// Width of a cell wall, in physical pixels.
const WALL_WIDTH_PX: f32 = 1.0;

/// Cell size, in physical pixels, below which walls are fully faded out.
const WALL_FADE_START_PX: f32 = 3.0;

/// Cell size, in physical pixels, above which walls are fully opaque.
const WALL_FADE_END_PX: f32 = 6.0;

/// How opaque the cell walls should be at a given zoom.
///
/// Walls are meaningful when a cell is comfortably larger than the wall itself
/// and pure noise when it is not, so they fade out rather than turning the image
/// into a grey mush as the user zooms away.
pub fn wall_opacity(pixels_per_cell: f32) -> f32 {
    smoothstep(WALL_FADE_START_PX, WALL_FADE_END_PX, pixels_per_cell)
}

/// Hermite interpolation between `low` and `high`, clamped at both ends.
fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
    if high <= low {
        return if x < low { 0.0 } else { 1.0 };
    }
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * 2.0f32.mul_add(-t, 3.0)
}

/// Smallest radius a residue marker is drawn at, in physical pixels.
///
/// Residues do not fade with zoom the way walls do: they are sparse, and they
/// are the obstruction that forces any edge to disagree at all, so they stay
/// findable when the whole field is on screen.
const RESIDUE_MIN_RADIUS_PX: f32 = 2.5;

/// Largest radius a residue marker is drawn at, in physical pixels.
const RESIDUE_MAX_RADIUS_PX: f32 = 7.0;

/// Residue radius as a fraction of a cell, between those two bounds.
const RESIDUE_RADIUS_PER_CELL: f32 = 0.16;

/// How large to draw the residue markers at a given zoom.
pub fn residue_radius(pixels_per_cell: f32) -> f32 {
    (pixels_per_cell * RESIDUE_RADIUS_PER_CELL).clamp(RESIDUE_MIN_RADIUS_PX, RESIDUE_MAX_RADIUS_PX)
}

/// The uniform block handed to `grid.wgsl`.
///
/// Encoded as four `vec4<f32>`, matching `struct Uniforms` there. Grouping
/// everything into `vec4`s means the uniform-address-space alignment rules are
/// satisfied by construction, with no padding fields to keep in sync.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridUniforms {
    /// Data-space coordinate at the viewport's top-left corner.
    pub data_min: [f32; 2],
    /// Data-space coordinate at the viewport's bottom-right corner.
    pub data_max: [f32; 2],
    /// `[cols, rows]`.
    pub extent: [f32; 2],
    /// Physical pixels per cell.
    pub pixels_per_cell: f32,
    /// Wall width, in physical pixels.
    pub wall_width_px: f32,
    /// Value that maps to the start of the colormap.
    pub value_min: f32,
    /// Reciprocal of the value range.
    pub value_range_inv: f32,
    /// `1.0` to wrap values to `(-π, π]` before colouring, `0.0` otherwise.
    pub wrap_mode: f32,
    /// Wall opacity, from [`wall_opacity`].
    pub wall_opacity: f32,
    /// `1.0` to draw the unwrapping overlay, `0.0` to draw the field alone.
    pub overlay_enabled: f32,
    /// Radius of a residue marker, in physical pixels.
    pub residue_radius_px: f32,
}

impl GridUniforms {
    /// Size of the encoded block, in bytes: four `vec4<f32>`.
    pub const SIZE: usize = 64;

    /// Builds the uniforms for one frame.
    ///
    /// `visible` is the data-space rectangle covered by the widget, and
    /// `pixels_per_cell` is in *physical* pixels, so the wall width and its
    /// fade behave the same on a high-DPI display as on a normal one.
    /// `overlay` turns on the per-edge wall colours and residue markers.
    pub fn new(
        visible: Rect,
        rows: usize,
        cols: usize,
        pixels_per_cell: f32,
        value_range: (f32, f32),
        wrapped: bool,
        overlay: bool,
    ) -> Self {
        let (min, max) = value_range;
        // A degenerate range would divide by zero; map the whole field to the
        // middle of the colormap instead, which is what a constant field means.
        let (value_min, value_range_inv) = if max > min && (max - min).is_finite() {
            (min, 1.0 / (max - min))
        } else {
            (min - 0.5, 1.0)
        };

        Self {
            data_min: [visible.min.x, visible.min.y],
            data_max: [visible.max.x, visible.max.y],
            extent: [cols as f32, rows as f32],
            pixels_per_cell,
            wall_width_px: WALL_WIDTH_PX,
            value_min,
            value_range_inv,
            wrap_mode: if wrapped { 1.0 } else { 0.0 },
            wall_opacity: wall_opacity(pixels_per_cell),
            overlay_enabled: if overlay { 1.0 } else { 0.0 },
            residue_radius_px: residue_radius(pixels_per_cell),
        }
    }

    /// Encodes the block for upload, little-endian as every wgpu backend expects.
    ///
    /// Done by hand rather than by transmuting a `#[repr(C)]` struct: the crate
    /// forbids `unsafe`, and this keeps the layout explicit and testable.
    fn to_bytes(self) -> [u8; Self::SIZE] {
        let floats: [f32; 16] = [
            // bounds
            self.data_min[0],
            self.data_min[1],
            self.data_max[0],
            self.data_max[1],
            // grid
            self.extent[0],
            self.extent[1],
            self.pixels_per_cell,
            self.wall_width_px,
            // shading
            self.value_min,
            self.value_range_inv,
            self.wrap_mode,
            self.wall_opacity,
            // overlay
            self.overlay_enabled,
            self.residue_radius_px,
            0.0,
            0.0,
        ];

        let mut bytes = [0u8; Self::SIZE];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(floats) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// The phase samples, uploaded as a single-channel float texture.
struct DataTexture {
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    rows: usize,
    cols: usize,

    /// The field these texels came from.
    ///
    /// Holding the `Arc` is what makes the staleness check sound: while it is
    /// alive nothing else can occupy that allocation, so pointer equality means
    /// the same immutable samples. A hand-maintained counter cannot do this —
    /// the viewer shows two different fields of identical size, and a counter
    /// that identified only the *scene* let a tab switch slip through and left
    /// the other tab's samples on screen.
    source: Arc<PhaseField>,
}

/// The colormap, uploaded as a 1-D RGBA8 ramp.
///
/// Only the view is kept: a `wgpu::TextureView` keeps its texture alive, and
/// unlike the data texture this one is never written to again.
struct LutTexture {
    view: wgpu::TextureView,
    colormap: Colormap,
}

/// The per-edge and per-corner overlay data, uploaded as integer textures.
struct OverlayTextures {
    edges: wgpu::TextureView,
    residues: wgpu::TextureView,
    /// The analysis these texels came from; see [`DataTexture::source`].
    source: Arc<Unwrapping>,
}

/// Everything the grid needs on the GPU, kept in `egui_wgpu`'s callback
/// resources so it outlives any individual frame.
pub struct GridRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    max_texture_dimension: u32,

    data: Option<DataTexture>,
    lut: Option<LutTexture>,
    overlay: Option<OverlayTextures>,

    /// Bound whenever there is no overlay, so the bind group is always
    /// complete and the shader's `overlay_enabled` flag is the only thing
    /// deciding whether the data is read.
    empty_edges: wgpu::TextureView,
    empty_residues: wgpu::TextureView,

    bind_group: Option<wgpu::BindGroup>,

    /// Size already reported as too large, so the log is not spammed every frame.
    reported_oversize: Option<(usize, usize)>,
}

impl GridRenderer {
    /// Creates the renderer and stores it in the render state's callback
    /// resources. Call once, from the app's constructor.
    pub fn install(render_state: &RenderState) {
        let renderer = Self::new(&render_state.device, render_state.target_format);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(renderer);
    }

    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phase_grid_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });

        let bind_group_layout = create_bind_group_layout(device);
        let pipeline = create_pipeline(device, &module, &bind_group_layout, target_format);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_grid_uniforms"),
            size: GridUniforms::SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("phase_grid_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            max_texture_dimension: device.limits().max_texture_dimension_2d,
            data: None,
            lut: None,
            overlay: None,
            empty_edges: create_uint_texture(
                device,
                "phase_grid_empty_edges",
                wgpu::TextureFormat::Rgba8Uint,
                1,
                1,
            ),
            empty_residues: create_uint_texture(
                device,
                "phase_grid_empty_residues",
                wgpu::TextureFormat::R8Uint,
                1,
                1,
            ),
            bind_group: None,
            reported_oversize: None,
        }
    }
}

/// The single bind group: uniforms, the field, the colormap ramp and its sampler.
fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("phase_grid_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(GridUniforms::SIZE as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    // Read with `textureLoad`, never filtered: a float
                    // texture does not need to be filterable for that, which
                    // keeps this working on backends that cannot filter one.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Creates an integer texture that the shader reads with `textureLoad`.
fn create_uint_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Uploads `data` into a freshly created integer texture.
///
/// `size` is `(cols, rows)`; the row stride is derived from the format, so the
/// caller cannot get it out of step with it.
fn upload_uint_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    size: (usize, usize),
    data: &[u8],
) -> wgpu::TextureView {
    let (cols, rows) = size;
    let bytes_per_texel = format.block_copy_size(None).unwrap_or(1) as usize;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: dimension(cols),
            height: dimension(rows),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(dimension(cols * bytes_per_texel)),
            rows_per_image: Some(dimension(rows)),
        },
        wgpu::Extent3d {
            width: dimension(cols),
            height: dimension(rows),
            depth_or_array_layers: 1,
        },
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Builds the render pipeline for the grid.
fn create_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("phase_grid_pipeline_layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    // Mirrors what egui's own pipeline does: colours are authored in gamma
    // space, so they are converted to linear light only when the target
    // format applies the sRGB transfer function on write.
    let fragment_entry_point = if target_format.is_srgb() {
        "fs_main_linear_framebuffer"
    } else {
        "fs_main_gamma_framebuffer"
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("phase_grid_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            // The triangle is generated from `@builtin(vertex_index)`.
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        // These two must match the render pass egui set up. They are
        // correct for eframe's defaults, which `main.rs` does not override;
        // enabling `NativeOptions::multisampling` or `depth_buffer` would
        // mean matching them here too.
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry_point),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                // Premultiplied alpha, exactly as egui blends, so masked
                // cells composite correctly over the panel behind them.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl GridRenderer {
    /// Brings the GPU-side copies of the field and colormap up to date and
    /// writes this frame's uniforms.
    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, callback: &GridCallback) {
        let data_changed = self.sync_data(device, queue, &callback.field);
        let lut_changed = self.sync_lut(device, queue, callback.colormap);
        let overlay_changed = self.sync_overlay(device, queue, callback.overlay.as_ref());

        if data_changed || lut_changed || overlay_changed || self.bind_group.is_none() {
            self.rebuild_bind_group(device);
        }

        queue.write_buffer(&self.uniform_buffer, 0, &callback.uniforms.to_bytes());
    }

    /// Returns `true` if the texture object was replaced.
    fn sync_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: &Arc<PhaseField>,
    ) -> bool {
        let (rows, cols) = (field.rows(), field.cols());

        if rows == 0 || cols == 0 {
            return self.data.take().is_some();
        }

        let limit = self.max_texture_dimension as usize;
        if rows > limit || cols > limit {
            if self.reported_oversize != Some((rows, cols)) {
                log::error!(
                    "phase field is {rows} × {cols}, larger than the device's \
                     maximum texture dimension of {limit}; not rendering it"
                );
                self.reported_oversize = Some((rows, cols));
            }
            return self.data.take().is_some();
        }
        self.reported_oversize = None;

        if !needs_upload(self.data.as_ref().map(|data| &data.source), field) {
            return false;
        }

        let reusable = self
            .data
            .as_ref()
            .is_some_and(|data| data.rows == rows && data.cols == cols);

        let replaced = if reusable {
            false
        } else {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("phase_grid_data"),
                size: wgpu::Extent3d {
                    width: dimension(cols),
                    height: dimension(rows),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.data = Some(DataTexture {
                view,
                texture,
                rows,
                cols,
                source: Arc::clone(field),
            });
            true
        };

        let Some(data) = self.data.as_mut() else {
            return replaced;
        };

        {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &data.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &to_le_bytes(field.as_slice()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimension(cols * 4)),
                    rows_per_image: Some(dimension(rows)),
                },
                wgpu::Extent3d {
                    width: dimension(cols),
                    height: dimension(rows),
                    depth_or_array_layers: 1,
                },
            );
            data.source = Arc::clone(field);
        }

        replaced
    }

    /// Returns `true` if the texture object was replaced.
    fn sync_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, colormap: Colormap) -> bool {
        if self
            .lut
            .as_ref()
            .is_some_and(|lut| lut.colormap == colormap)
        {
            return false;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("phase_grid_lut"),
            size: wgpu::Extent3d {
                width: dimension(LUT_LEN),
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &colormap.lut(LUT_LEN),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimension(LUT_LEN * 4)),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: dimension(LUT_LEN),
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.lut = Some(LutTexture { view, colormap });
        true
    }

    /// Uploads the overlay textures when the unwrapping behind them changes.
    ///
    /// Returns `true` if the bound views changed.
    fn sync_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: Option<&OverlaySource>,
    ) -> bool {
        let Some(source) = source else {
            return self.overlay.take().is_some();
        };

        let (rows, cols) = (source.unwrapping.rows(), source.unwrapping.cols());
        if rows < 2 || cols < 2 {
            // Fewer than two rows or columns leaves no inner corner, so there
            // is no residue texture to build.
            return self.overlay.take().is_some();
        }

        if !needs_upload(
            self.overlay.as_ref().map(|overlay| &overlay.source),
            &source.unwrapping,
        ) {
            return false;
        }

        let edges = upload_uint_texture(
            device,
            queue,
            "phase_grid_edges",
            wgpu::TextureFormat::Rgba8Uint,
            (cols, rows),
            &source.unwrapping.edge_texels(),
        );
        let residues = upload_uint_texture(
            device,
            queue,
            "phase_grid_residues",
            wgpu::TextureFormat::R8Uint,
            (cols - 1, rows - 1),
            &source.unwrapping.residue_texels(),
        );

        self.overlay = Some(OverlayTextures {
            edges,
            residues,
            source: Arc::clone(&source.unwrapping),
        });
        true
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        let (Some(data), Some(lut)) = (self.data.as_ref(), self.lut.as_ref()) else {
            self.bind_group = None;
            return;
        };

        self.bind_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phase_grid_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&data.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&lut.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            self.overlay
                                .as_ref()
                                .map_or(&self.empty_edges, |overlay| &overlay.edges),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            self.overlay
                                .as_ref()
                                .map_or(&self.empty_residues, |overlay| &overlay.residues),
                        ),
                    },
                ],
            }),
        );
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            // Nothing uploaded yet, or the field is empty or too large.
            return;
        };
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Whether an uploaded copy is stale and has to be replaced.
///
/// Identity, not contents or size: the viewer shows several fields of exactly
/// the same shape, so anything coarser than "is this the very same allocation"
/// silently keeps the wrong samples on the GPU. Holding the `Arc` while it is
/// compared is what makes pointer equality sound — the allocation cannot be
/// reused underneath it.
fn needs_upload<T>(uploaded: Option<&Arc<T>>, incoming: &Arc<T>) -> bool {
    !uploaded.is_some_and(|uploaded| Arc::ptr_eq(uploaded, incoming))
}

/// Narrows a length to a texture dimension.
///
/// Every caller has already checked its value against the device's limit, which
/// is itself a `u32`, so the saturation never happens in practice. It exists so
/// that no size in this module is narrowed by a silent cast.
fn dimension(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// Re-encodes floats as the little-endian bytes a texture upload wants.
fn to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// One frame's worth of "draw the grid like this".
///
/// Holds only plain data — the GPU resources live in the callback resources —
/// so it satisfies the `Send + Sync` bound `CallbackTrait` requires.
pub struct GridCallback {
    field: Arc<PhaseField>,
    colormap: Colormap,
    uniforms: GridUniforms,
    overlay: Option<OverlaySource>,
}

/// The unwrapping whose per-edge and per-corner data should be drawn over the
/// field.
#[derive(Clone)]
pub struct OverlaySource {
    /// The analysed unwrapping. Replacing it with a different `Arc` is what
    /// tells the renderer to re-upload.
    pub unwrapping: Arc<Unwrapping>,
}

impl GridCallback {
    /// Draws `field` with `colormap`.
    ///
    /// The renderer decides whether its uploaded copy is stale by comparing the
    /// `Arc` itself, so passing a different field — including simply switching
    /// which one the UI is showing — is all it takes to refresh the GPU.
    pub fn new(field: Arc<PhaseField>, colormap: Colormap, uniforms: GridUniforms) -> Self {
        Self {
            field,
            colormap,
            uniforms,
            overlay: None,
        }
    }

    /// Draws the unwrapping overlay on top of the field.
    ///
    /// Without a matching `overlay_enabled` in the uniforms this is inert, so
    /// the two are set together by the widget.
    pub fn with_overlay(mut self, overlay: Option<OverlaySource>) -> Self {
        self.overlay = overlay;
        self
    }
}

impl CallbackTrait for GridCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get_mut::<GridRenderer>() {
            renderer.prepare(device, queue, self);
        } else {
            log::warn!("GridRenderer was never installed; the phase image will not be drawn");
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<GridRenderer>() {
            renderer.paint(render_pass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    fn uniforms() -> GridUniforms {
        GridUniforms::new(
            Rect::from_min_max(pos2(-1.5, 2.0), pos2(8.5, 6.0)),
            4,
            10,
            12.0,
            (-1.0, 3.0),
            false,
            false,
        )
    }

    /// The byte layout is the contract with `grid.wgsl`; if it drifts, the image
    /// silently renders garbage rather than failing to compile.
    #[test]
    fn uniforms_encode_four_vec4s_in_shader_order() {
        let bytes = uniforms().to_bytes();
        assert_eq!(bytes.len(), 64, "four vec4<f32> is 64 bytes");

        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word.copy_from_slice(chunk);
                f32::from_le_bytes(word)
            })
            .collect();

        assert_eq!(
            decoded,
            vec![
                // bounds: top-left then bottom-right
                -1.5,
                2.0,
                8.5,
                6.0, //
                // grid: cols, rows, pixels per cell, wall width
                10.0,
                4.0,
                12.0,
                WALL_WIDTH_PX, //
                // shading: value_min, 1/range, wrap mode, wall opacity
                -1.0,
                0.25,
                0.0,
                1.0, //
                // overlay: disabled, residue radius, two spare slots
                0.0,
                residue_radius(12.0),
                0.0,
                0.0,
            ],
            "field order must match `struct Uniforms` in grid.wgsl"
        );
    }

    #[test]
    fn a_degenerate_value_range_maps_to_the_middle_of_the_colormap() {
        let flat = GridUniforms::new(Rect::ZERO, 2, 2, 10.0, (7.0, 7.0), false, false);
        let t = (7.0 - flat.value_min) * flat.value_range_inv;
        assert!(
            (t - 0.5).abs() < 1e-6,
            "a constant field should be a single mid-colormap tone, got t = {t}"
        );
    }

    #[test]
    fn wrap_mode_is_a_flag() {
        let wrapped = GridUniforms::new(Rect::ZERO, 1, 1, 10.0, (0.0, 1.0), true, false);
        let plain = GridUniforms::new(Rect::ZERO, 1, 1, 10.0, (0.0, 1.0), false, false);
        assert_eq!(wrapped.wrap_mode, 1.0, "wrapped mode sets the flag");
        assert_eq!(plain.wrap_mode, 0.0, "unbounded mode clears it");
    }

    /// Residues deliberately do *not* follow the wall fade: they stay visible
    /// when the whole field is on screen, which is when you need to find them.
    #[test]
    fn residue_markers_stay_visible_at_every_zoom() {
        for pixels_per_cell in [0.05, 1.0, 4.0, 20.0, 400.0] {
            let radius = residue_radius(pixels_per_cell);
            assert!(
                radius >= RESIDUE_MIN_RADIUS_PX,
                "at {pixels_per_cell} px/cell a residue shrank to {radius} px"
            );
            assert!(
                radius <= RESIDUE_MAX_RADIUS_PX,
                "at {pixels_per_cell} px/cell a residue grew to {radius} px"
            );
        }
        assert!(
            residue_radius(1.0) > 0.0 && wall_opacity(1.0) == 0.0,
            "at one pixel per cell the walls are gone but the residues are not"
        );
    }

    #[test]
    fn the_overlay_flag_reaches_the_shader() {
        let off = uniforms();
        assert_eq!(off.overlay_enabled, 0.0, "plain fields draw no overlay");

        let on = GridUniforms::new(Rect::ZERO, 4, 4, 10.0, (0.0, 1.0), false, true);
        assert_eq!(on.overlay_enabled, 1.0, "the overlay sets the flag");
    }

    #[test]
    fn walls_fade_out_when_cells_get_small() {
        assert_eq!(
            wall_opacity(1.0),
            0.0,
            "at one pixel per cell the walls would be all there is"
        );
        assert_eq!(
            wall_opacity(WALL_FADE_START_PX),
            0.0,
            "the fade starts fully transparent"
        );
        assert_eq!(
            wall_opacity(WALL_FADE_END_PX),
            1.0,
            "the fade ends fully opaque"
        );
        assert_eq!(wall_opacity(100.0), 1.0, "zoomed in, walls are solid");

        let mut previous = -1.0;
        for i in 0..=100 {
            let opacity = wall_opacity(i as f32 * 0.1);
            assert!(
                opacity >= previous,
                "opacity must increase with zoom, dipped at {i}"
            );
            previous = opacity;
        }
    }

    /// The regression test for the tab-switch bug: the viewer's two tabs hold
    /// two fields of identical size, so a staleness check based on size — or on
    /// a counter identifying the scene rather than the field — reported "already
    /// uploaded" and left the previous tab's samples on screen.
    #[test]
    fn identical_fields_from_different_allocations_still_need_uploading() {
        let uploaded = Arc::new(PhaseField::linear_gradient(8, 8, 1.0, 1.0));
        let twin = Arc::new(PhaseField::linear_gradient(8, 8, 1.0, 1.0));
        assert_eq!(
            uploaded.as_slice(),
            twin.as_slice(),
            "the two fields are deliberately indistinguishable by value"
        );

        assert!(
            needs_upload(None, &uploaded),
            "an empty GPU always needs the first upload"
        );
        assert!(
            !needs_upload(Some(&uploaded), &Arc::clone(&uploaded)),
            "the same field must not be re-uploaded every frame"
        );
        assert!(
            needs_upload(Some(&uploaded), &twin),
            "a different field of the same shape must still be uploaded"
        );

        let other_shape = Arc::new(PhaseField::linear_gradient(4, 16, 1.0, 1.0));
        assert!(
            needs_upload(Some(&uploaded), &other_shape),
            "so must one of a different shape"
        );
    }

    #[test]
    fn float_encoding_is_little_endian_and_packed() {
        assert_eq!(
            to_le_bytes(&[1.0, -2.0]),
            [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat(),
            "samples must be packed tightly, in order"
        );
        assert!(
            to_le_bytes(&[]).is_empty(),
            "an empty field uploads nothing"
        );
    }
}
