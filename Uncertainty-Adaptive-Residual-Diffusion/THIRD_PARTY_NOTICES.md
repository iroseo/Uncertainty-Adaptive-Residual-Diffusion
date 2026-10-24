# Source provenance and third-party notices

The UARD implementation includes project-specific components and adapted research infrastructure. The acknowledgements below describe code provenance; they do not assign ownership of third-party work to the UARD authors.

| Project | Components or relationship | Source / notice |
| --- | --- | --- |
| E3Diff | The starting codebase for the training wrapper, U-Net/CPEN backbone, diffusion infrastructure, data utilities and logging. These files include UARD adaptations. | [DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff) |
| SR3 PyTorch implementation | An upstream source of the diffusion, U-Net and training infrastructure used by E3Diff. | [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement); [Apache-2.0 license copy](licenses/SR3_Apache-2.0.txt) |
| SoftPool | Pooling operator used by the conditioning backbone. The supplied configurations use the portable PyTorch implementation; the compiled extension is not bundled. | [alexandrosstergiou/SoftPool](https://github.com/alexandrosstergiou/SoftPool); [upstream license](licenses/SoftPool_LICENSE.txt) |
| img2img-turbo | Acknowledged by the E3Diff starting codebase. The optional vision-aided loss is not enabled in the supplied configurations and its package is not vendored. | [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo) |

The U-Net source also retains its inline WaveGrad positional-encoding reference. Existing source comments and copyright notices have been preserved. Changes made when preparing this preview are listed in [SOURCE_CHANGES.md](docs/SOURCE_CHANGES.md).

The local UARD research tree did not contain a project-wide license for E3Diff-derived code, and the E3Diff repository root inspected for this release did not supply one. The SR3 license copy applies to the corresponding upstream material; it is not a blanket license for every file in this preview. No new project-wide MIT or Apache license is assigned here. Third-party code and installed dependencies remain subject to their applicable terms.

The unused NVIDIA `perceptual.py` module and compiled SoftPool source tree are not distributed in this preview.
