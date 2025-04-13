# VAEforPointCloud

~~~py
VAEforPointCloud/
├── lib/
│   ├── __init__.py            # An empty file required to recognize this directory as a module
│   ├── ChamferDis.py          # Chamfer Distance calculation
│   ├── Dataloader.py          # DataLoader for preparing training data
│   ├── EarthMoversDis.py      # Earth Mover's Distance calculation
│   ├── file_io.py             # File input/output functions for reading OFF files
│   ├── ☆mogvae_v2_models.py  # Mixture of Gaussians VAE model (MoGVAE) for non-flattened input
│   ├── point_cloud.py         # Functions for point cloud visualization and rotation
│   ├── sampling.py            # PointSampler class for point cloud data
│   ├── train.py               # Training function for VAE models
│   ├── utils.py               # Utility functions (memory check, data handling)
│   ├── ☆vae_v2_models.py     # Standard VAE model (standVAE) for non-flattened input
│   ├── ☆vqvae_v2_models.py   # VQ-VAE model for non-flattened input
│   ├── ☆SetVAE.py   # SetVAE model for non-flattened input
│   ├── ☆ISAB.py              # ISAB(Induced set attention block) model for non-flattened input
│   └── visualize_loss.py      # Functions for visualizing training loss and results
|
├── main_for_standardVAE_v2_(MSE).ipynb # Main VAE script for training and evaluation ( Loss function MSE + KL_D ) for non-flattened input
├── main_for_MoGVAE_v2_(MSE).ipynb      # Main MoG-VAE script for training and evaluation ( Loss function MSE + KL_D ) for non-flattened input
├── main_for_MoGVAE_v2_(MSE+CD).ipynb   # Main MoG-VAE script for training and evaluation ( Loss function MSE + CD+ KL_D ) for non-flattened input
├── main_for_SetVAE_v2_(MSE).ipynb      # Main SetVAE w/ ISAB( Induced set attention block) script for training and evaluation ( Loss function MSE + KL_D ) for non-flattened input
├── main_for_VQVAE_v2_(MSE).ipynb       # Main VQ-VAE script for training and evaluation ( Loss function MSE ) for non-flattened input
├── requirements.txt                    # List of required Python packages
└── README.md                           # Project description

~~~

| Architecture         | VAE                        | MoG-VAE                   | MoG-VAE                       | SetVAE                        | SetVAE                            | VQ-VAE                       | ISAB+VQ-VAE                 | MAB+VQ-VAE                  | ISAB+SoftVQ-VAE              |
|----------------------|----------------------------|---------------------------|-------------------------------|-------------------------------|-----------------------------------|------------------------------|-----------------------------|-----------------------------|------------------------------|
| **Encoder**          | Pointwise conv + Max Pooling | Same as Left              | Same as Left                  | Induced Set Attention Block   | Same as Left                       | Pointwise conv + Max Pooling | Induced Set Attention Block | Multihead Attention Block    | Induced Set Attention Block |
| **Decoder**          | Deconvolution (Transpose Conv) | Same as Left              | Same as Left                  | Same as Left                  | Same as Left                       | Same as Left                 | Same as Left                | Same as Left                | Same as Left                |
| **Loss**             | MSE + KLD                  | MSE + KLD                 | MSE + 2×CD + KLD              | MSE + 0.8 × KLD               | 0.5×MSE + 10×CD + 0.4×KLD          | MSE + Codebook + Commitment | Same as Left               | Same as Left               | MSE + Codebook (SoftVQ) + Commitment |
| **Learning Rate**    | 1.0E-05                    | 1.0E-05                   | 1.0E-04                       | 5.0E-05                       | 1.0E-04                            | 1.0E-03                      | 1.0E-04                     | 1.0E-04                     | 1.0E-04                     |
| **CD**               |                            |                           |                               |                               |                                   |                              |                             |                             |                              |
| Design 1             | 0.0245                     | 0.0247                    | 0.0145                        | 0.0239                        | 0.0166                            | 0.0188                       | 0.0116                      | 0.0133                      | 0.0014                      |
| Design 2             | 0.0247                     | 0.0226                    | 0.0154                        | 0.0187                        | 0.0172                            | 0.0208                       | 0.0112                      | 0.0046                      | 0.0005                      |
| Design 3             | 0.0390                     | 0.0161                    | 0.0231                        | 0.0188                        | 0.0226                            | 0.0105                       | 0.0051                      | 0.0092                      | 0.0006                      |
| Design 4             | 0.0303                     | 0.0227                    | 0.0287                        | 0.0210                        | 0.0156                            | 0.0130                       | 0.0035                      | 0.0033                      | 0.0005                      |
| Design 5             | 0.0333                     | 0.0287                    | 0.0286                        | 0.0343                        | 0.0223                            | 0.0188                       | 0.0094                      | 0.0070                      | 0.0009                      |
| Design 6             | 0.0292                     | 0.0174                    | 0.0218                        | 0.0185                        | 0.0131                            | 0.0121                       | 0.0087                      | 0.0052                      | 0.0004                      |
| Design 7             | 0.0463                     | 0.0277                    | 0.0198                        | 0.0186                        | 0.0205                            | 0.0106                       | 0.0119                      | 0.0208                      | 0.0004                      |
| Design 8             | 0.0286                     | 0.0281                    | 0.0236                        | 0.0151                        | 0.0172                            | 0.0135                       | 0.0125                      | 0.0052                      | 0.0006                      |
| Design 9             | 0.0315                     | 0.0316                    | 0.0221                        | 0.0180                        | 0.0349                            | 0.0258                       | 0.0166                      | 0.0068                      | 0.0007                      |
| **Average**          | **0.0319**                 | **0.0244**                | **0.0220**                    | **0.0208**                    | **0.0200**                        | **0.0160**                   | **0.0101**                  | **0.0084**                  | **0.0007**                  |
| **EMD**              |                            |                           |                               |                               |                                   |                              |                             |                             |                              |
| Design 1             | 0.0167                     | 0.0181                    | 0.0088                        | 0.0210                        | 0.0112                            | 0.0135                       | 0.0068                      | 0.0084                      | 0.0007                      |
| Design 2             | 0.0145                     | 0.0131                    | 0.0082                        | 0.0103                        | 0.0093                            | 0.0116                       | 0.0058                      | 0.0023                      | 0.0002                      |
| Design 3             | 0.0298                     | 0.0088                    | 0.0142                        | 0.0105                        | 0.0140                            | 0.0055                       | 0.0026                      | 0.0047                      | 0.0003                      |
| Design 4             | 0.0195                     | 0.0136                    | 0.0189                        | 0.0119                        | 0.0086                            | 0.0069                       | 0.0018                      | 0.0017                      | 0.0002                      |
| Design 5             | 0.0217                     | 0.0176                    | 0.0177                        | 0.0239                        | 0.0132                            | 0.0104                       | 0.0048                      | 0.0035                      | 0.0004                      |
| Design 6             | 0.0267                     | 0.0111                    | 0.0140                        | 0.0131                        | 0.0079                            | 0.0070                       | 0.0047                      | 0.0027                      | 0.0002                      |
| Design 7             | 0.0400                     | 0.0176                    | 0.0112                        | 0.0103                        | 0.0118                            | 0.0054                       | 0.0062                      | 0.0117                      | 0.0002                      |
| Design 8             | 0.0213                     | 0.0200                    | 0.0161                        | 0.0086                        | 0.0103                            | 0.0076                       | 0.0069                      | 0.0027                      | 0.0003                      |
| Design 9             | 0.0197                     | 0.0200                    | 0.0130                        | 0.0100                        | 0.0238                            | 0.0157                       | 0.0091                      | 0.0035                      | 0.0003                      |
| **Average**          | **0.0233**                 | **0.0155**                | **0.0136**                    | **0.0133**                    | **0.0122**                        | **0.0093**                   | **0.0054**                  | **0.0046**                  | **0.0003**                  |


