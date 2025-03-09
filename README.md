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


| Architecture | VAE | MoG-VAE | MoG-VAE | SetVAE | VQ-VAE |
|-------------|----------------|----------------|--------------------|----------------|---------------------------|
| Loss        | MSE + KLD | MSE + KLD | MSE + 2 × CD + KLD | MSE + KLD | MSE + Codebook + Commitment |
| Learning Rate | 1.0E-05 | 1.0E-05 | 1.0E-05 | 1.0E-04 | 5.0E-05 | 1.0E-03 |
| **CD** |  |  |  |  |  |
| Design 1 | 0.0245 | 0.0247 | 0.0145 | 0.0239 | 0.0188 |
| Design 2 | 0.0247 | 0.0226 | 0.0154 | 0.0187 | 0.0208 |
| Design 3 | 0.0271 | 0.0239 | 0.0150 | 0.0213 | 0.0188 |
| Design 4 | 0.0303 | 0.0227 | 0.0287 | 0.0267 | 0.0216 |
| Design 5 | 0.0292 | 0.0171 | 0.0281 | 0.0216 | 0.0223 |
| Design 6 | 0.0323 | 0.0246 | 0.0170 | 0.0237 | 0.0171 |
| Design 7 | 0.0286 | 0.0241 | 0.0146 | 0.0183 | 0.0177 |
| Design 8 | 0.0315 | 0.0316 | 0.0221 | 0.0210 | 0.0258 |
| **Average** | **0.0319** | **0.0244** | **0.0220** | **0.0208** | **0.0160** |
| **EMD** |  |  |  |  |  |
| Design 1 | 0.0167 | 0.0181 | 0.0088 | 0.0210 | 0.0135 |
| Design 2 | 0.0145 | 0.0131 | 0.0072 | 0.0184 | 0.0116 |
| Design 3 | 0.0298 | 0.0083 | 0.0103 | 0.0210 | 0.0136 |
| Design 4 | 0.0159 | 0.0120 | 0.0116 | 0.0177 | 0.0151 |
| Design 5 | 0.0175 | 0.0175 | 0.0137 | 0.0239 | 0.0239 |
| Design 6 | 0.0270 | 0.0200 | 0.0147 | 0.0239 | 0.0170 |
| Design 7 | 0.0204 | 0.0126 | 0.0170 | 0.0199 | 0.0147 |
| Design 8 | 0.0232 | 0.0170 | 0.0116 | 0.0157 | 0.0157 |
| Design 9 | 0.0197 | 0.0200 | 0.0130 | 0.0100 | 0.0175 |
| **Average** | **0.0233** | **0.0155** | **0.0136** | **0.0133** | **0.0093** |

