# VAEforPointCloud

~~~py
VAEforPointCloud/
├── lib/
│   ├── __init__.py            # An empty file required to recognize this directory as a module
│   ├── ChamferDis.py          # Chamfer Distance calculation
│   ├── Dataloader.py          # DataLoader for preparing training data
│   ├── EarthMoversDis.py      # Earth Mover's Distance calculation
│   ├── file_io.py             # File input/output functions for reading OFF files
│   ├── mogvae_models.py       # Mixture of Gaussians VAE model (MoGVAE) for flattened input
│   ├── mogvae_v2_models.py    # Mixture of Gaussians VAE model (MoGVAE) for non-flattened input
│   ├── point_cloud.py         # Functions for point cloud visualization and rotation
│   ├── sampling.py            # PointSampler class for point cloud data
│   ├── train.py               # Training function for VAE models
│   ├── utils.py               # Utility functions (memory check, data handling)
│   ├── vae_models.py          # Standard VAE model (standVAE)
│   ├── vqvae_models.py        # VQ-VAE model
│   └── visualize_loss.py      # Functions for visualizing training loss and results
|
├── main_for_standardVAE(MSE).ipynb     # Main VAE script for training and evaluation ( Loss function MSE + KL_D )
├── main_for_MoGVAE(MSE).ipynb          # Main MoG-VAE script for training and evaluation ( Loss function MSE + KL_D ) for flattened input
├── main_for_MoGVAE_v2_(MSE).ipynb      # Main MoG-VAE script for training and evaluation ( Loss function MSE + KL_D ) for non-flattened input
├── main_for_MoGVAE(MSE+CD).ipynb       # Main MoG-VAE script for training and evaluation ( Loss function MSE + CD+ KL_D ) for flattened input
├── main_for_MoGVAE_v2_(MSE+CD).ipynb   # Main MoG-VAE script for training and evaluation ( Loss function MSE + CD+ KL_D ) for non-flattened input
├── main_for_VQVAE.ipynb                # Main VQ-VAE script for training and evaluation
├── requirements.txt                    # List of required Python packages
└── README.md                           # Project description
~~~
