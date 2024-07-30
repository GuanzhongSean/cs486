# A3Q4: Setup script for Assignment 3 Question 4

## (Optional) Install Miniconda

```bash
# Define Miniconda version and installation path
MINICONDA_VERSION=latest
INSTALL_PATH=$HOME/miniconda

# Download Miniconda installer
echo "Downloading Miniconda installer..."
if [ "$(uname)" == "Darwin" ]; then
    OS="MacOSX"
elif [ "$(uname)" == "Linux" ]; then
    OS="Linux"
else
    echo "Unsupported OS"
    exit 1
fi
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-${OS}-x86_64.sh
bash miniconda.sh -b -p $INSTALL_PATH

# Initialize Miniconda
echo "Initializing Miniconda..."
source $INSTALL_PATH/bin/conda init

# Clean up
rm miniconda.sh

echo "Miniconda installation is complete."
echo "To start using it, open a new terminal or run 'source ~/.bashrc' (or 'source ~/.zshrc' if you use zsh)."
```

## Prepare the environment
```bash
conda create -n a3q4 python=3.10 -y
conda activate a3q4
pip install numpy scikit-learn pandas scipy tqdm 

# [Optional] if you want to practice with real data and ipynb
conda install jupyter
pip install matplotlib 
wget https://zenodo.org/records/6355684/files/aloi-hsb-2x2x2.csv.gz
gzip -d aloi-hsb-2x2x2.csv.gz
```