<h1 align="center" id="title">Data Mining Assignment</h1>

<p id="description">This is an assignnement in Data Mining Course. The goal of the assignment is to predict movies that will win an oscar using classification methods.</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Virtual Environment</p>

```
pip install virtualenv  python3 -m venv venv

# Activate the virtual environment 
# On Windows: 
venv\Scripts\activate 

# On macOS/Linux: 
source venv/bin/activate
```

<p>2. Installing Required Packages</p>

```
pip install -r requirements.txt
```

<p>3. Run Preprocess &amp; Classification</p>

```
# Format:
python3 main.py <lr/knn/rf/dtc/gb> <stats/nostats> <prepro/noprepro> <corel/nocorel>

# Example:
python3 main.py gb stats prepro corel
```

<p>4. Run Classification (without running preprocessing again)</p>

```
python3 main.py gb stats noprepro
```

<p>5. Run Clustering</p>

```
# Format:
python3 clusteringMain.py <KM/DBSCAN/HAC/Birch> <numClusters> <rs/ss/mm>

# Example:
python3 clusteringMain.py HAC 2 mm
```

<p>6. Run the Final Model</p>

```
# For classification:
python3 main.py gb stats prepro corel

# For clustering:
python3 clusteringMain.py HAC 2 mm
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   Scikit-learn
*   Pandas
*   Numpy
*   Matplotlib
