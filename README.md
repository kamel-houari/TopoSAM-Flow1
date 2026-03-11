# TopoSAM-Flow1
Weakly-Supervised Industrial Image Segmentation via Differentiable Variational Flow and Topological Priors
# TopoSAM-Flow

**Weakly-Supervised Industrial Image Segmentation via Differentiable Variational Flow and Topological Priors**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)

## 📋 Description

TopoSAM-Flow est une méthode de segmentation d'images industrielles faiblement supervisée qui combine:
- **SAM (Segment Anything Model)** pour la génération de pseudo-masques
- **Régularisation variationnelle différentiable** inspirée des modèles à contours actifs
- **Contrainte topologique** via l'homologie persistante pour garantir la continuité des défauts

## 📊 Résultats

| Dataset | mIoU | FPS | Annotation Cost Reduction |
|---------|------|-----|--------------------------|
| NEU | 78.9% | 72 | 91% |
| MVTec AD | 89.7% | 71 | 91% |
| RSDDs | 84.6% | 68 | 91% |

## 🚀 Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/KHouari/TopoSAM-Flow.git
cd TopoSAM-Flow

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
