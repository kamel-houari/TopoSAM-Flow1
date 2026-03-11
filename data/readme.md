# Data Preparation for TopoSAM-Flow

## Datasets

### NEU Surface Defect Database
- **Download**: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_dataset.html
- **Annotation**: Bounding boxes (weak supervision)
- **Classes**: 6 defect types (rolling_in, patches, crazing, pitted_surface, inclusion, scratches)
- **Images**: 1,800 grayscale images (200×200)

### RSDDs Rail Defect Dataset
- **Download**: http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx
- **Annotation**: Pixel-level masks (for evaluation)
- **Classes**: Crack defects
- **Images**: 195 images (Type-I: 67, Type-II: 128)

### MVTec AD
- **Download**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Annotation**: Pixel-level anomaly masks
- **Categories**: 15 (5 textures + 10 objects)
- **Images**: 5,354 total (3,629 train, 1,725 test)

## Preparation Scripts

Run the preparation scripts to download and organize datasets:

```bash
# NEU
python data/prepare_neu.py --download --output ./data/NEU

# RSDDs
python data/prepare_rsdds.py --download --output ./data/RSDDs

# MVTec AD (requires manual download first)
python data/prepare_mvtec.py --input /path/to/mvtec_download --output ./data/MVTec_AD