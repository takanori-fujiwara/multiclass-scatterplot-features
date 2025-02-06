# Install required pacakges:
# >> pip3 install -r requirements.txt
# Install other packages by following the websites below:
# - ccpca: https://github.com/takanori-fujiwara/ccpca
# - phate: https://github.com/KrishnaswamyLab/PHATE

# NOTE: Python3.13 cannot install phate
# NOTE: But also, if using Python3.12, pandas has conflict with numpy 2.0 => pip3 install numpy==1.26.4

if __name__ == "__main__":
    # 1. Downloading data from OpenML
    from data_downloader import (
        download_openml_data,
        download_ucimlrepo_data,
        download_visumap_data,
        download_clm_data,
        download_misc_data,
        download_sedlmair2012_scatter_data,
    )

    download_openml_data(output_dir="original_data")
    download_ucimlrepo_data(output_dir="original_data")
    download_visumap_data(output_dir="original_data")
    download_clm_data(output_dir="original_data")
    download_misc_data(output_dir="original_data")
    download_sedlmair2012_scatter_data(output_dir="scatter_data")

    # 2. Generating scatter data and plots
    from scatter_data_generator import ScatterDataGenerator

    sdg = ScatterDataGenerator()
    sdg.run_all()

    # 3. Compute scatter metainfo (e.g., SepMe)
    from metadata_generator import MetadataGenerator

    mdg = MetadataGenerator()
    # mdg.extract_scatter_metadata(start_index=0)
    mdg.extract_scatter_metadata()
