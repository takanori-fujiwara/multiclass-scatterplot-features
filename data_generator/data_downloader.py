import gzip
import pickle
import requests

from io import BytesIO
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

import openml

import ucimlrepo
import json
import urllib.request
import urllib.parse
import certifi
import ssl

# local files
import utils


def download_openml_data(
    queries=[
        "NumberOfClasses >= 2",
        "NumberOfInstances < 10000",
        "NumberOfMissingValues == 0",
    ],
    output_dir="original_data_openml",
    drop_duplicates=True,
    clean_data=True,
):
    # TODO: add more data by allowing missing values and applying mean imputation
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datalist = openml.datasets.list_datasets(output_format="dataframe")
    datalist = datalist.query(" and ".join(queries))

    if drop_duplicates:
        # avoid duplicated data (naive judge based on n_instances and n_features)
        datalist = datalist.drop_duplicates(
            subset=["NumberOfInstances", "NumberOfFeatures"], keep="last"
        )

    for i, did in enumerate(datalist["did"]):
        print(f"{i} / {len(datalist)}")
        dataset = openml.datasets.get_dataset(did)

        # handle cases with more than two target attrs
        target_attr = dataset.default_target_attribute.split(",")[0]
        if did == 42716:
            target_attr = "class"  # they said "y" but actuall "class"
        elif did in [43147, 43149]:
            continue  # these datasets have weird esc seq \w and cannot be parsed

        X, y, categorical_indicator, attr_names = dataset.get_data(target=target_attr)
        if clean_data:
            X = utils.clean_data(X, categorical_indicator=categorical_indicator)
            # avoid cases there is only one attribute
            if X.shape[1] < 2:
                continue

        attr_names = np.array(X.columns)
        X = np.array(X)

        y = y.astype("category")
        target_names = y.cat.categories
        y = np.array(y.cat.codes)

        original_data = {
            "openml_id": did,
            "dataset": dataset.name,
            "url": dataset.url,
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/openml_{did}.pkl", "wb") as f:
            pickle.dump(original_data, f)


def _ucimlrepo_available_datasets(
    filter: Optional[str] = None,
    search: Optional[str] = None,
    area: Optional[str] = None,
):
    """Version to return dataframe instead of just print"""
    # validate filter input
    if filter:
        if type(filter) != str:
            raise ValueError("Filter must be a string")
        filter = filter.lower()

    # validate search input
    if search:
        if type(search) != str:
            raise ValueError("Search query must be a string")
        search = search.lower()

    # construct endpoint URL
    api_list_url = "https://archive.ics.uci.edu/api/datasets/list"
    query_params = {}
    if filter:
        query_params["filter"] = filter
    else:
        query_params["filter"] = "python"  # default filter should be 'python'
    if search:
        query_params["search"] = search
    if area:
        query_params["area"] = area

    api_list_url += "?" + urllib.parse.urlencode(query_params)

    # fetch list of datasets from API
    data = None
    try:
        response = urllib.request.urlopen(
            api_list_url, context=ssl.create_default_context(cafile=certifi.where())
        )
        resp_json = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError):
        raise ConnectionError("Error connecting to server")

    if resp_json["status"] != 200:
        error_msg = (
            resp_json["message"] if "message" in resp_json else "Internal Server Error"
        )
        raise ValueError(resp_json["message"])

    data = resp_json["data"]

    if len(data) == 0:
        print("No datasets found")
        return

    return pd.DataFrame(data)


def _download_ucimlrepo_exceptional_data(data_id, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    X = None
    if data_id == 45:
        name = "heart+disease"
        url = f"https://archive.ics.uci.edu/static/public/{data_id}/{name}.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"processed.cleveland.data") as f:
                X = pd.read_csv(f, header=None)
                X = X.replace("?", np.nan).dropna()

                y = X.iloc[:, -1].astype("category")
                target_names = y.cat.categories
                y = np.array(y.cat.codes)

                X = X.iloc[:, :-1]
                attr_names = np.array(
                    [
                        "age",
                        "sex",
                        "cp",
                        "trestbps",
                        "chol",
                        "fbs",
                        "restecg",
                        "thalach",
                        "exang",
                        "oldpeak",
                        "slope",
                        "ca",
                        "thal",
                    ]
                )
                X = np.array(X)
    elif data_id == 166:
        name = "hill+valley"
        url = f"https://archive.ics.uci.edu/static/public/{data_id}/{name}.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"Hill_Valley_without_noise_Training.data") as f:
                training = pd.read_csv(f)
            with zf.open(f"Hill_Valley_without_noise_Testing.data") as f:
                testing = pd.read_csv(f)

            X = pd.concat((training, testing))

            y = X["class"].astype("category")
            target_names = y.cat.categories
            y = np.array(y.cat.codes)

            X = X.drop(columns=["class"])
            attr_names = np.array(X.columns)
            X = np.array(X)
    elif data_id == 181:
        name = "libras+movement"
        url = f"https://archive.ics.uci.edu/static/public/{data_id}/{name}.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"movement_libras.data") as f:
                X = pd.read_csv(f, header=None)

                y = X.iloc[:, -1].astype("category")
                target_names = y.cat.categories
                y = np.array(y.cat.codes)

                X = X.iloc[:, :-1]
                attr_names = np.array(X.columns)
                X = np.array(X)
    elif data_id == 236:
        name = "seeds"
        url = f"https://archive.ics.uci.edu/static/public/{data_id}/{name}.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"seeds_dataset.txt") as f:
                X = pd.read_csv(f, sep=r"\s+", header=None)

                y = X.iloc[:, -1].astype("category")
                target_names = y.cat.categories
                y = np.array(y.cat.codes)

                X = X.iloc[:, :-1]
                attr_names = np.array(
                    [
                        "area",
                        "perimeter",
                        "compactness",
                        "length of kernel",
                        "width of kernel",
                        "asymmetry coefficient",
                        "length of kernel groove",
                    ]
                )
                X = np.array(X)
    elif data_id == 333:
        name = "forest+type+mapping"
        url = f"https://archive.ics.uci.edu/static/public/{data_id}/{name}.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"training.csv") as f:
                training = pd.read_csv(f)
            with zf.open(f"testing.csv") as f:
                testing = pd.read_csv(f)

            X = pd.concat((training, testing))

            y = X["class"].astype("category")
            target_names = y.cat.categories
            y = np.array(y.cat.codes)

            X = X.drop(columns=["class"])
            attr_names = np.array(X.columns)
            X = np.array(X)

    if X is not None:
        original_data = {
            "ucimlrepo_id": data_id,
            "dataset": name,
            "url": url,
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/ucimlrepo_{data_id}.pkl", "wb") as f:
            pickle.dump(original_data, f)


def download_ucimlrepo_data(
    tasks=["Classification"],
    n_instances_range=[0, 9999],
    accept_missing_values=False,
    output_dir="original",
    clean_data=True,
    data_ids=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    datalist = _ucimlrepo_available_datasets()
    if data_ids is None:
        data_ids = datalist["id"]

    for i, did in enumerate(data_ids):
        print(f"{i} / {len(data_ids)}")
        try:
            # uci repo can cause an issue related to tokenizing
            dataset = ucimlrepo.fetch_ucirepo(id=did)
        except:
            continue

        tasks_satisfied = True in [task in dataset.metadata.tasks for task in tasks]
        n_instances_satisfied = (
            dataset.metadata.num_instances >= n_instances_range[0]
        ) and (dataset.metadata.num_instances <= n_instances_range[1])
        missing_values_satisfied = accept_missing_values or (
            dataset.metadata.has_missing_values == "no"
        )

        if tasks_satisfied and n_instances_satisfied and missing_values_satisfied:
            X = dataset.data.features
            y = dataset.data.targets

            # even some cases no class info for classification task
            if y is None:
                if "Classification" in tasks:
                    continue
                else:
                    y = pd.DataFrame(np.zeros(X.shape[0], dtype=int))

            if clean_data:
                X = utils.clean_data(X)
                # avoid cases there is only one attribute
                if X.shape[1] < 2:
                    continue

            attr_names = np.array(X.columns)
            X = np.array(X)

            y = y.iloc[:, 0].astype("category")
            target_names = y.cat.categories
            y = np.array(y.cat.codes)

            original_data = {
                "ucimlrepo_id": did,
                "dataset": dataset.metadata.name,
                "url": dataset.metadata.data_url,
                "X": X,
                "y": y,
                "attribute_names": attr_names,
                "target_names": target_names,
            }

            with open(f"{output_dir}/ucimlrepo_{did}.pkl", "wb") as f:
                pickle.dump(original_data, f)

    for i, did in enumerate(data_ids):
        _download_ucimlrepo_exceptional_data(did, output_dir=output_dir)


def download_visumap_data(
    files=[
        "Sp500_2006.xvmz",
        # "SP500_2008_2012.xvmz", # cannot decompress
        "CanadianEquity.xvmz",
        # "ETF2006.xvmz", # cannot find
        "InternationalIndices.xvmz",
        "TSE300.xvmz",
        "Currency.xvmz",
        "Industries.xvmz",
        "EcoliProteins.xvmz",
        "ItalianWine.xvmz",
        # "yeast.xvmz",  # cannot find
        "AtomStructure.xvmz",
        "FisherIris.xvmz",
        "FunctionalGenomics.xvmz",
        "Haplotype2M.xvmz",
        "WorldPopulation.xvmz",
        "CalgaryPopulation.xvmz",
        "ADA.xvmz",
        # "MusicNewsgroups.xvmz", # no class structure
        "WorldMap.xvmz",
        "Meteorology.xvmz",
        "UnEvenDensity.xvmz",
        "DoubleSphere.xvmz",
        "SwissRoll.xvmz",
        "TwoSquares.xvmz",
        "KleinBottle4D.xvmz",
        "Spheroid.xvmz",
        "NetSystemClasses.xvmz",
    ],
    n_instances_range=[0, 9999],
    min_n_classes=2,
    output_dir="original_data_visumap",
    clean_data=True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    urls = [f"https://visumap.com/images/{filename}" for filename in files]
    for i, url in enumerate(urls):
        print(f"{i} / {len(urls)}: {url}")
        r = requests.get(url)

        xml_string = gzip.decompress(r.content)
        root = ElementTree.fromstring(xml_string)
        ns = root.tag.split("}")[0].strip("{")  # namespace

        dataset = (
            root.find(f"{{{ns}}}DatasetList")
            .find(f"{{{ns}}}Dataset")
            .find(f"{{{ns}}}Table")
        )
        attr_names = np.array(
            [elm.attrib["Id"] for elm in dataset.findall(f"{{{ns}}}Column")]
        )
        attr_dtypes = {}
        for name, dtype in zip(
            attr_names,
            [elm.attrib["Type"] for elm in dataset.findall(f"{{{ns}}}Column")],
        ):
            attr_dtypes[name] = float if dtype != "e" else object

        inst_names = np.array(
            [elm.attrib["Id"] for elm in dataset.findall(f"{{{ns}}}Row")]
        )
        y = []
        # NOTE: VisuMap data might have different class tag from "Name" or "Type"
        for elm in dataset.findall(f"{{{ns}}}Row"):
            if "Name" in elm.attrib:
                y.append(elm.attrib["Name"])
            elif "Type" in elm.attrib:
                y.append(elm.attrib["Type"])
            else:
                y.append("none")
        y = pd.DataFrame(y)

        X = pd.DataFrame(
            [elm.text.split("|") for elm in dataset.findall(f"{{{ns}}}Row")],
            columns=attr_names,
        )
        X = X.astype(attr_dtypes)

        if clean_data:
            X = utils.clean_data(X)
            # avoid cases there is only one attribute
            if X.shape[1] < 2:
                continue

        if (X.shape[0] < n_instances_range[0]) or (X.shape[0] > n_instances_range[1]):
            continue

        attr_names = np.array(X.columns)
        X = np.array(X)

        y = y.iloc[:, 0].astype("category")
        target_names = y.cat.categories
        y = np.array(y.cat.codes)
        if len(np.unique(y)) < min_n_classes:
            continue

        original_data = {
            "visumap_id": i,
            "dataset": url.split("/")[-1].strip(".xvmz"),
            "url": url,
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/visumap_{i}.pkl", "wb") as f:
            pickle.dump(original_data, f)


def download_clm_data(
    files=[
        "banknote_authentication",
        "birds_bones_and_living_habits",
        "blood_transfusion_service_center",
        "boston",
        "breast_cancer_coimbra",
        "breast_cancer_wisconsin_original",
        "breast_cancer_wisconsin_prognostic",
        "breast_tissue",
        "cardiovascular_study",
        "cifar10",
        "classification_in_asteroseismology",
        "cnae9",
        "coil20",
        "credit_risk_classification",
        "crowdsourced_mapping",
        "customer_classification",
        "date_fruit",
        "dermatology",
        "diabetic_retinopathy_debrecen",
        "dry_bean",
        "durum_wheat_features",
        "echocardiogram",
        "ecoli",
        "epileptic_seizure_recognition",
        "extyaleb",
        "fashion_mnist",
        "fetal_health_classification",
        "flickr_material_database",
        "fraud_detection_bank",
        "glass_identification",
        "har",
        "harbermans_survival",
        "hate_speech",
        "heart_attack_analysis_prediction_dataset",
        "heart_disease",
        "hepatitis",
        "hiva",
        "htru2",
        "human_stress_detection",
        "image_segmentation",
        "imdb",
        "insurance_company_benchmark",
        "ionosphere",
        "iris",
        "labeled_faces_in_the_wild",
        "letter_recognition",
        "magic_gamma_telescope",
        "mammographic_mass",
        "microbes",
        "mnist64",
        "mobile_price_classification",
        "music_genre_classification",
        "olivetti_faces",
        "optical_recognition_of_handwritten_digits",
        "orbit_classification_for_prediction_nasa",
        "paris_housing_classification",
        "parkinsons",
        "patient_treatment_classification",
        "pen_based_recognition_of_handwritten_digits",
        "ph_recognition",
        "pima_indians_diabetes_database",
        "pistachio",
        "planning_relax",
        "predicting_pulsar_star",
        "pumpkin_seeds",
        "raisin",
        "rice_dataset_cammeo_and_osmancik",
        "rice_seed_gonen_jasmine",
        "secom",
        "seeds",
        "seismic_bumps",
        "sentiment_labeld_sentences",
        "siberian_weather_stats",
        "skillcraft1_master_table_dataset",
        "smoker_condition",
        "sms_spam_collection",
        "spambase",
        "spectf_heart",
        "statlog_german_credit",
        "statlog_image_segmentation",
        "street_view_house_numbers",
        "student_grade",
        "taiwanese_bankruptcy_prediction",
        "turkish_music_emotion",
        "user_knowledge_modeling",
        "water_quality",
        "weather",
        "website_phishing",
        "wilt",
        "wine",
        "wine_customer",
        "wine_quality",
        "wireless_indoor_localization",
        "world12d",
        "yeast",
        "zoo",
    ],
    output_dir="original_data_clm",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    url_base = (
        "https://github.com/hj-n/ext-clustering-validation-datasets/raw/master/npy/"
    )

    for i, filename in enumerate(files):
        print(f"{i} / {len(files)}")
        url = f"{url_base}{filename}_npy.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"{filename}/label.npy") as f:
                y = np.load(BytesIO(f.read()))
                y = pd.DataFrame(y)
                y = y.iloc[:, 0].astype("category")
                target_names = y.cat.categories
                y = np.array(y.cat.codes)
            with zf.open(f"{filename}/data.npy") as f:
                X = np.load(BytesIO(f.read()))

        original_data = {
            "clm_id": i,
            "dataset": filename,
            "url": url,
            "X": X,
            "y": y,
            "attribute_names": np.arange(X.shape[1]),
            "target_names": target_names,
        }

        with open(f"{output_dir}/clm_{i}.pkl", "wb") as f:
            pickle.dump(original_data, f)


def download_misc_data(
    dataset_names=["swanson", "cereal", "digits5-9"], output_dir="original_data_misc"
):
    # xmdv datasets listed in https://www.cs.ubc.ca/labs/imager/tr/2013/ScatterplotEval/ScatterplotEval_Suppl.pdf
    # and sklearn datasets listed Supp. of Wang et al., A Perception-Driven Approach to Supervised Dimensionality Reduction for Visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. swanson
    if "swanson" in dataset_names:
        url = "http://davis.wpi.edu/~xmdv/datasets/swanson.zip"
        with ZipFile(BytesIO(requests.get(url).content)) as zf:
            with zf.open(f"swanson.okc") as f:
                content = f.read().decode()

        lines = content.splitlines()
        n_attrs, n_insts = list(map(int, lines[0].split()))
        attr_names = np.array(lines[1 : 1 + n_attrs])
        X = pd.DataFrame(
            [
                list(map(float, line.split()[:n_attrs]))
                for line in lines[1 + n_attrs * 2 : 1 + n_attrs * 2 + n_insts]
            ],
            columns=attr_names,
        )

        y = X["b_commw"].astype("category")
        target_names = y.cat.categories
        y = np.array(y.cat.codes)
        X = np.array(X.drop(columns=["b_commw"]))

        original_data = {
            "misc_id": 0,
            "dataset": "swanson",
            "url": url,
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/misc_0.pkl", "wb") as f:
            pickle.dump(original_data, f)

    # 2. cereal
    if "cereal" in dataset_names:
        url = "https://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html"
        content = requests.get(url).content.decode()
        lines = content.split("<PRE>\n")[1].split("</PRE>")[0].splitlines()
        data = np.array([line.split() for line in lines])
        attr_names = data[0]
        X = pd.DataFrame(data[1:], columns=attr_names)

        y = X["mfr"].astype("category")
        target_names = y.cat.categories
        y = np.array(y.cat.codes)

        X = np.array(X.drop(columns=["name", "mfr", "type", "rating"]))

        original_data = {
            "misc_id": 1,
            "dataset": "cereals",
            "url": url,
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/misc_1.pkl", "wb") as f:
            pickle.dump(original_data, f)

    # 3. digits5-9
    if "digits5-9" in dataset_names:
        # Note: Although Wang et al. stated digits5-8, Supp results show 5 classes. So, should be 5-9
        from sklearn.datasets import load_digits

        digits = load_digits()
        X = digits.data
        y = digits.target
        attr_names = digits.feature_names

        X = X[y >= 5]
        y = y[y >= 5]
        target_names = ["5", "6", "7", "8", "9"]
        y = y - 5  # to start from "5"

        original_data = {
            "misc_id": 2,
            "dataset": "digits5-9",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits",
            "X": X,
            "y": y,
            "attribute_names": attr_names,
            "target_names": target_names,
        }

        with open(f"{output_dir}/misc_2.pkl", "wb") as f:
            pickle.dump(original_data, f)


# def download_xmdv_data(
#     files=[
#         "aaup.zip",
#         "acorns.zip",
#         "amexa.zip",
#         "astronomy.zip",
#         "cars.zip",
#         "census_income.zip",
#         "cereal.zip",
#         "coal_disasters.zip",
#         "detroit.zip",
#         "energy.zip",
#         "glasglow.zip",
#         "homicide.zip",
#         "homicideClean.zip",
#         "htong.zip",
#         "iris.zip",
#         "nasdaqg.zip",
#         "netperf.zip",
#         "niche-ll.zip",
#         "ohsumed.zip",
#         "out5d.zip",
#         "parents.zip",
#         "poverty.zip",
#         "rubber.zip",
#         "scanbio.zip",
#         "scanbio2.zip",
#         "scanbio3.zip",
#         "skyserver.zip",
#         "subway.zip",
#         "supercos.zip",
#         "supercos2.zip",
#         "swanson.zip",
#         "synt2k.zip",
#         "synt16.zip",
#         "ticdata2000.zip",
#         "usmoney.zip",
#         "uspop.zip",
#         "uvw.zip",
#         "venus.zip",
#         "voy.zip",
#         "webstats.zip",
#     ],
#     n_instances_range=[0, 9999],
#     min_n_classes=2,
#     output_dir="original_data_xmdv",
#     clean_data=True,
# ):


#     urls = [f"https://davis.wpi.edu/~xmdv/datasets/{filename}" for filename in files]
#     for i, url in enumerate(urls):
#         print(f"{i} / {len(urls)}: {url}")
#         with ZipFile(BytesIO(requests.get(url).content)) as zf:
#             with zf.open(f"{filename.strip('.zip')}.okc") as f:
#                 content = f.read().decode()

#         lines = content.splitlines()
#         n_attrs, n_insts = list(map(int, lines[0].split()))
#         if (n_insts < n_instances_range[0]) or (n_insts > n_instances_range[1]):
#             continue

#         attr_names = np.array(lines[1 : 1 + n_attrs])
#         min_max_info = np.array(
#             [
#                 list(map(float, line.split()))
#                 for line in lines[1 + n_attrs : 1 + n_attrs * 2]
#             ]
#         )

#         X = np.array(
#             [
#                 list(map(float, line.split()[:n_attrs]))
#                 for line in lines[1 + n_attrs * 2 : 1 + n_attrs * 2 + n_insts]
#             ]
#         )


def download_sedlmair2012_scatter_data(
    output_dir="scatter_data_binarized_sedlmair2012",
    output_scatter_info_path="scatter_info_sedlmair2012.csv",
):
    """
    Download data uded for Sedlmair2012, Sedlmair2013, Sedlmair 2015, Aupetit 2016.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url = "https://sepme.vda.univie.ac.at/SRC/EUROVIS_data.zip"
    scatter_info = []
    assign_id = utils.id_generator()

    with ZipFile(BytesIO(requests.get(url).content)) as zf:
        for name in zf.namelist():
            if Path(name).suffix == ".csv":
                with zf.open(name) as f:
                    scatter_data = pd.read_csv(f, header=None)
                    scatter_data.columns = ["x", "y", "label"]

                    stem_name = Path(name).stem
                    data_name = stem_name.split("_")[0]
                    dr_name = stem_name.split("_")[1]

                    # NOTE: Boston Housing should not be included due to the ethical reason https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html
                    if data_name == "boston":
                        continue

                    scatter_id = assign_id()
                    scatter_data.to_csv(f"{output_dir}/{scatter_id}.csv", index=False)
                    scatter_info.append(
                        {
                            "id": scatter_id,
                            "data": data_name,
                            "dr": dr_name,
                            "scaler": "unknown",
                        }
                    )
    pd.DataFrame(scatter_info).to_csv(output_scatter_info_path, index=False)


def download_wang2018_data(output_dir="original_data_wang2018"):
    # 1 Abalone, 12 Balance Scale, 15 Breast Cancer Wisconsin (Original)
    # 30 Contraceptive Method Choice (cmc), 33 Dermatology, 44 Hayes-Roth, 52 Ionosphere, 53 Iris, 58 Lenses
    # 78 Page Blocks Classification, 94 Spambase, 107 Waveform Database Generator (Version 1)
    # 109 Wine, 110	Yeast, 111 Zoo
    # 146 satimage, 147 Statlog (Image Segmentation), 148 Statlog (Shuttle) (separated into two sizes in wang2018)
    # 166 Hill-Valle, 181 movement-1, 236 seeds, 333 Forest-Typ
    # 17 breast+cancer+wisconsin+diagnostic ("wbc-clas" in Wang2018)
    # 45 heart+disease ("processed-"" in Wang2018)
    # (duplicated 78 page+blocks+classification ("white-ball" in Wang2018))

    # NOTE cannot find: combined-s, Connection, italianwin, scene, test--int, white-ball
    download_ucimlrepo_data(
        data_ids=[
            1,
            12,
            15,
            17,
            30,
            33,
            44,
            45,
            52,
            53,
            58,
            78,
            94,
            107,
            109,
            110,
            111,
            146,
            147,
            148,
            166,
            181,
            236,
            333,
        ],
        output_dir=output_dir,
        clean_data=True,
    )
    download_misc_data(dataset_names=["cereal", "digits5-9"], output_dir=output_dir)

    # NOTE: WorldPopulation is used as 11D and 9D in wang2018 but original one has 12D
    download_visumap_data(
        files=[
            "EcoliProteins.xvmz",
            # "MusicNewsgroups.xvmz", # removed: no class structure
            "TSE300.xvmz",
            "WorldMap.xvmz",
            "WorldPopulation.xvmz",
        ],
        output_dir=output_dir,
    )

    # pima-india
    download_openml_data(queries=["did == 43483"], output_dir=output_dir)
