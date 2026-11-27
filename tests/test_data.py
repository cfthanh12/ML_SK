"""TC-D: checks on the real dataset in data/raw (skipped if the CSV is absent)."""

import pytest
from sklearn.model_selection import train_test_split

from app.features import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from app.train import DEFAULT_DATA_PATH, load_dataset, prepare_dataset

pytestmark = pytest.mark.skipif(not DEFAULT_DATA_PATH.is_file(), reason="dataset not present")


@pytest.fixture(scope="module")
def raw():
    return load_dataset(DEFAULT_DATA_PATH)


def test_d01_schema_and_size(raw) -> None:
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(raw.columns)
    assert len(raw) == 5000


def test_d02_no_missing_values(raw) -> None:
    assert not raw[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()


def test_d03_target_on_0_10_scale(raw) -> None:
    assert raw[TARGET_COLUMN].between(0, 10).all()


def test_d04_negative_activity_clipped_without_touching_raw(raw) -> None:
    before = int((raw["Physical_Activity_Hours"] < 0).sum())
    features, _, report = prepare_dataset(raw)
    assert before == report["negative_physical_activity_values_clipped"] == 10
    assert (features["Physical_Activity_Hours"] >= 0).all()
    assert int((raw["Physical_Activity_Hours"] < 0).sum()) == before


def test_d05_duplicates_reported(raw) -> None:
    _, _, report = prepare_dataset(raw)
    assert report["duplicate_rows_kept"] == 2


def test_d06_train_test_split_disjoint(raw) -> None:
    features, target, _ = prepare_dataset(raw)
    x_train, x_test, _, _ = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert (len(x_train), len(x_test)) == (4000, 1000)


def test_d07_regional_platforms_are_tied_to_one_country(raw) -> None:
    """Confounding check: some platforms appear in only one country, so their mean score
    reflects that country, and the report must not claim "platform X is better"."""
    regional = ["LINE", "KakaoTalk", "WeChat", "VKontakte"]
    countries = raw[raw["Most_Used_Platform"].isin(regional)].groupby("Most_Used_Platform")["Country"].nunique()
    assert set(countries.index) == set(regional)
    assert (countries == 1).all()
