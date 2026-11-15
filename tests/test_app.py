"""Streamlit smoke tests: every page must render without exceptions (needs `python train.py` first)."""
import pytest

import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["app.py", "pages/1_Demand_Forecasting.py", "pages/2_Expiry_Risk.py",
         "pages/3_Inventory_Optimization.py", "pages/4_Model_Performance.py", "pages/5_Data_Management.py"]

needs_db = pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `python train.py` first")


@needs_db
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page):
    at = AppTest.from_file(str(config.BASE_DIR / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


@needs_db
def test_forecast_page_medicine_switch():
    at = AppTest.from_file(str(config.BASE_DIR / "pages/1_Demand_Forecasting.py"), default_timeout=60).run()
    at.selectbox[0].select_index(5).run()
    assert not at.exception
