from src.pipeline import PipelineConfig
from src.utils import COLS

pipeline_config = PipelineConfig(
    cols_to_drop=[COLS.ID_GRID, COLS.SUP_TOTAL],
    feature_creator_params={
        "total_col": COLS.SUP_TOTAL,
        "constr_col": COLS.SUP_CONSTR,
        "new_col_name": COLS.SUP_DESCUBIERTA,
    },
    median_imputer_cols=[COLS.ANTIGUEDAD, COLS.SUP_DESCUBIERTA, COLS.SUP_CONSTR],
    outlier_clipper_params={
        "cols_to_clip": [
            COLS.SUP_CONSTR,
            COLS.SUP_DESCUBIERTA,
            COLS.ANTIGUEDAD,
            COLS.DORMITORIOS,
            COLS.BANOS,
            COLS.AMBIENTES,
            COLS.COCHERAS,
            COLS.TARGET,
        ],
        "upper_pct": 0.98,
    },
    discrete_numeric_cols=[COLS.AMBIENTES, COLS.DORMITORIOS, COLS.BANOS, COLS.COCHERAS],
    continuous_numeric_cols=[
        COLS.SUP_CONSTR,
        COLS.SUP_DESCUBIERTA,
        COLS.ANTIGUEDAD,
    ],
    standard_scale_cols=[
        COLS.LONGITUD,
        COLS.LATITUD,
    ],
    boolean_imputer_cols=[
        COLS.AMOBLADO,
        COLS.GIMNASIO,
        COLS.LAUNDRY,
        COLS.CALEFACCION,
        COLS.AIRE,
        COLS.SEGURIDAD,
        COLS.PILETA,
        COLS.TENNIS,
        COLS.RECEPCION,
        COLS.BUSINESS,
        COLS.ESTACIONAMIENTO,
        "SUM",
        COLS.JACUZZI,
    ],
    ohe_cols=[COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO],
    target_clipper_params={"cols_to_clip": [COLS.TARGET], "upper_pct": 0.98},
)
