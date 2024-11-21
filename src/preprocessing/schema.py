from schema import Schema, SchemaError, Regex, And, Or, Optional, Use
import yaml
import logging

logger = logging.getLogger('ErNESTO-DT')

schemas = {}

string_pattern = Regex(r'^[a-zA-Z0-9_. ]+$',
                       error="Error in string '{}': it can only have a-z, A-Z, 0-9, and _.")
path_pattern = Regex(r'^[a-zA-Z0-9_\-./]+$',
                     error="Error in path '{}': it can only have a-z, A-Z, 0-9, ., / and _.")
class_pattern = Regex(r'^[a-zA-Z0-9]+$',
                      error="Error in class name '{}': it can only have a-z, A-Z and 0-9.")
var_pattern = Regex(r'^[a-z_]+$',
                    error="Error in variable '{}': it can only have a-z and _.")
label_pattern = Regex(r'^[a-zA-Z0-9_\[\]() ]+$',
                      error="Error in label '{}': it can only have a-z, A-Z, [,], and _.")
unit_pattern = Regex(r'^[a-zA-Z_]+$',
                     error="Error in unit identifier '{}': it can only have a-z, A-Z.")

ground_data = Schema(
    {
        # Ground data structure
        "file": And(str, string_pattern),
        "load": And(str, var_pattern),
        "time_format": Or('seconds', 'timestamp'),
        "vars": [
            Or(
                {
                    "var": And(str, var_pattern, Use(str.lower)),
                    "label": And(str, label_pattern),
                    "unit": And(str, unit_pattern)
                }
            )
        ],
        Optional("cycle_for"): Or(int, None)
    }
)

schedule = Schema(
    {
        "instructions": [Or(str, string_pattern)],
        Optional("constants"): {And(str, var_pattern): Or(float, And(int, Use(float)))} 
    }
)

battery_param = Schema(
    {
        "var": And(str, var_pattern, Use(str.lower)),
        "value": Or(float, int),
        "unit": And(str, unit_pattern)
    }
)

bound_param = Schema(
    {        
        "low": Or(float, And(int, Use(float))),
        "high": Or(float, And(int, Use(float)))
    }
)

battery = Schema(
    {
        "sign_convention": Or('active', 'passive'),
        "params": {And(str, var_pattern): battery_param},
        Optional("bounds"): {And(str, var_pattern): bound_param},
        "init":
            {
                Optional('voltage'): Or(float, And(int, Use(float))),
                Optional('current'): Or(float, And(int, Use(float))),
                'temperature': Or(float, And(int, Use(float))),
                "soc": And(Or(float, And(int, Use(float))), lambda n: 0 <= n <= 1),
                "soh": And(Or(float, And(int, Use(float))), lambda n: 0 <= n <= 1),
            },
        Optional("reset_soc_every"): Or(int, None)
    }
)

optimizer = Schema(
    {
        "algorithm": str,
        "max_iter": Or(int, None),
        "tol": Or(float, None),
        "alpha": Or(float, None),
        "batch_size": Or(int, None),
        "n_restarts": Or(int, None),
        "search_bounds": {And(str, label_pattern): bound_param},
        "scale_factors": {And(str, label_pattern): Or(float, int)},
    }
)

adaptation = Schema(
    {   
        "grid": [{
            "cluster": And(str, path_pattern),
            "region" : {
                And(str, var_pattern): bound_param},    
        }]
        
    }
)

config_schema = Schema(
    {
        # Summary
        Optional("experiment_name"): And(str, string_pattern),
        Optional("description"): And(str),
        Optional("goal"): And(str),
        "destination_folder": And(str, string_pattern),
        
        # Ground data options
        "input": {
            Optional("ground_data"): ground_data,
            Optional("schedule"): schedule,
        },
        # Simulation options
        Optional("optimizer"): optimizer,
        Optional("adaptation"): adaptation,
        Optional("iterations"): Or(int, None),
        Optional("timestep"): Or(int, float, None),
        Optional("check_soh_every"): Or(int, None),
        Optional("clear_collections_every"): Or(int, None),
        # Battery parameters
        "battery": battery,
    }
)

asset = Schema(
    {
        "category": And(str, var_pattern),
        "file": Or(None, And(str, path_pattern))
    }
)

assets_schema = Schema(
    {
        "models_path": And(str, path_pattern),
        "models": {
            str: asset,
        }
    }
)

single_comp_hardcoded_lookup = Schema(
    {
        "selected_type": Or('scalar', 'lookup'),
        Optional("scalar"): Or(float, And(int, Use(float))),
        Optional("lookup"): {
            "inputs": {
                Optional('temp'): [Or(float, int)],
                Optional('soc'): [And(Or(float, And(int, Use(float))), lambda n: 0 <= n <= 1)],
                Optional('soh'): [And(Or(float, And(int, Use(float))), lambda n: 0 <= n <= 1)],
            },
            "output": [Or(float, And(int, Use(float)))]
        }
    },
)

single_comp_csv_lookup = Schema(
    {
        "selected_type": Or('scalar', 'lookup'),
        Optional("scalar"): Or(float, And(int, Use(float))),
        Optional("lookup"): {
            "table": And(str, path_pattern),
            "inputs": [{
                "var": And(str, var_pattern),
                "label": And(str, label_pattern),
                "unit": Or(And(str, unit_pattern), None)
            }],
            "output": {
                "var": And(str, var_pattern),
                "label": And(str, label_pattern),
                "unit": Or(And(str, unit_pattern), None)
            }
        }
    },
)

first_order_thevenin = Schema(
    {   # First Order Thevenin
        "r0": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "r1": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "c": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "v_ocv": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup)
    }
)

second_order_thevenin = Schema(
    {   # Second Order Thevenin
        "r0": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "r1": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "c1": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "r2": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "c2": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "v_ocv": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup)
    }
)

rc_thermal = Schema(
    {  # RC_thermal
        "r_term": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
        "c_term": Or(single_comp_csv_lookup, single_comp_hardcoded_lookup),
    }
)

r2c_thermal = Schema(
    {
        Optional("lambda"): single_comp_hardcoded_lookup,
        Optional("length"): single_comp_hardcoded_lookup,
        Optional("area_int"): single_comp_hardcoded_lookup,
        Optional("area_surf"): single_comp_hardcoded_lookup,
        Optional("h"): single_comp_hardcoded_lookup,
        Optional("mass"): single_comp_hardcoded_lookup,
        Optional("cp"): single_comp_hardcoded_lookup,
        "c_term": single_comp_hardcoded_lookup,
        "r_cond": single_comp_hardcoded_lookup,
        "r_conv": single_comp_hardcoded_lookup,
        "dv_dT": single_comp_hardcoded_lookup
    }
)

mlp_thermal = Schema(
    {  # RC_thermal
        "input_size": And(int),
        "hidden_size": And(int),
        "output_size": And(int),
        "model_state": And(str, path_pattern),
        "scaler": And(str, path_pattern),
        "cuda": Or(False, True)
    }
)

bolun = Schema(
    {  # Bolun
        "SEI": {
            "alpha_sei": Or(float, And(int, Use(float))),
            "beta_sei": Or(float, And(int, Use(float))),
        },
        "stress_factors": {
            "calendar": [And(str, var_pattern)],
            "cyclic": [And(str, var_pattern)],
        },
        "cycle_counting_mode": Or('rainflow', 'streamflow', 'fastflow', only_one=True),
        #"compute_every": And(int)
    },
)

stress_model_schema = Schema(
    {
        "time": {
            "k_t": Or(float, And(int, Use(float))),
        },
        "soc": {
            "k_soc":Or(float, And(int, Use(float))),
            "soc_ref": Or(float, And(int, Use(float)))
        },
        "temperature": {
            "k_temp": Or(float, And(int, Use(float))),
            "temp_ref": Or(float, And(int, Use(float)))
        },
        "dod_bolun": {
            "k_delta1": Or(float, And(int, Use(float))),
            "k_delta2": Or(float, And(int, Use(float))),
            "k_delta3": Or(float, And(int, Use(float)))
        },
        Optional("dod_quadratic"): Or(float, And(int, Use(float))),
        Optional("dod_exponential"): Or(float, And(int, Use(float))),
    }
)

model_schema = Schema(
    {
        "type": And(str, var_pattern),
        "class_name": And(str, class_pattern),
        Optional("components"): Or(first_order_thevenin, second_order_thevenin, rc_thermal, r2c_thermal, mlp_thermal, bolun),
        Optional("stress_models"): stress_model_schema
    }
)

schemas['driven'] = config_schema
schemas['scheduled'] = config_schema
schemas['adaptive'] = config_schema
schemas['assets'] = assets_schema
schemas['model'] = model_schema


def _check_schema(yaml_dict: dict, schema_type: str):
    """
    
    
    Args:
        yaml_dict (dict): _description_
        schema_type (str): _description_
    """
    try:
        schemas[schema_type].validate(yaml_dict)
    except SchemaError as se:
        raise se


def read_yaml(yaml_file: str, yaml_type: str):
    """

    Args:
        yaml_file (str):
        yaml_type (str):

    Returns:

    """
    _file_types = ['assets', 'model', 'driven', 'scheduled', 'adaptive']

    if yaml_type not in _file_types:
        logger.error("The schema type '{}' of file {} is not existing!".format(yaml_type, yaml_file))
        exit(1)

    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)

    try:
        _check_schema(params, yaml_type)
    except SchemaError as se:
        logger.error("Error within the yaml file '{}': {}".format(yaml_file, se.args[0]))
        exit(1)

    return params


