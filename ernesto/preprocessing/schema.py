from schema import Schema, SchemaError, Regex, And, Or, Optional, Use
import yaml
import logging

logger = logging.getLogger('ErNESTO-DT')

schemas = {}

string_pattern = Regex(r'^[a-zA-Z0-9_./]+$',
                       error="Error in string '{}': it can only have a-z, A-Z, 0-9, and _.")
path_pattern = Regex(r'^[a-zA-Z0-9_\-./]+$',
                     error="Error in path '{}': it can only have a-z, A-Z, 0-9, ., / and _.")
class_pattern = Regex(r'^[a-zA-Z0-9]+$',
                      error="Error in class name '{}': it can only have a-z, A-Z and 0-9.")
var_pattern = Regex(r'^[a-zA-Z_]+$',
                    error="Error in variable '{}': it can only have a-z and _.")
label_pattern = Regex(r'^[a-zA-Z0-9_\[\].() ]+$',
                      error="Error in label '{}': it can only have a-z, A-Z, [,], and _.")
unit_pattern = Regex(r'^[a-zA-Z_]+$',
                     error="Error in unit identifier '{}': it can only have a-z, A-Z.")

absolute_value = Or(float, And(int, Use(float)))
percentage_value = Or(float, And(int, Use(float)), lambda n: 0 <= n <= 1)

ground_data = Schema(
    {
        # Ground data structure
        "file": And(str, path_pattern),
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
        "init": {And(str, var_pattern): Or(absolute_value, percentage_value)},
        Optional("reset_soc_every"): Or(int, None)
    }
)

optimizer = Schema(
    {
        "algorithm": Or(str, None),
        "max_iter": Or(int, None),
        "disp": Or(bool, None),
        "tol": Or(float, int, None),
        "ftol": Or(float, int, None),
        "alpha": Or(float, int, None),
        "beta": Or(float, int, None),
        "batch_size": Or(int, None),
        "n_guesses": Or(int, None),
        "n_jobs": Or(int, None),
        "search_bounds": {And(str, label_pattern): bound_param},
        "scale_factors": {And(str, label_pattern): Or(float, int)},
    }
)

cluster_config = Schema(
    {
        "original_csv": Or(And(str, path_pattern), None),
        Optional("destination_file"): Or(And(str, path_pattern), None),
        "name": And(str, label_pattern),
        Optional("region") : {
            And(str, var_pattern): bound_param},    
    }
)

parameter_space = Schema(
    {
        "domain_variables": [And(str, label_pattern)],
        "param_variables": [And(str, label_pattern)],
        "clusters": [cluster_config]
    }
)

adaptation = Schema(
    {   
        "param_names": [And(str, label_pattern)],
        "threshold": Or(float, int),
    }
)

config_schema = Schema(
    {
        # Summary
        Optional("experiment_name"): And(str),
        Optional("description"): And(str),
        Optional("goal"): And(str),
        "destination_folder": And(str, path_pattern),
        
        # Ground data options
        "input": {
            Optional("ground_data"): ground_data,
            Optional("schedule"): schedule,
        },
        # Simulation options
        Optional("start_at"): Or(int, None),
        Optional("iterations"): Or(int, None),
        Optional("timestep"): Or(int, float, None),
        Optional("check_soh_every"): Or(int, None),
        Optional("clear_collections_every"): Or(int, None),
        
        # Adaptation options
        Optional("adaptation"): adaptation,
        Optional("optimizer"): optimizer,
        Optional("parameter_space"): parameter_space,
        
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

hardcoded_lookup = Schema(
    {
        "selected_type": Or('scalar', 'lookup'),
        Optional("scalar"): Or(float, And(int, Use(float))),
        Optional("lookup"): {
            "inputs": Or(absolute_value, percentage_value),
            "output": absolute_value
        }
    },
)

csv_lookup = Schema(
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

mlp_thermal = Schema(
    {  # RC_thermal
        "input_size": int,
        "hidden_size": int,
        "output_size": int,
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

physical_model = Schema({And(str, label_pattern): Or(csv_lookup, hardcoded_lookup)})
data_driven_model = Schema({And(str, label_pattern): mlp_thermal})

model_schema = Schema(
    {
        "type": And(str, var_pattern),
        "class_name": And(str, class_pattern),
        Optional("components"): Or(physical_model, data_driven_model, bolun),
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


