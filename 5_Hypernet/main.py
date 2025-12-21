from gen_rays import get_ref_dirs, circle_points
import yaml

config_file = "configs.yaml"
with open(config_file) as stream:
    cfg = yaml.safe_load(stream)


ray_test = get_ref_dirs(cfg["N_OBJ"])
test_rays = circle_points(cfg["K"], min_angle=cfg["MIN_ANGLE"], max_angle=cfg["MAX_ANGLE"])