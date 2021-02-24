"""Run autopilot_env with IDM autopilot.

I.e. "autopilot" simply returns actual observations of the
controlled vehicle and actions calculated by IDMVehicle
"""
import bdgym.scripts.driver_assistant.utils as utils


if __name__ == "__main__":
    parser = utils.test_parser()
    args = utils.parse_parser(parser)
    utils.run(args)
